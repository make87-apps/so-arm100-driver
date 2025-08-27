import enum
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Optional, Tuple, Callable

import base64
import cv2
import mcp
import numpy as np
import math
import random
from mcp.server.fastmcp.utilities.types import Image
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.constants import HF_LEROBOT_CALIBRATION, ROBOTS
from lerobot.robots import Robot
from lerobot.motors import MotorCalibration
from lerobot.robots.so100_follower import SO100FollowerEndEffectorConfig, SO100FollowerEndEffector
from lerobot.teleoperate import TeleoperateConfig
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig
from lerobot.utils.robot_utils import busy_wait


from app.so100_autodetect import find_so100_port, get_camera_info
from app.so100_fpv_follower import SO100FPVFollower


@TeleoperatorConfig.register_subclass("mcp")
@dataclass
class MCPTeleopConfig(TeleoperatorConfig):
    port: int = 9988



class ArmPreset(enum.Enum):
    rest = "rest"
    top_down = "top_down"
    work_hover = "work_hover"


class MCPEndEffectorTeleop(Teleoperator):
    """
    Teleop class to use mcp inputs for end effector control.
    Designed to be used with the `So100FollowerEndEffector` robot.
    """
    config_class = MCPTeleopConfig
    name = "mcp"

    def __init__(self, config: MCPTeleopConfig, robot: Robot):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type
        self.robot = robot
        self.delta_queue = Queue()
        self.current_pressed = []
        self.logs = {}
        server = mcp.server.FastMCP(name="image_describer", host="0.0.0.0", port=config.port)
        self._server_thread = None
        self.server = server


        self._goal_lock = threading.Lock()
        self._goal_xyz: Optional[Tuple[float, float, float]] = None
        self._pos_tol = 0.005           # 5 mm
        self._max_step_per_tick = 0.25  # maps error meters → [-1, 1] deltas

        self.last_delta = None
        self.random_move_time = time.time()

        @server.tool(description="Move the arm by setting the deltas. Values are in [-1, 1]. delta_pitch and delta_yaw and delta_roll are in degrees.")
        def move_arm_vector(delta_forward: float, delta_left: float, delta_up: float, gripper: float, delta_pitch: float = 0.0, delta_yaw: float = 0.0, delta_roll: float = 0.0) -> str:
            #self.delta_queue.put((delta_forward, delta_left, delta_up, gripper + 1))  # gripper in [0, 2]
            self.delta_queue.put(
                {
                    "delta_x": delta_forward, 
                    "delta_y": delta_left, 
                    "delta_z": delta_up, 
                    "gripper": gripper,
                    "delta_pitch": delta_pitch,
                    "delta_yaw": delta_yaw,
                    "delta_roll": delta_roll
                }
            )
            time.sleep(1)

            return "Move arm command succelfully sent"


        @server.tool(description="Get the current image from the robot's gripper camera as jpeg bytes.")
        def get_gripper_image() -> Image:
            observation = self.robot.get_observation()
            if "gripper" in observation:
                image = observation["gripper"]
                jpegimg = cv2.imencode(".jpg", image[...,::-1])[1].tobytes()
            else:
                jpegimg = b""
            return Image(data=jpegimg, format="jpeg")


    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._server_thread is not None and self._server_thread.is_alive()

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        # self.server.run(transport="streamable-http")
        # run server in thread
        if not self.is_connected:
            thr = threading.Thread(target=self.server.run, kwargs={"transport": "streamable-http"}, daemon=True)
            thr.start()
            self._server_thread = thr

    def calibrate(self) -> None:
        pass

    def _drain_pressed_keys(self):
        while not self.delta_queue.empty():
            vals = self.delta_queue.get_nowait()
            self.current_pressed.append(vals)

        if time.time() - self.random_move_time > 1.5 and False:
            if self.last_delta:
                self.delta_queue.put({
                        "delta_x": -self.last_delta["delta_x"], 
                        "delta_y": -self.last_delta["delta_y"], 
                        "delta_z": -self.last_delta["delta_z"],
                        "gripper": 0.,
                        "delta_pitch": -self.last_delta.get("delta_pitch", 0.0),
                        "delta_yaw": -self.last_delta.get("delta_yaw", 0.),
                        "delta_roll": -self.last_delta.get("delta_roll", 0.)
                    })
            delta = {
                        "delta_x": random.random() * 0.01, 
                        "delta_y": random.random() * 0.01, 
                        "delta_z": random.random() * 0.01,
                        "gripper": 0.,
                        "delta_pitch": random.random() * 90, 
                        "delta_yaw": random.random() * 90,
                        "delta_roll": random.random() * 15,
                    }
            self.delta_queue.put(delta)
            self.last_delta = delta
            self.random_move_time = time.time()

    def configure(self):
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if self.is_connected:
            if self._server_thread is not None:
                self._server_thread.join(timeout=1)
                self._server_thread = None

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (4,),
            "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2,"gripper": 3, "delta_pitch": 4, "delta_yaw": 5, "delta_roll": 5},
        }

    def get_action(self) -> Dict[str, Any]:
        if not self.is_connected:
            raise RuntimeError("KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`.")

        self._drain_pressed_keys()
        action_dict = {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0, "gripper": 1.0, "delta_pitch": 0.0, "delta_yaw": 0.0, "delta_roll": 0.0}
        for val in self.current_pressed:
            action_dict["delta_x"] += val["delta_x"]
            action_dict["delta_y"] += val["delta_y"]
            action_dict["delta_z"] += val["delta_z"]
            action_dict["gripper"] = val["gripper"]
            action_dict["delta_pitch"] += val["delta_pitch"]
            action_dict["delta_yaw"] += val["delta_yaw"]
            action_dict["delta_roll"] += val["delta_roll"]
        self.current_pressed.clear()
        return action_dict




def teleop_loop(
        teleop: Teleoperator, robot: Robot, fps: int, display_data: bool = False, duration: float | None = None,
        on_new_image: Optional[Callable[[], np.ndarray]] = None
):
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        action = teleop.get_action()
        
        if on_new_image:
            observation = robot.get_observation()
            if "gripper" in observation:
                image = observation["gripper"]
                on_new_image(image[...,::-1])

        try:
            robot.send_action(action)
        except Exception as e:
            print(e)
            pass
        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)

        loop_s = time.perf_counter() - loop_start

        #print("\n" + "-" * (display_len + 10))
        #print(f"{'NAME':<{display_len}} | {'NORM':>7}")
        #for motor, value in action.items():
        #    print(f"{motor:<{display_len}} | {value:>7.2f}")
        #print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        if duration is not None and time.perf_counter() - start >= duration:
            return


def teleoperate(camera_paths: Dict[str, str],
                index: int = 0,
                calibration: Optional[Dict] = None, 
                on_new_image: Optional[Callable[[], np.ndarray]] = None,
                ):
    port = find_so100_port(index=index)

    cameras = dict()
    for name, path in camera_paths.items():
        w, h, fps = get_camera_info(index_or_path=path)
        cameras[name] = OpenCVCameraConfig(
            index_or_path=Path(path),
            width=w, height=h, fps=fps
        )
        
    robot_cfg = SO100FollowerEndEffectorConfig(
        port=port,
        id=f"so100-{index}",
        cameras=cameras,
        end_effector_step_sizes={
            "x": 0.05,
            "y": 0.05,
            "z": 0.05,
        },
        end_effector_bounds={
            "min": [-0.40, -0.50, 0.02],  # X/Y limits over your table # 2 cm above the surface to avoid scraping
            "max": [ 0.40,  0.60, 0.45],               # and a sane Z-max
        },
        urdf_path=str(Path(__file__).parent.absolute() / "so101_new_calib.urdf")
    )
    if calibration:
        calibration_file = (
                HF_LEROBOT_CALIBRATION / ROBOTS / "so100_follower" / f"{robot_cfg.id}.json"
        )
        robot_cfg.calibration_dir = calibration_file.parent
        calibration_file.parent.mkdir(parents=True, exist_ok=True)
        with calibration_file.open("w") as f:
            json.dump(calibration, f, indent=4)

    robot = SO100FPVFollower(config=robot_cfg)
    teleop = MCPEndEffectorTeleop(config=MCPTeleopConfig(), robot=robot)


    teleop.connect()
    robot.connect(calibrate=False)
    # just use default
    cfg = TeleoperateConfig(
        robot=robot_cfg,
        teleop=teleop.config
    )

    try:
        teleop_loop(teleop, robot, cfg.fps, display_data=cfg.display_data, duration=cfg.teleop_time_s, on_new_image=on_new_image)
    except KeyboardInterrupt:
        pass
    finally:
        # if cfg.display_data:
        #     rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()
