import enum
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Optional, Tuple

import base64
import cv2
import mcp
from mcp.server.fastmcp.utilities.types import Image
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.constants import HF_LEROBOT_CALIBRATION, ROBOTS
from lerobot.robots import Robot
from lerobot.robots.so100_follower import SO100FollowerEndEffectorConfig, SO100FollowerEndEffector
from lerobot.teleoperate import TeleoperateConfig
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig
from lerobot.utils.robot_utils import busy_wait

from app.so100_autodetect import find_so100_port, get_camera_info


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


        # --- goal-based absolute pose driving (via delta interface) ---
        self._goal_lock = threading.Lock()
        self._goal_xyz: Optional[Tuple[float, float, float]] = None
        self._pos_tol = 0.005          # 5 mm tolerance to stop
        self._max_step_per_tick = 0.25 # error normalization scale → maps to [-1,1]

        # Build presets from bounds, with sane defaults
        xmin, ymin, zmin = self.robot.config.end_effector_bounds["min"]
        xmax, ymax, zmax = self.robot.config.end_effector_bounds["max"]
        xc = 0.5 * (xmin + xmax)
        yc = 0.5 * (ymin + ymax)

        table_z = None
        try:
            cal = getattr(self.robot, "calibration", None)
            if isinstance(cal, dict):
                table_z = cal.get("table_z", None)
        except Exception:
            table_z = None
        safe_z_hover = (table_z + 0.05) if isinstance(table_z, (int, float)) else max(zmin + 0.03, zmin + 0.02)

        self._presets: Dict[str, Tuple[float, float, float]] = {
            "rest": (
                max(xmin + 0.08, xmin),
                max(ymin + 0.12, ymin),
                min(zmax * 0.6, zmax - 0.02),
            ),
            "top_down": (
                xc,
                min(yc + 0.10, ymax - 0.02),
                min(zmax - 0.03, zmax),
            ),
            "work_hover": (
                xc,
                min(yc + 0.20, ymax - 0.02),
                max(safe_z_hover, zmin + 0.03),
            ),
        }

        @server.tool(description="Move the arm by a delta in x, y, z (FLU coordinates) and open/close the gripper. Values are in [-1, 1].")
        def move_arm_vector(delta_forward: float, delta_left: float, delta_up: float, gripper: float) -> bool:
            #self.delta_queue.put((delta_forward, delta_left, delta_up, gripper + 1))  # gripper in [0, 2]
            self.delta_queue.put(
                {
                    "delta_x": delta_forward, 
                    "delta_y": delta_left, 
                    "delta_z": delta_up, 
                    "gripper": gripper
                }
            )
            return True


        @server.tool(description="Get the current image from the robot's gripper camera as jpeg bytes.")
        def get_gripper_image() -> Image:
            observation = self.robot.get_observation()
            if "gripper" in observation:
                image = observation["gripper"]
                jpegimg = cv2.imencode(".jpg", image[...,::-1])[1].tobytes()
            else:
                jpegimg = b""
            return Image(data=jpegimg, format="jpeg")

        @server.tool(description="Move the end-effector to a preset pose.")
        def move_to_preset(preset: ArmPreset) -> bool:
            target = self._presets.get(preset.value)
            if target is None:
                return False
            with self._goal_lock:
                self._goal_xyz = self._clamp_to_bounds(target)
            return True

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
            "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
        }

    def get_action(self) -> Dict[str, Any]:
        if not self.is_connected:
            raise RuntimeError(
                "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()

        action_dict = {
            "delta_x": 0.0,
            "delta_y": 0.0,
            "delta_z": 0.0,
            "gripper": 1.0,
        }

        # Generate action based on current key states
        for val in self.current_pressed:
            action_dict["delta_x"] += val["delta_x"]
            action_dict["delta_y"] += val["delta_y"]
            action_dict["delta_z"] += val["delta_z"]
            action_dict["gripper"] = val["gripper"]  # just take the last

        self.current_pressed.clear()



        return action_dict

    


def teleop_loop(
        teleop: Teleoperator, robot: Robot, fps: int, display_data: bool = False, duration: float | None = None
):
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        action = teleop.get_action()
        if display_data:
            observation = robot.get_observation()
        try:
            robot.send_action(action)
        except ...:
            pass
        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)

        loop_s = time.perf_counter() - loop_start

        print("\n" + "-" * (display_len + 10))
        print(f"{'NAME':<{display_len}} | {'NORM':>7}")
        for motor, value in action.items():
            print(f"{motor:<{display_len}} | {value:>7.2f}")
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        if duration is not None and time.perf_counter() - start >= duration:
            return


def teleoperate(camera_paths: Dict[str, str],
                index: int = 0,
                calibration: Optional[Dict] = None, ):
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
            "x": 0.1,
            "y": 0.1,
            "z": 0.1,
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

    robot = SO100FollowerEndEffector(config=robot_cfg)
    teleop = MCPEndEffectorTeleop(config=MCPTeleopConfig(), robot=robot)

    teleop.connect()
    robot.connect(calibrate=False)
    # just use default
    cfg = TeleoperateConfig(
        robot=robot_cfg,
        teleop=teleop.config
    )

    try:
        teleop_loop(teleop, robot, cfg.fps, display_data=cfg.display_data, duration=cfg.teleop_time_s)
    except KeyboardInterrupt:
        pass
    finally:
        # if cfg.display_data:
        #     rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()
