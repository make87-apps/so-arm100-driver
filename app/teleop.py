import enum
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Optional

import cv2
import mcp
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


class ArmAction(enum.Enum):
    up = "up"
    down = "down"
    left = "left"
    right = "right"
    forward = "forward"
    backward = "backward"
    gripper_open = "gripper_open"
    gripper_close = "gripper_close"
    gripper_stay = "gripper_close"


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
        self.event_queue = Queue()
        self.current_pressed = {}
        self.logs = {}
        server = mcp.server.FastMCP(name="image_describer", host="0.0.0.0", port=config.port)
        self._server_thread = None
        self.server = server

        @server.tool(description="Move the arm by a certain amount in a direction.")
        def move_arm(action: ArmAction, value: int = 1):
            def move_arm_impl():
                self.event_queue.put((action.value, value))
                time.sleep(0.1)
                self.event_queue.put((action.value, value))

            return move_arm_impl

        @server.tool(description="Get the current image from the robot's gripper camera as jpeg bytes.")
        def get_gripper_image() -> bytes:
            observation = self.robot.get_observation()
            if "gripper" in observation:
                image = observation["gripper"]
                jpegimg = cv2.imencode(".jpg", image)[1].tobytes()
            else:
                jpegimg = b""
            return jpegimg

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
        while not self.event_queue.empty():
            key_char, val = self.event_queue.get_nowait()
            if key_char in self.current_pressed:
                self.current_pressed[key_char] += val
            else:
                self.current_pressed[key_char] = val

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
        delta_x = 0.0
        delta_y = 0.0
        delta_z = 0.0
        gripper_action = 1.0

        # Generate action based on current key states
        for key, val in self.current_pressed.items():
            if key == ArmAction.up:
                delta_y = -int(val)
            elif key == ArmAction.down:
                delta_y = int(val)
            elif key == ArmAction.left:
                delta_x = int(val)
            elif key == ArmAction.right:
                delta_x = -int(val)
            elif key == ArmAction.forward:
                delta_z = -int(val)
            elif key == ArmAction.backward:
                delta_z = int(val)
            elif key == ArmAction.gripper_open:
                # Gripper actions are expected to be between 0 (close), 1 (stay), 2 (open)
                gripper_action = 2
            elif key == ArmAction.gripper_close:
                gripper_action = 0
            elif key == ArmAction.gripper_stay:
                gripper_action = 1

        self.current_pressed.clear()

        action_dict = {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
            "gripper": gripper_action,
        }

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

        robot.send_action(action)
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
        cameras=cameras
    )
    if calibration:
        calibration_file = (
                HF_LEROBOT_CALIBRATION / ROBOTS / "so100_follower" / f"{robot_cfg.id}.json"
        )
        calibration_file.parent.mkdir(parents=True, exist_ok=True)
        with calibration_file.open("w") as f:
            json.dump(calibration, f, indent=4)

    robot = SO100FollowerEndEffector(config=robot_cfg)
    teleop = MCPEndEffectorTeleop(config=MCPTeleopConfig(), robot=robot)

    teleop.connect()
    robot.connect()
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
