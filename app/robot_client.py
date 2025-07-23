import logging
import threading
import time
from dataclasses import asdict
from pprint import pformat
from typing import Dict, Callable

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.scripts.server.configs import RobotClientConfig
from lerobot.scripts.server.helpers import (
    Action,
    Observation,
)
from lerobot.scripts.server.robot_client import RobotClient


class CustomRobotClient(RobotClient):

    def __init__(self, config: RobotClientConfig,
                 on_observation_callback: Callable[[Observation], None],
                 on_action_callback: Callable[[Dict], None],
                 ):
        super().__init__(config)
        self.on_observation_callback = on_observation_callback
        self.on_action_callback = on_action_callback
        self.pause = threading.Event()
        self.current_task = config.task

    def custom_control_loop(self, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        while self.running:
            if self.pause.is_set():
                self.logger.info("Control loop paused, waiting for resume...")
                self.pause.wait()

            control_loop_start = time.perf_counter()
            """Control loop: (1) Performing actions, when available"""
            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)
                self.on_action_callback(_performed_action)

            """Control loop: (2) Streaming observations to the remote policy server"""
            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(self.current_task, verbose)
                self.on_observation_callback(_captured_observation)

            self.logger.info(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action

    def pause_control_loop(self):
        """Pause the control loop"""
        self.pause.set()
        self.logger.info("Control loop paused")

    def continue_control_loop(self):
        """Resume the control loop"""
        self.pause.clear()
        self.logger.info("Control loop resumed")

    @staticmethod
    def run_robot_client(
        robot_config: RobotClientConfig,
        on_observation_callback: Callable[[Observation], None],
        on_action_callback: Callable[[Dict], None],
        get_next_task: Callable[[], str],
    ):

        logging.basicConfig(level=logging.INFO)
        logging.info("Starting RobotClient with config:\n%s", pformat(asdict(robot_config)))

        # 5) Instantiate and start
        client = CustomRobotClient(
            config=robot_config,
            on_observation_callback=on_observation_callback,
            on_action_callback=on_action_callback,
        )
        #  client.policy_config.lerobot_features = {k.replace("observation.images.image", "observation.image"): v for k, v in client.policy_config.lerobot_features.items()}
        if not client.start():
            raise RuntimeError("Failed to start RobotClient!")

        client.pause_control_loop()

        # 6) Start thread to receive actions
        recv_thread = threading.Thread(target=client.receive_actions, daemon=True)
        recv_thread.start()

        control_loop = threading.Thread(
            target=client.custom_control_loop,
            daemon=True
        )
        control_loop.start()

        run = True
        try:
            while run:
                # Check for new task.. blocking wait
                new_task = get_next_task()
                if new_task == "exit" or new_task == "quit" or new_task == "stop":
                    client.pause_control_loop()
                else:
                    client.current_task = new_task
                    client.continue_control_loop()
        finally:
            client.stop()
            recv_thread.join()
            control_loop.join()
            logging.info("Robot client stopped.")
