from pathlib import Path
from typing import Optional, Dict
import logging

from lerobot.cameras.opencv import OpenCVCameraConfig
from serial.tools import list_ports
from lerobot.robots.so100_follower import SO100FollowerConfig
from lerobot.scripts.server.configs import RobotClientConfig
from lerobot.scripts.server.robot_client import async_client


import threading
from pprint import pformat
from dataclasses import asdict

from lerobot.scripts.server.robot_client import RobotClient
from lerobot.datasets.utils import hw_to_dataset_features



SO100_USB_VID = 6790      
KNOWN_SO100_PIDS = {
    29987,
    21987,
    21795,
    21797,
    21971
}


def find_so100_port(index: int = 0) -> str:
    """Scan for SO-100 USB serial devices and return the port at the given index."""
    all_ports = list(list_ports.comports())
    matching_ports = [
        p.device
        for p in all_ports
        if (p.vid is not None and p.pid is not None) and
           (int(p.vid) == SO100_USB_VID or int(p.pid) in KNOWN_SO100_PIDS)
    ]
    if not matching_ports:
        raise RuntimeError("No SO-100 USB device found.")
    if index >= len(matching_ports):
        raise RuntimeError(f"Requested index {index}, but only {len(matching_ports)} SO-100 device(s) found.")
    return matching_ports[index]


def start_so100_robot_client(
    server_address: str,
    task: str,
    camera_paths: Dict[str, str],
    index: int = 0,
    policy_type: str = "smolvla",
    pretrained_name_or_path: str = "helper2424/smolvla_rtx_movet",
    policy_device: str = "cuda:0",
    fps: int = 10,
    actions_per_chunk: int = 10,
):
    # 1) Build the robot & camera config
    port = find_so100_port(index=index)
    robot_cfg = SO100FollowerConfig(
        port=port,
        id=f"so100-{index}",
        cameras={
            name: OpenCVCameraConfig(
                index_or_path=Path(path),
                width=1920, height=1080, fps=fps
            )
            for name, path in camera_paths.items()
        },
    )

    # 4) Build the full client config
    cfg = RobotClientConfig(
        robot=robot_cfg,
        server_address=server_address,
        task=task,
        policy_type=policy_type,
        pretrained_name_or_path=pretrained_name_or_path,
        policy_device=policy_device,
        fps=fps,
        actions_per_chunk=actions_per_chunk,
        #lerobot_features=lerobot_features,  # <— inject the patched features
        debug_visualize_queue_size=False,
        verify_robot_cameras=False,
    )

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting RobotClient with config:\n%s", pformat(asdict(cfg)))

    # 5) Instantiate and start
    client = RobotClient(cfg)
    #  client.policy_config.lerobot_features = {k.replace("observation.images.image", "observation.image"): v for k, v in client.policy_config.lerobot_features.items()}
    if not client.start():
        raise RuntimeError("Failed to start RobotClient!")

    # 6) Start thread to receive actions
    recv_thread = threading.Thread(target=client.receive_actions, daemon=True)
    recv_thread.start()

    try:
        # 7) Main control loop blocks here
        client.control_loop(task)
    finally:
        client.stop()
        recv_thread.join()
        if cfg.debug_visualize_queue_size:
            client.visualize_action_queue_size()
        logging.info("Robot client stopped.")
