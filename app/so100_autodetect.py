from typing import Optional, Dict
import logging

from serial.tools import list_ports
from lerobot.robots.so100_follower.configuration_so100 import SO100FollowerConfig
from lerobot.scripts.server.configs import RobotClientConfig
from lerobot.scripts.server.robot_client import async_client


SO100_USB_VID = 0x1A86
KNOWN_SO100_PIDS = {0x7523, 0x55E3, 0x5523, 0x5525}


def find_so100_port(index: int = 0) -> str:
    """Scan for SO-100 USB serial devices and return the port at the given index."""
    matching_ports = [
        p.device
        for p in list_ports.comports()
        if p.vid == SO100_USB_VID and p.pid in KNOWN_SO100_PIDS
    ]
    if not matching_ports:
        raise RuntimeError("No SO-100 USB device found.")
    if index >= len(matching_ports):
        raise RuntimeError(f"Requested index {index}, but only {len(matching_ports)} SO-100 device(s) found.")
    return matching_ports[index]


def start_so100_robot_client(
    server_address: str,
    camera_paths: Dict[str, str],
    index: int = 0,
    task: str = "default",
    policy_type: str = "smolvla",
    pretrained_name_or_path: str = "lerobot/smolvla_base",
    policy_device: str = "cpu",
    fps: int = 10,
    actions_per_chunk: int = 10,
    environment_dt: float = 0.1,
):
    """
    Auto-detect a SO-100 robot and start the gRPC client that connects to a remote policy server.
    """
    port = find_so100_port(index=index)
    print(f"[INFO] Found SO-100 on port: {port}")

    config = RobotClientConfig(
        robot=SO100FollowerConfig(
            type="so100_follower",
            port=port,
            cameras={k: {
                "type": "opencv",
                "width": 1920,
                "height": 1080,
                "fps": 30,
                "index_or_path": v
            } for k, v in camera_paths.items()},
            id=f"so100-{index}",
        ),
        server_address=server_address,
        policy_type=policy_type,
        pretrained_name_or_path=pretrained_name_or_path,
        policy_device=policy_device,
        fps=fps,
        actions_per_chunk=actions_per_chunk,
        chunk_size_threshold=0.5,
        aggregate_fn_name="latest",
        task=task,
        debug_visualize_queue_size=False,
        verify_robot_cameras=False,
        environment_dt=environment_dt,
    )

    logging.basicConfig(level=logging.INFO)
    async_client(config)
