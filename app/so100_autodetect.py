from pathlib import Path
from typing import Dict

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so100_follower import SO100FollowerConfig
from lerobot.scripts.server.configs import RobotClientConfig
from serial.tools import list_ports

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


def get_so100_config(
    server_address: str,
    camera_paths: Dict[str, str],
    task: str = "",
    index: int = 0,
    policy_type: str = "smolvla",
    pretrained_name_or_path: str = "helper2424/smolvla_rtx_movet",
    policy_device: str = "cuda:0",
    fps: int = 10,
    actions_per_chunk: int = 10,
) -> RobotClientConfig:
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
        debug_visualize_queue_size=False,
        verify_robot_cameras=False,
    )
    return cfg