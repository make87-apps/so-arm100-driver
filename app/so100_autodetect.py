from pathlib import Path
from typing import Dict, Union, Tuple
import cv2

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

def get_camera_info(index_or_path: Union[int, str, Path]) -> Tuple[int, int, float]:
    cap = cv2.VideoCapture(str(index_or_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera at {index_or_path}")

    # Read a frame to make sure the stream is active
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise RuntimeError(f"Failed to read from camera at {index_or_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    cap.release()
    return width, height, fps


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
    policy_device: str = "cpu",
    actions_per_chunk: int = 10,
) -> RobotClientConfig:
    # 1) Build the robot & camera config
    port = find_so100_port(index=index)

    cameras = dict()
    for name, path in camera_paths.items():
        w, h, fps = get_camera_info(index_or_path=path)
        cameras[name] = OpenCVCameraConfig(
                index_or_path=Path(path),
                width=w, height=h, fps=fps
            )

    robot_cfg = SO100FollowerConfig(
        port=port,
        id=f"so100-{index}",
        cameras=cameras
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