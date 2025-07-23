import time
from datetime import datetime, timezone

import make87
import logging

from lerobot.scripts.server.helpers import Observation
from make87.interfaces.zenoh import ZenohInterface
from make87.peripherals import CameraPeripheral
from make87_messages.image.compressed.image_jpeg_pb2 import ImageJPEG
import make87 as m87

from app.robot_client import CustomRobotClient
from app.so100_autodetect import get_so100_config
import cv2

logger = logging.getLogger(__name__)


def main():
    config = make87.config.load_config_from_env()
    lerobot_interface = config.interfaces.get("lerobot")
    if not lerobot_interface:
        logger.warning("No lerobot found in interface configuration.")
        return

    robotclient = lerobot_interface.clients.get("robotclient")
    if not robotclient:
        logger.warning("No robotclient client found in configuration.")
        return

    robot_index = make87.config.get_config_value(config, "robot_index", default=0, converter=int)
    fps = make87.config.get_config_value(config, "fps", default=30, converter=int)
    actions_per_chunk = make87.config.get_config_value(config, "actions_per_chunk", default=10, converter=int)
    pretrained_name_or_path = make87.config.get_config_value(config, "pretrained_name_or_path",
                                                             default="helper2424/smolvla_rtx_movet")

    manager = make87.peripherals.manager.PeripheralManager(make87_config=config)
    camera: CameraPeripheral = manager.get_peripheral_by_name("CAMERA")

    zenoh_interface = ZenohInterface(name="zenoh-client", make87_config=config)
    action_publisher = zenoh_interface.get_publisher(name="AGENT_LOGS")
    camera_publisher = zenoh_interface.get_publisher(name="CAMERA_IMAGE")
    agent_chat_provider = zenoh_interface.get_queryable(name="AGENT_CHAT")

    server_address = f"{robotclient.vpn_ip}:{robotclient.vpn_port}"

    cam_name = "front"

    def on_observation_callback(observation: Observation):
        if not observation:
            return
        frame = observation["front"]
        try:
            ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            msg = ImageJPEG(data=jpeg.tobytes())
            message_encoded = m87.encodings.ProtobufEncoder(message_type=ImageJPEG).encode(msg)
            camera_publisher.put(message_encoded)
        except Exception as e:
            logger.error(f"Error sending image: {e}")
            return

    def on_action_callback(action):
        pass

    def get_next_task():
        """Retrieve the next task from the agent chat provider."""
        try:
            prompt = agent_chat_provider.recv()
            return prompt.payload.to_bytes().decode("utf-8")
        except Exception as e:
            logger.error(f"Error receiving task: {e}")
            return ""

    robot_config = get_so100_config(
        server_address=server_address,
        fps=fps,
        actions_per_chunk=actions_per_chunk,
        pretrained_name_or_path="helper2424/smolvla_check_async",
        index=robot_index,
        camera_paths={
            cam_name: camera.reference
        },
    )

    CustomRobotClient.run_robot_client(
        get_next_task=get_next_task,
        robot_config=robot_config,
        on_action_callback=on_action_callback,
        on_observation_callback=on_observation_callback,
    )


if __name__ == "__main__":
    main()
