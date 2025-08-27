import logging
import os

import make87
import make87 as m87
from make87.interfaces.zenoh import ZenohInterface
from make87.peripherals import CameraPeripheral
from make87_messages.image.compressed.image_jpeg_pb2 import ImageJPEG
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def run_policy_controlled():
    from lerobot.scripts.server.helpers import Observation
    from make87.interfaces.zenoh import ZenohInterface
    from app.robot_client import CustomRobotClient, get_so100_policy_config

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
    actions_per_chunk = make87.config.get_config_value(config, "actions_per_chunk", default=10, converter=int)
    pretrained_name_or_path = make87.config.get_config_value(config, "pretrained_name_or_path",
                                                             default="rtsmc/smolvla_box_in_bin_so101_test")
    policy_type = make87.config.get_config_value(config, "policy_type",
                                                             default="smolvla")
    camera_1_name = make87.config.get_config_value(config, "camera_1_name", default="front")
    camera_2_name = make87.config.get_config_value(config, "camera_2_name", default="wrist")
    calibration = make87.config.get_config_value(config, "calibration")

    manager = make87.peripherals.manager.PeripheralManager(make87_config=config)
    try:
        camera_1: CameraPeripheral = manager.get_peripheral_by_name("CAMERA_1")
    except KeyError:
        camera_1 = None

    try:
        camera_2: CameraPeripheral = manager.get_peripheral_by_name("CAMERA_2")
    except KeyError:
        camera_2 = None

    zenoh_interface = ZenohInterface(name="zenoh-client", make87_config=config)
    action_publisher = zenoh_interface.get_publisher(name="AGENT_LOGS")
    camera_1_publisher = zenoh_interface.get_publisher(name="CAMERA_1_IMAGE")
    camera_2_publisher = zenoh_interface.get_publisher(name="CAMERA_2_IMAGE")
    agent_chat_provider = zenoh_interface.get_queryable(name="AGENT_CHAT")

    server_address = f"{robotclient.vpn_ip}:{robotclient.vpn_port}"

    def on_observation_callback(observation: Observation):
        if not observation:
            return
        if camera_1_name in observation:
            frame = observation[camera_1_name]
            try:
                ret, jpeg = cv2.imencode(".jpg", frame[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
                msg = ImageJPEG(data=jpeg.tobytes())
                message_encoded = m87.encodings.ProtobufEncoder(message_type=ImageJPEG).encode(msg)
                camera_1_publisher.put(message_encoded)
            except Exception as e:
                logger.error(f"Error sending image: {e}")
                return
        if camera_2_name in observation:
            frame = observation[camera_2_name]
            try:
                ret, jpeg = cv2.imencode(".jpg", frame[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
                msg = ImageJPEG(data=jpeg.tobytes())
                message_encoded = m87.encodings.ProtobufEncoder(message_type=ImageJPEG).encode(msg)
                camera_2_publisher.put(message_encoded)
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

    camera_paths = dict()
    if camera_1:
        camera_paths[camera_1_name] = camera_1.reference
    if camera_2:
        camera_paths[camera_2_name] = camera_2.reference
    robot_config = get_so100_policy_config(
        server_address=server_address,
        actions_per_chunk=actions_per_chunk,
        policy_type=policy_type,
        pretrained_name_or_path=pretrained_name_or_path,
        index=robot_index,
        camera_paths=camera_paths,
    )

    CustomRobotClient.run_robot_client(
        get_next_task=get_next_task,
        robot_config=robot_config,
        on_action_callback=on_action_callback,
        on_observation_callback=on_observation_callback,
        calibration=calibration,
    )


def run_teleop():
    from app.teleop import teleoperate
    config = make87.config.load_config_from_env()

    robot_index = make87.config.get_config_value(config, "robot_index", default=0, converter=int)
    calibration = make87.config.get_config_value(config, "calibration")

    manager = make87.peripherals.manager.PeripheralManager(make87_config=config)
    camera_1: CameraPeripheral = manager.get_peripheral_by_name("CAMERA_1")
    zenoh_interface = ZenohInterface(name="zenoh-client", make87_config=config)
    camera_1_publisher = zenoh_interface.get_publisher(name="IMAGE")

    def on_new_image(img: np.ndarray):
        try:
            ret, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            msg = ImageJPEG(data=jpeg.tobytes())
            message_encoded = m87.encodings.ProtobufEncoder(message_type=ImageJPEG).encode(msg)
            camera_1_publisher.put(message_encoded)
        except Exception as e:
            logger.error(f"Error sending image: {e}")
            return

    teleoperate(camera_paths={"gripper": camera_1.reference},
                index=robot_index,
                calibration=calibration,
                on_new_image=on_new_image)



if __name__ == "__main__":
    if os.environ.get("TELEOP", None) is None:
        run_policy_controlled()
    else:
        run_teleop()
