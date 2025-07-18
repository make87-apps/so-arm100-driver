import time
from datetime import datetime, timezone

import make87
import logging

from make87.interfaces.zenoh import ZenohInterface
from make87.peripherals import CameraPeripheral

from app.so100_autodetect import start_so100_robot_client

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
    fps = make87.config.get_config_value(config, "fps", default=10, converter=int)
    actions_per_chunk = make87.config.get_config_value(config, "actions_per_chunk", default=10, converter=int)

    manager = make87.peripherals.manager.PeripheralManager(make87_config=config)
    camera: CameraPeripheral = manager.get_peripheral_by_name("CAMERA")


    zenoh_interface = ZenohInterface(name="zenoh-client", make87_config=config)
    action_publisher = zenoh_interface.get_publisher(name="AGENT_LOGS")
    agent_chat_provider = zenoh_interface.get_provider(name="AGENT_CHAT")

    server_address = f"{robotclient.vpn_ip}:{robotclient.vpn_port}"

    while True:
        try:
            prompt = agent_chat_provider.recv()
            text_prompt = prompt.payload.to_bytes().decode("utf-8")
        except Exception as e:
            logger.error(f"Error receiving prompt: {e}")
            time.sleep(1)
            continue

        start_so100_robot_client(
            task=text_prompt,
            index=robot_index,
            server_address=server_address,
            fps=fps,
            actions_per_chunk=actions_per_chunk,
            camera_paths={
                "front": camera.reference
            }
        )


if __name__ == "__main__":
    main()
