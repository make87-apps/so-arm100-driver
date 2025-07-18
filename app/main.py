from datetime import datetime, timezone

import make87
import logging
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
    environment_dt = make87.config.get_config_value(config, "environment_dt", default=0.1, converter=float)

    server_address = f"{robotclient.vpn_ip}:{robotclient.vpn_port}"
    start_so100_robot_client(
        index=robot_index,
        server_address=server_address,
        fps=fps,
        actions_per_chunk=actions_per_chunk,
        environment_dt=environment_dt
    )


if __name__ == "__main__":
    main()
