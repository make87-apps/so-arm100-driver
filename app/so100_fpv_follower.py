import logging
from typing import Any

import numpy as np
from lerobot.errors import DeviceNotConnectedError
from lerobot.robots.so100_follower import SO100FollowerEndEffector, SO100FollowerEndEffectorConfig


logger = logging.getLogger(__name__)

class SO100FPVFollower(SO100FollowerEndEffector):
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Transform action from end-effector space (x,y,z + pitch,yaw) to joint space and send to motors.

        Expected dict keys:
          - 'delta_x', 'delta_y', 'delta_z' (linear deltas)
          - 'delta_pitch', 'delta_yaw' (angular deltas, scaled by step sizes; radians per unit)
          - optional 'gripper' in [0..2] where 1.0 is no change
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Defaults for step sizes if not provided
        step = self.config.end_effector_step_sizes
        sx = step.get("x", 1.0)
        sy = step.get("y", 1.0)
        sz = step.get("z", 1.0)
        sp = step.get("pitch", 0.02)  # ~1.1° per unit
        syaw = step.get("yaw", 0.02)  # ~1.1° per unit

        # Parse action
        if isinstance(action, dict):
            # Position deltas
            if not all(k in action for k in ["delta_x", "delta_y", "delta_z"]):
                logger.warning(
                    f"Expected action keys 'delta_x', 'delta_y', 'delta_z', got {list(action.keys())}"
                )
                delta_ee = np.zeros(3, dtype=np.float32)
            else:
                delta_ee = np.array(
                    [
                        action["delta_x"] * sx,
                        action["delta_y"] * sy,
                        action["delta_z"] * sz,
                        ],
                    dtype=np.float32,
                )

            # Orientation deltas (end-effector frame)
            delta_pitch = float(action.get("delta_pitch", 0.0)) * sp
            delta_yaw   = float(action.get("delta_yaw", 0.0)) * syaw

            # Gripper channel (map same as before; default to 1.0=no change)
            if "gripper" not in action:
                action["gripper"] = [1.0]
            grip_in = action["gripper"]
        else:
            # If someone passes a numpy array: [dx, dy, dz, (grip?)] — orientation would be missing; reject.
            logger.warning("Array action without orientation not supported; ignoring and sending zeros.")
            delta_ee = np.zeros(3, dtype=np.float32)
            delta_pitch = 0.0
            delta_yaw = 0.0
            grip_in = [1.0]

        # Initialize state if needed
        if self.current_joint_pos is None:
            current_joint_pos = self.bus.sync_read("Present_Position")
            self.current_joint_pos = np.array([current_joint_pos[name] for name in self.bus.motors])

        if self.current_ee_pos is None:
            self.current_ee_pos = self.kinematics.forward_kinematics(self.current_joint_pos)

        # Build desired pose: translate + rotate (R_current @ Rz(yaw) @ Ry(pitch))
        desired_ee_pos = np.eye(4, dtype=np.float32)

        # Position with bounds
        t = self.current_ee_pos[:3, 3] + delta_ee
        if self.end_effector_bounds is not None:
            t = np.clip(t, self.end_effector_bounds["min"], self.end_effector_bounds["max"])
        desired_ee_pos[:3, 3] = t

        # Orientation update in end-effector frame (yaw about local Z, then pitch about local Y)
        R_cur = self.current_ee_pos[:3, :3]
        R_delta = _rot_z(delta_yaw) @ _rot_y(delta_pitch)
        desired_ee_pos[:3, :3] = (R_cur @ R_delta).astype(np.float32)

        # IK: joint targets (degrees, consistent with existing API)
        target_joint_values_in_degrees = self.kinematics.inverse_kinematics(
            self.current_joint_pos, desired_ee_pos
        )

        # Joint space action
        joint_action = {
            f"{key}.pos": target_joint_values_in_degrees[i]
            for i, key in enumerate(self.bus.motors.keys())
        }

        # Gripper handling
        joint_action["gripper.pos"] = np.clip(
            self.current_joint_pos[-1] + (grip_in[-1] - 1.0) * self.config.max_gripper_pos,
            5,
            self.config.max_gripper_pos,
            )

        # Update caches
        self.current_ee_pos = desired_ee_pos.copy()
        self.current_joint_pos = target_joint_values_in_degrees.copy()
        self.current_joint_pos[-1] = joint_action["gripper.pos"]

        return super().send_action(joint_action)

def _rot_y(pitch: float) -> np.ndarray:
    pitch = np.deg2rad(pitch)
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([[ c, 0,  s],
                     [ 0, 1,  0],
                     [-s, 0,  c]], dtype=np.float32)

def _rot_z(yaw: float) -> np.ndarray:
    yaw = np.deg2rad(yaw)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[ c, -s, 0],
                     [ s,  c, 0],
                     [ 0,  0, 1]], dtype=np.float32)