#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Piper Dual Leader Teleoperator for Hardware-level Teleoperation.

This teleoperator reads Master arm control commands from the Follower's CAN interface.
It is designed for the hardware teleop setup where:
- PC connects to Follower arms via USB
- Leader (Master) arms send control frames to Followers via CAN bus
- This teleoperator reads those control frames to capture operator intent
"""

import logging
import time

from piper_sdk import C_PiperInterface_V2

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_piper_dual_leader import PIPERDualLeaderConfig

logger = logging.getLogger(__name__)


class PIPERDualLeader(Teleoperator):
    """Teleoperator that reads Master control commands from Follower's CAN interface.

    In hardware-level teleoperation:
    - Leader arms are connected to Follower arms via CAN bus
    - PC connects to Follower arms via USB
    - This class reads the Master's control frames (canid 0x155, 0x156, etc.)
      that Followers receive from Leaders

    This allows recording the operator's intended action (from Master)
    while the robot state (observation) comes from Follower feedback.
    """

    config_class = PIPERDualLeaderConfig
    name = "piper_dual_leader"

    def __init__(self, config: PIPERDualLeaderConfig):
        super().__init__(config)
        self.config = config

        # Joint factor: 1000 * 180 / 3.14 ≈ 57324.840764
        # Converts from 0.001 degrees to radians
        self.joint_factor = 57324.840764

        # Create piper interfaces (connect to Follower's CAN to read Master commands)
        self.left_piper = C_PiperInterface_V2(config.left_port)
        self.right_piper = C_PiperInterface_V2(config.right_port)

        self._is_connected = False

    @property
    def action_features(self) -> dict[str, type]:
        """Action features for dual arm teleoperator."""
        motor_order = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
        features = {}
        for motor in motor_order:
            features[f"left_{motor}.pos"] = float
            features[f"right_{motor}.pos"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        """No feedback features for this teleoperator."""
        return {}

    def configure(self, **kwargs):
        pass

    def send_feedback(self, *args, **kwargs):
        pass

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True  # No calibration needed - just reading CAN messages

    def connect(self) -> None:
        """Connect to Follower CAN interfaces to read Master commands."""
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Connect to CAN ports (read-only, no enable needed)
        self.left_piper.ConnectPort()
        self.right_piper.ConnectPort()

        self._is_connected = True
        logger.info(f"{self} connected to CAN interfaces for reading Master commands.")

    def calibrate(self):
        """No calibration needed - we're just reading CAN messages."""
        pass

    def get_action(self) -> dict[str, float]:
        """Read Master control commands from both Follower's CAN interfaces.

        Returns the control commands sent by Leader (Master) arms.
        Uses GetArmJointCtrl() and GetArmGripperCtrl() to read control frames.
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()

        # Read left Master control commands
        left_joint_msg = self.left_piper.GetArmJointCtrl()
        left_joint_ctrl = left_joint_msg.joint_ctrl
        left_gripper_msg = self.left_piper.GetArmGripperCtrl()
        left_gripper_ctrl = left_gripper_msg.gripper_ctrl

        # Read right Master control commands
        right_joint_msg = self.right_piper.GetArmJointCtrl()
        right_joint_ctrl = right_joint_msg.joint_ctrl
        right_gripper_msg = self.right_piper.GetArmGripperCtrl()
        right_gripper_ctrl = right_gripper_msg.gripper_ctrl

        # Convert to radians and build action dict
        action = {
            "left_joint_1.pos": left_joint_ctrl.joint_1 / self.joint_factor,
            "left_joint_2.pos": left_joint_ctrl.joint_2 / self.joint_factor,
            "left_joint_3.pos": left_joint_ctrl.joint_3 / self.joint_factor,
            "left_joint_4.pos": left_joint_ctrl.joint_4 / self.joint_factor,
            "left_joint_5.pos": left_joint_ctrl.joint_5 / self.joint_factor,
            "left_joint_6.pos": left_joint_ctrl.joint_6 / self.joint_factor,
            "left_gripper.pos": left_gripper_ctrl.grippers_angle / 1_000_000.0,
            "right_joint_1.pos": right_joint_ctrl.joint_1 / self.joint_factor,
            "right_joint_2.pos": right_joint_ctrl.joint_2 / self.joint_factor,
            "right_joint_3.pos": right_joint_ctrl.joint_3 / self.joint_factor,
            "right_joint_4.pos": right_joint_ctrl.joint_4 / self.joint_factor,
            "right_joint_5.pos": right_joint_ctrl.joint_5 / self.joint_factor,
            "right_joint_6.pos": right_joint_ctrl.joint_6 / self.joint_factor,
            "right_gripper.pos": right_gripper_ctrl.grippers_angle / 1_000_000.0,
        }

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")

        return action

    def disconnect(self) -> None:
        """Disconnect from CAN interfaces."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Note: C_PiperInterface_V2 doesn't have explicit disconnect
        self._is_connected = False
        logger.info(f"{self} disconnected.")
