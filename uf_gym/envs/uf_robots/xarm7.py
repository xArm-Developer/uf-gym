from typing import Optional
import os
import numpy as np
from gymnasium import spaces
import math
from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet

urdf_file_dir = os.path.join(os.path.dirname(__file__), "../../../urdf/")

class XArm7(PyBulletRobot):
    """XArm7 robot in PyBullet.

    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optionnal): Position of the base base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles.
            Defaults to "ee".
    """

    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = False,
        base_position: Optional[np.ndarray] = None,
        control_type: str = "ee",
    ) -> None:
        base_position = base_position if base_position is not None else np.zeros(3)
        self.block_gripper = block_gripper
        self.control_type = control_type
        n_action = 3 if self.control_type == "ee" else 7  # control (x, y z) if "ee", else, control the 7 joints
        n_action += 0 if self.block_gripper else 1
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
        super().__init__(
            sim,
            body_name= "xarm7_with_gripper" if not self.block_gripper else "xarm7",
            file_name= urdf_file_dir + ("xarm/xarm7_with_gripper.urdf" if not self.block_gripper else "xarm/xarm7_robot.urdf"),
            base_position=base_position,
            action_space=action_space,
            joint_indices=np.array([1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15]) if not self.block_gripper else np.array([1, 2, 3, 4, 5, 6, 7]),
            joint_forces=np.array([60.0, 60.0, 40.0, 40.0, 40.0, 20.0, 20.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]) if not self.block_gripper else np.array([60.0, 60.0, 40.0, 40.0, 40.0, 20.0, 20.0]),

        )
        nuetual_gripper_jnt = 0.43;
        self.fingers_indices = np.array([10, 13])
        self.neutral_joint_values = np.array([0.0, -0.93368, 0.0, 0.68935, 0.0, 1.6213, 0.0, nuetual_gripper_jnt, nuetual_gripper_jnt, nuetual_gripper_jnt, nuetual_gripper_jnt, nuetual_gripper_jnt, nuetual_gripper_jnt ]) if not self.block_gripper else np.array([0.0, -0.93368, 0.0, 0.68935, 0.0, 1.6213, 0.0])
        self.ee_link = 7 if self.block_gripper else 16
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[1], spinning_friction=0.001)

    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.control_type == "ee":
            ee_displacement = action[:3]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
        else:
            arm_joint_ctrl = action[:6]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)

        if self.block_gripper:
            target_finger_joint = 0
            target_angles = target_arm_angles
        else:
            fingers_ctrl = action[-1] * 0.2  # limit maximum change in position
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl
            asin_val = max(min(1.0, (target_fingers_width - 0.018) / 0.11), -1.0)
            target_finger_joint = 0.69 - math.asin(asin_val)
            target_angles = np.concatenate((target_arm_angles, [target_finger_joint, target_finger_joint, target_finger_joint, target_finger_joint, target_finger_joint, target_finger_joint]))
        self.control_joints(target_angles=target_angles)

    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        ee_displacement = ee_displacement[:3] * 0.05  # limit maximum change in position
        # get the current position and the target position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_displacement
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # compute the new joint angles
        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=target_ee_position, orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        target_arm_angles = target_arm_angles[:7]  # remove fingers angles
        return target_arm_angles

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 7 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * 0.05  # limit maximum change in position
        # get the current position and the target position
        current_arm_joint_angles = np.array([self.get_joint_angle(joint=i) for i in range(7)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    def get_obs(self) -> np.ndarray:
        # end-effector position and velocity
        ee_position = np.array(self.get_ee_position())
        ee_velocity = np.array(self.get_ee_velocity())
        # fingers opening
        if not self.block_gripper:
            fingers_width = self.get_fingers_width()
            observation = np.concatenate((ee_position, ee_velocity, [fingers_width]))
        else:
            observation = np.concatenate((ee_position, ee_velocity))
        return observation

    def reset(self) -> None:
        self.set_joint_neutral()
        pass

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)

    def get_fingers_width(self) -> float:
        """Get the distance between the fingers."""
        drive_jnt = self.sim.get_joint_angle(self.body_name, self.fingers_indices[0])
        return 0.018 + 0.11 * math.sin(0.069-drive_jnt)

    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the end-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)
