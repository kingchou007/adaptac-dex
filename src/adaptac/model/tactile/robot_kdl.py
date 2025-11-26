from urdfpy import URDF
from math import pi as PI
import os


class RobotKDL(object):
    def __init__(
        self, urdf_path, arm_joints_name_order=None, hand_joints_name_order=None
    ):

        # Getting the URDF path
        self.robot = URDF.load(urdf_path)
        print("finished loading robot urdf")

        self.robot_links_info = {}
        for link in self.robot.links:
            # print(link.name)
            self.robot_links_info[link.name] = link

        self.config = {}
        for joint in self.robot.actuated_joints:
            # print(f'assigning joint values:{joint.name} to 0.0')
            self.config[joint.name] = 0.0

        self.arm_joints_name_order = arm_joints_name_order
        self.hand_joints_name_order = hand_joints_name_order

    def forward_kinematics(self, joint_values, link_name=None):
        if "arm" in joint_values:
            if self.arm_joints_name_order is not None:
                arm_joint_values = joint_values["arm"]
                for idx, joint_name in enumerate(self.arm_joints_name_order):
                    self.config[joint_name] = arm_joint_values[idx]
            else:
                raise ValueError(
                    "arm_joints_name_order is None, please provide the correct joint order"
                )

        if "hand" in joint_values:
            if self.hand_joints_name_order is not None:
                hand_joint_values = joint_values["hand"]
                for idx, joint_name in enumerate(self.hand_joints_name_order):
                    self.config[joint_name] = hand_joint_values[idx]
            else:
                raise ValueError(
                    "hand_joints_name_order is None, please provide the correct joint order"
                )

        fk = self.robot.link_fk(self.config)

        if link_name is not None:
            return fk[self.robot_links_info[link_name]]
        else:
            return fk
