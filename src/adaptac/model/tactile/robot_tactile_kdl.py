import numpy as np
from scipy.spatial.transform import Rotation as R
from adaptac.model.tactile.robot_kdl import RobotKDL
from adaptac.model.tactile.constants import *


class RobotTactileKDL:
    def __init__(
        self, urdf_path, arm_joints_name_order=None, hand_joints_name_order=None
    ):
        self.robot_kdl = RobotKDL(
            urdf_path, arm_joints_name_order, hand_joints_name_order
        )
        self.point_per_sensor = len(PAXINI_PULP_COORD)

    def forward_kinematics(
        self,
        arm_joint_state,
        hand_joint_state,
        tactile_list=[],
        coords_type="full",
        coords_space="base",
        base="arm",
    ):
        joint_values = {}
        joint_values["arm"] = arm_joint_state
        joint_values["hand"] = hand_joint_state

        link_poses = self.robot_kdl.forward_kinematics(joint_values)
        assert tactile_list == list(TACTILE_INFO.keys())

        tactile_points = []

        if base == "hand":
            hand_base_pose = link_poses[self.robot_kdl.robot_links_info["palm_lower"]]

        for tactile in tactile_list:
            link = TACTILE_INFO[tactile]
            link_pose = link_poses[self.robot_kdl.robot_links_info[link]]
            if base == "hand":
                link_pose = np.linalg.inv(hand_base_pose) @ link_pose
            elif base == "camera":
                from utils.constants import EXTRINSIC_MATRIX

                link_pose = np.linalg.inv(EXTRINSIC_MATRIX) @ link_pose
            # print(tactile, link)
            if "thumb" in link:
                if "fingertip" in link:
                    if tactile == "thumb_tip":
                        tactile_real_points = self.get_tactile_points(
                            THUMB_TIP_TACTILE_ORI_COORD,
                            PAXINI_THUMB_TIP_COORD,
                            link_pose,
                            coords_type,
                            coords_space,
                        )
                    elif tactile == "thumb_pulp":
                        tactile_real_points = self.get_tactile_points(
                            THUMB_PULP_TACTILE_ORI_COORD,
                            PAXINI_THUMB_PULP_COORD,
                            link_pose,
                            coords_type,
                            coords_space,
                        )
                    else:
                        raise ValueError("Invalid tactile")
            elif "fingertip" in link:
                tactile_real_points = self.get_tactile_points(
                    OTHER_TIP_TACTILE_ORI_COORD,
                    PAXINI_TIP_COORD,
                    link_pose,
                    coords_type,
                    coords_space,
                )
            elif "dip" in link:
                tactile_real_points = self.get_tactile_points(
                    OTHER_PULP_TACTILE_ORI_COORD,
                    PAXINI_PULP_COORD,
                    link_pose,
                    coords_type,
                    coords_space,
                )
            else:
                raise ValueError("Invalid link")

            tactile_points.append(tactile_real_points)

        return tactile_points

    def get_tactile_points(
        self, tactile_ori, tactile_points, link_pose, coords_type, coords_space
    ):
        if coords_type == "full":
            local_points = tactile_ori + tactile_points
            # local to world
            rotation = link_pose[:3, :3]
            translation = link_pose[:3, 3]
            real_points = (rotation @ local_points.T).T + translation
            real_angle = R.from_matrix(rotation).as_euler("xyz").reshape(1, 3) / np.pi
            real_points = np.concatenate(
                [real_points, np.repeat(real_angle, self.point_per_sensor, axis=0)],
                axis=1,
            )
        elif coords_type == "original":
            local_points = np.array([tactile_ori])
            # local to world
            rotation = link_pose[:3, :3]
            translation = link_pose[:3, 3]
            real_points = (rotation @ local_points.T).T + translation
            real_angle = R.from_matrix(rotation).as_euler("xyz").reshape(1, 3) / np.pi

            if coords_space == "canonical":
                min_point = np.min(tactile_points, axis=0)
                max_point = np.max(tactile_points, axis=0)
                diagonal_length = np.linalg.norm(max_point - min_point)
                center_point = (min_point + max_point) / 2
                tactile_points = 2 * (tactile_points - center_point) / diagonal_length
                # ! tactile pad origin xyz rpy/np.pi  + tactile point xyz
                real_points = np.concatenate(
                    [
                        np.repeat(real_points, self.point_per_sensor, axis=0),
                        np.repeat(real_angle, self.point_per_sensor, axis=0),
                        tactile_points,
                    ],
                    axis=1,
                )

        return real_points
