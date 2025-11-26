"""
Evaluation Agent.
"""

import time
import numpy as np
import warnings
import rospy
import cv2
from termcolor import cprint
from collections import deque

from device.robot.flexiv import FlexivRobot
from device.sensors.paxini import PaxiniTactile
from utils.transformation import xyz_rot_transform
from device.camera.realsense import RealSenseRGBDCamera
from adaptac.dataset.utils.projector import Projector
from adaptac.dataset.gen_data.process_obs import get_processed_tactile_data
from utils.constants import *

warnings.filterwarnings(
    "ignore",
    message="Link .* is of type 'fixed' but set as active in the active_links_mask.*",
)

# load module according to hand type
HAND_TYPE = "Leap"
hand_module = __import__("holodex.robot.hand")
Hand_module_name = f"{HAND_TYPE}Hand"
# get relevant classes
Hand = getattr(hand_module.robot, Hand_module_name)


class Agent:
    """
    Evaluation agent with Flexiv arm, Dahuan gripper and Intel RealSense RGB-D camera.

    Follow the implementation here to create your own real-world evaluation agent.
    """

    def __init__(
        self,
        robot_sn,
        camera_serial,
        frame,
        obs_num,
        enable_tactile,
        tactile_rep_type="3d_canonical_data",
        tactile_frame="camera",
        **kwargs,
    ):
        self.camera_serial = camera_serial
        self.frame = frame
        self.obs_num = obs_num
        self.enable_tactile = enable_tactile

        print("Init robot, gripper, and camera.")
        self.robot = FlexivRobot(robot_sn)
        self.robot.home_robot()
        time.sleep(1.5)

        self.hand = Hand()
        rospy.sleep(1.5)
        self.home_hand()
        tactile_num = 2 if enable_tactile else 0
        if self.enable_tactile:
            self.tactile = PaxiniTactile(tactile_num)
            self.tactile_rep_type = tactile_rep_type
            self.tactile_frame = tactile_frame

        # setup the camera
        self.camera = RealSenseRGBDCamera(serial=self.camera_serial)
        for _ in range(30):
            self.camera.get_rgbd_image()

        self.rgb_frames = deque([], maxlen=self.obs_num)
        self.depth_frames = deque([], maxlen=self.obs_num)
        self.arm_ee_poses = deque([], maxlen=self.obs_num)
        self.hand_joint_poses = deque([], maxlen=self.obs_num)
        self.tactile_datas = deque([], maxlen=self.obs_num)
        print("Initialization Finished.")

    @property
    def intrinsics(self):
        return np.array(
            [
                [INTRINSICS_MATRIX_515["fx"], 0, INTRINSICS_MATRIX_515["cx"]],
                [0, INTRINSICS_MATRIX_515["fy"], INTRINSICS_MATRIX_515["cy"]],
                [0, 0, 1],
            ]
        )

    @property
    def ready_rot_6d(self):
        return np.array([-1, 0, 0, 0, 1, 0])

    @property
    def home_robot_hand_pos(self):
        return np.array(
            [
                0.2378,
                0.0644,
                0.0107,
                -0.0169,
                0.1503,
                0.0721,
                -0.0614,
                -0.3605,
                -0.0276,
                0.0629,
                -0.0874,
                -0.3620,
                0.2500,
                0.0199,
                1.3300,
                -0.6504,
            ]
        )

    def home_hand(self):
        self.hand.move(self.home_robot_hand_pos)

    def get_observation(self):
        colors, depths = self.camera.get_rgbd_image()

        colors = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)

        if self.frame == "camera":
            arm_ee_pose = Projector().project_tcp_to_camera_coord(
                self.robot.get_tcp_position()
            )
        else:
            arm_ee_pose = self.robot.get_tcp_position()

        hand_joint_pos = self.hand.get_hand_position()

        if self.enable_tactile:
            tactile_data = self.get_tactile()
        else:
            tactile_data = None

        self.rgb_frames.append(colors)
        self.depth_frames.append(depths)
        self.arm_ee_poses.append(arm_ee_pose)
        self.hand_joint_poses.append(hand_joint_pos)
        self.tactile_datas.append(tactile_data)

        colors_obs = np.stack(self.rgb_frames, axis=0)
        depths_obs = np.stack(self.depth_frames, axis=0)
        arm_ee_pose_obs = np.stack(self.arm_ee_poses, axis=0)
        hand_joint_pos_obs = np.stack(self.hand_joint_poses, axis=0)
        tactile_data_obs = np.stack(self.tactile_datas, axis=0)

        return (
            colors_obs,
            depths_obs,
            arm_ee_pose_obs,
            hand_joint_pos_obs,
            tactile_data_obs,
        )

    def get_tactile(self) -> np.ndarray:
        state_data = {
            "arm_abs_joint": np.array([self.robot.get_joint_position()]),
            "hand_abs_joint": np.array([self.hand.get_hand_position()]),
        }

        tactile = self.tactile.get_tactile(
            state_data,
            tactile_rep_type=self.tactile_rep_type,
            tactile_frame=self.tactile_frame,
        )
        return tactile

    def set_tcp_pose(
        self, pose, rotation_rep, rotation_rep_convention=None, blocking=False
    ):
        tcp_pose = xyz_rot_transform(
            pose,
            from_rep=rotation_rep,
            to_rep="quaternion",
            from_convention=rotation_rep_convention,
        )
        self.robot.move(tcp_pose)
        if blocking:
            time.sleep(0.1)

    def set_hand_joint_pos(self, hand_pos, blocking=False):
        self.hand.move(hand_pos)
        if blocking:
            time.sleep(0.1)

    def stop(self):
        print("Stop the robot.")
        self.robot.stop()



if __name__ == "__main__":
    rospy.init_node("eval_agent")
    agent = Agent(robot_sn="Rizon4-062521", camera_serial="f1230963")
    agent.get_tactile()
