import flexivrdk
import rospy
import numpy as np
import time
import spdlog
import time
from scipy.spatial.transform import Rotation as R
import threading
from queue import Queue
from collections import deque
from utils.constants import *


class FlexivRobot(object):
    def __init__(self, robot_sn):
        self.robot_sn = robot_sn
        self.logger = spdlog.ConsoleLogger("RobotController")
        self.mode = flexivrdk.Mode
        print(self.mode)

        self.initialize_connection()
        self.vel = 1.0
        self.dof = 7

        # Control parameters
        self.control_freq = 30
        self.control_period = 1 / self.control_freq
        self.tcp_position = None
        self.stop_event = threading.Event()

    def initialize_connection(self):
        try:
            self.flexiv = flexivrdk.Robot(self.robot_sn)
            if self.flexiv.fault():
                self.logger.warn("Fault occured the connected robot")

                if not self.flexiv.ClearFault():
                    return 1
                self.logger.info("cleared")

            self.flexiv.Enable()

            self.flexiv.SwitchMode(self.mode.NRT_CARTESIAN_MOTION_FORCE)

            while not self.flexiv.operational():
                time.sleep(1)

        except Exception as e:
            self.logger.error(f"Failed to connect to Flexiv robot: {e}")

    def _parse_pt_states(self, pt_states, parse_target):
        """
        Parse the value of a specified primitive state from the pt_states string list.

        Parameters
        ----------
        pt_states : str list
            Primitive states string list returned from Robot::primitive_states().
        parse_target : str
            Name of the primitive state to parse for.

        Returns
        ----------
        str
            Value of the specified primitive state in string format. Empty string is
            returned if parse_target does not exist.
        """
        for state in pt_states:
            # Split the state sentence into words
            words = state.split()

            if words[0] == parse_target:
                return words[-1]

        return ""

    def get_joint_position(self):
        try:
            return self.flexiv.states().q
        except Exception as e:
            self.logger.error(f"Failed to get arm position: {e}")
            return

    def home_robot(self):
        self.flexiv.SwitchMode(self.mode.NRT_PRIMITIVE_EXECUTION)
        flexiv_home_js = FLEXIV_POSITIONS["home_js"]
        self.flexiv.ExecutePrimitive(
            f"MoveJ(target={' '.join(map(str, flexiv_home_js))}, jntVelScale=10)"
        )

        while (
            self._parse_pt_states(self.flexiv.primitive_states(), "reachedTarget")
            != "1"
        ):
            time.sleep(1)

        self.flexiv.SwitchMode(self.mode.NRT_CARTESIAN_MOTION_FORCE)

    def get_tcp_position(self, euler=False, degree=True):
        # p_x, p_y, p_z, wx, r_x, r_y, r_z
        try:
            tcp_position = self.flexiv.states().tcp_pose

            # print(f"now tcp quat: {tcp_position}")

            if euler:
                rot = self.quat2eulerZYX(tcp_position[3:], degree)
                trans = tcp_position[:3]
                return trans + rot

            return tcp_position
        except Exception as e:
            self.logger.error(f"Failed to get TCP position: {e}")
            return None

    def read_tcp_position(self):
        while not self.stop_event.is_set():
            try:
                self.tcp_position = self.flexiv.states().tcp_pose

            except Exception as e:
                self.logger.error(f"TCP position reading error: {e}")
                time.sleep(0.1)

    def movement_handler(self):
        while not self.stop_event.is_set():
            try:

                if not self.move_queue.empty():
                    target_pose = self.move_queue.get()
                    if self.tcp_position is not None:
                        # send command
                        try:
                            self.flexiv.SendCartesianMotionForce(
                                list(target_pose), [0] * 6, 0.5, 1.0, 0.4
                            )
                            time.sleep(1 / 30)
                        except Exception as e:
                            self.logger.error(f"SendCartesianMotionForce failed: {e}")

            except Exception as e:
                self.logger.error(f"Movement execution error: {e}")
                time.sleep(0.1)

    def move_to_position(self, target_pose):
        """Queue a movement command"""
        self.move_queue.put(list(target_pose))

    def move(self, target_arm_pose):
        # input 7
        print(f"move to {target_arm_pose}")
        try:
            self.flexiv.SendCartesianMotionForce(
                target_arm_pose, [0] * 6, 0.2, 1.0, 0.2, 0.2
            ) 
            tcp_position = self.get_tcp_position()
        except Exception as e:
            self.logger.error(f"Failed to move the robot: {e}")

    def move_joint(self, target_joint):
        pass

    def quat2eulerZYX(self, quat, degree=False):
        eulerZYX = (
            R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            .as_euler("xyz", degrees=degree)
            .tolist()
        )
        return eulerZYX

    def eulerZYX2quat(self, euler, degree=False):
        if degree:
            euler = np.radians(euler)

        tmp_quat = R.from_euler("xyz", euler).as_quat().tolist()
        quat = [tmp_quat[3], tmp_quat[0], tmp_quat[1], tmp_quat[2]]
        return quat

    def cleanup(self):
        """Cleanup function to properly stop all threads"""
        self.stop_event.set()
        self.tcp_reader_thread.join()
        self.movement_thread.join()


if __name__ == "__main__":
    controller = FlexivRobot()
    controller.home_robot()
