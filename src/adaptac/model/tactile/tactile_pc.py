import numpy as np
from scipy.spatial.transform import Rotation as R
from adaptac.model.tactile.constants import *
from utils.constants import EXTRINSIC_MATRIX


# Tactile sensor orientation offsets (in link frame)
THUMB_FINGERTIP_PULP_ORI = np.array([-0.01632, 0.0001, -0.0159])
THUMB_FINGERTIP_TIP_ORI = np.array([-0.01552, -0.02958, -0.0159])
FINGERTIP_TIP_ORI = np.array([-0.01553, -0.01398, 0.0159])
DIP_PULP_ORI = np.array([-0.0158, -0.01237, 0.0159])

# Tactile link names
TACTILE_LINKS = [
    "fingertip",
    "fingertip_2",
    "fingertip_3",
    "dip",
    "dip_2",
    "dip_3",
    "thumb_fingertip",
]

# Default tactile point color (red)
TACTILE_COLOR = np.array([255, 0, 0])


def draw_tactile_point(
    hand_joint_positions, arm_joint_positions, ee_pose, frame, robot_kdl
):
    """Generate tactile point cloud from robot joint states.
    
    Args:
        hand_joint_positions: Hand joint positions
        arm_joint_positions: Arm joint positions
        ee_pose: End-effector pose [x, y, z, qw, qx, qy, qz]
        frame: Target frame ('base' or 'camera')
        robot_kdl: Robot KDL model
    
    Returns:
        [N, 6] array of tactile points with color (xyz + rgb)
    """
    if frame not in ("base", "camera"):
        raise ValueError(f"Invalid frame '{frame}'. Choose 'base' or 'camera'.")

    joint_values = {"arm": arm_joint_positions, "hand": hand_joint_positions}
    point_cloud_list = []

    # Process each tactile link
    for link in robot_kdl.robot_links_info.keys():
        if link not in TACTILE_LINKS:
            continue

        link_pose = robot_kdl.forward_kinematics(joint_values, link_name=link)
        pulp_points = None
        tip_points = None

        # Process different link types
        if "thumb" in link and "fingertip" in link:
            pulp_points = get_tactile_points(
                THUMB_FINGERTIP_PULP_ORI, PAXINI_THUMB_PULP_COORD, link_pose
            )
            tip_points = get_tactile_points(
                THUMB_FINGERTIP_TIP_ORI, PAXINI_THUMB_TIP_COORD, link_pose
            )
        elif "fingertip" in link:
            tip_points = get_tactile_points(
                FINGERTIP_TIP_ORI, PAXINI_TIP_COORD, link_pose
            )
        elif "dip" in link:
            pulp_points = get_tactile_points(
                DIP_PULP_ORI, PAXINI_PULP_COORD, link_pose
            )

        # Add color and append to point cloud
        if pulp_points is not None:
            colored_points = np.hstack([pulp_points, np.tile(TACTILE_COLOR, (pulp_points.shape[0], 1))])
            point_cloud_list.append(colored_points)
        if tip_points is not None:
            colored_points = np.hstack([tip_points, np.tile(TACTILE_COLOR, (tip_points.shape[0], 1))])
            point_cloud_list.append(colored_points)

    if not point_cloud_list:
        return np.empty((0, 6))

    point_cloud = np.vstack(point_cloud_list)

    # Transform to camera frame if requested
    if frame == "camera":
        positions = point_cloud[:, :3]
        colors = point_cloud[:, 3:]
        homogeneous_positions = np.hstack([positions, np.ones((positions.shape[0], 1))])
        transformed_positions = (np.linalg.inv(EXTRINSIC_MATRIX) @ homogeneous_positions.T).T
        point_cloud = np.hstack([transformed_positions[:, :3], colors])

    return point_cloud


def get_tactile_points(tactile_ori, tactile_points, link_pose):
    """
    Transforms local tactile points to real-world coordinates.

    Parameters:
        tactile_ori (np.ndarray): Orientation offset for tactile points.
        tactile_points (np.ndarray): Local tactile points coordinates.
        link_pose (np.ndarray): Homogeneous transformation matrix of the link.

    Returns:
        np.ndarray: Transformed tactile points in real-world coordinates.
    """
    # Calculate local points by adding orientation offset
    local_points = tactile_ori + tactile_points

    # Extract rotation and translation from link pose
    rotation = link_pose[:3, :3]
    translation = link_pose[:3, 3]

    # Apply rotation and translation to get real-world points
    real_points = (rotation @ local_points.T).T + translation

    return real_points