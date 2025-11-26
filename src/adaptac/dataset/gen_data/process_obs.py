"""
Functions for processing robot sensor data.

This module provides utilities for:
- Transforming robot sensor data between coordinate frames (base, camera, hand)
- Generating and filtering point clouds from RGB-D camera images
- Processing tactile sensor information into various representations
- Applying workspace boundaries to filter point cloud data
- Converting end-effector poses between reference frames

"""

import open3d as o3d
import numpy as np
from adaptac.dataset.utils.projector import Projector
from adaptac.model.tactile.robot_tactile_kdl import RobotTactileKDL
from adaptac.model.tactile.constants import (
    URDF_PATH,
    ARM_JOINTS_ORDER,
    HAND_JOINTS_ORDER,
)
from adaptac.dataset.utils.constants import (
    INTRINSICS_MATRIX_515,
    EXTRINSIC_MATRIX,
    BASE_WORKSPACE_MAX,
    BASE_WORKSPACE_MIN,
    CAM_WORKSPACE_MAX,
    CAM_WORKSPACE_MIN,
    IMG_MEAN,
    IMG_STD,
)

from typing import Tuple, Dict, List, Optional, Union, Any
from utils.transformation import trans_points
from adaptac.model.tactile.tactile_processer import tactile_process


# Initialize global objects
projector = Projector()
robot_tactile_kdl = RobotTactileKDL(URDF_PATH, ARM_JOINTS_ORDER, HAND_JOINTS_ORDER)


def convert_tcp_data_to_camera(data: np.ndarray) -> np.ndarray:
    """
    Convert data from TCP to camera coordinate system.

    Args:
        data: TCP data with position and orientation

    Returns:
        Converted data in camera coordinate system
    """
    cam_tcp = projector.project_tcp_to_camera_coord(data[:7])
    return np.concatenate([cam_tcp, data[7:]])


def get_point_cloud(
    colors: np.ndarray,
    depths: np.ndarray,
    voxel_size: float = 0.005,
    frame: str = "base",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate point cloud from color and depth images.

    Args:
        colors: RGB color image
        depths: Depth image
        voxel_size: Size of voxels for downsampling
        frame: Coordinate frame ('base' or 'camera')

    Returns:
        Tuple of points and colors arrays
    """
    # Get image dimensions and camera parameters
    h, w = depths.shape
    fx, fy = INTRINSICS_MATRIX_515["fx"], INTRINSICS_MATRIX_515["fy"]
    cx, cy = INTRINSICS_MATRIX_515["cx"], INTRINSICS_MATRIX_515["cy"]
    scale = 1000

    # Convert to Open3D format
    colors_o3d = o3d.geometry.Image(colors.astype(np.uint8))
    depths_o3d = o3d.geometry.Image(depths.astype(np.float32))

    # Set up camera intrinsics
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy
    )

    # Create RGBD image and point cloud
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        colors_o3d, depths_o3d, scale, convert_rgb_to_intensity=False
    )
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)

    # Downsample point cloud
    cloud = cloud.voxel_down_sample(voxel_size)
    points = np.asarray(cloud.points, dtype=np.float32)
    colors = np.asarray(cloud.colors, dtype=np.float32)

    # Transform points if needed
    if frame == "base":
        points = trans_points(points, EXTRINSIC_MATRIX)  # camera to base transform

    return points, colors


def crop_pc(
    points: np.ndarray,
    colors: np.ndarray,
    frame: str = "base",
    crop_frame: Optional[str] = None,
) -> np.ndarray:
    """
    Crop point cloud to workspace boundaries.

    Args:
        points: 3D point coordinates
        colors: RGB colors for each point
        frame: Current coordinate frame ('base' or 'camera')
        crop_frame: Frame to use for cropping (optional)

    Returns:
        Cropped point cloud with concatenated position and color features
    """
    # Determine workspace boundaries based on frame
    if frame == "base":
        ws_max = BASE_WORKSPACE_MAX
        ws_min = BASE_WORKSPACE_MIN
    elif frame == "camera":
        if crop_frame is None:
            ws_max = CAM_WORKSPACE_MAX
            ws_min = CAM_WORKSPACE_MIN
        elif crop_frame == "base":
            # Transform points to base frame for cropping
            points = trans_points(points, EXTRINSIC_MATRIX)  # camera to base
            ws_max = BASE_WORKSPACE_MAX
            ws_min = BASE_WORKSPACE_MIN

    # Apply workspace boundary filters
    x_mask = (points[:, 0] >= ws_min[0]) & (points[:, 0] <= ws_max[0])
    y_mask = (points[:, 1] >= ws_min[1]) & (points[:, 1] <= ws_max[1])
    z_mask = (points[:, 2] >= ws_min[2]) & (points[:, 2] <= ws_max[2])
    mask = x_mask & y_mask & z_mask

    # Apply mask to points and colors
    filtered_points = points[mask]
    filtered_colors = colors[mask]

    # Normalize colors using ImageNet statistics
    normalized_colors = (filtered_colors - IMG_MEAN) / IMG_STD

    # Transform back to camera frame if needed
    if frame == "camera" and crop_frame == "base":
        filtered_points = trans_points(filtered_points, np.linalg.inv(EXTRINSIC_MATRIX))

    # Concatenate position and color features
    return np.concatenate([filtered_points, normalized_colors], axis=-1).astype(
        np.float32
    )


def get_processed_tactile_data(
    input_tactile: Union[np.ndarray, Dict[str, Any]],
    state_data: Dict[str, np.ndarray],
    tactile_rep_type: str = "3d_canonical_data",
    tactile_frame: str = "camera",
) -> Union[np.ndarray, Dict[str, Any]]:
    """
    Process tactile data based on representation type.

    Args:
        input_tactile: Raw tactile input data
        state_data: Robot state information
        tactile_rep_type: Type of tactile representation
        tactile_frame: Coordinate frame for tactile data

    Returns:
        Processed tactile data in the specified format
    """
    # Validate tactile frame
    if tactile_frame not in ["camera", "base", "hand"]:
        raise ValueError(f"Invalid tactile frame: {tactile_frame}")

    # Process based on representation type
    if tactile_rep_type == "3d_canonical_data":
        tactile_data = tactile_process(
            input_tactile,
            state_data,
            robot_tactile_kdl,
            end_idx=-0,
            resultant_type=None,
            base=tactile_frame,
        )
        return tactile_data.squeeze()
    elif tactile_rep_type == "no":
        # Return zeros if no tactile data is requested
        return np.zeros(360, dtype=np.float32)
    elif tactile_rep_type == "raw_data":
        # Flatten raw tactile data
        tactile_data = []
        for sensor_name in input_tactile[0]:
            tactile_data.extend(input_tactile[0][sensor_name].reshape(-1).tolist())
        return np.array(tactile_data, dtype=np.float32)
    elif tactile_rep_type == "raw_dict":
        # Return raw dictionary
        return input_tactile
    else:
        raise NotImplementedError(
            f"Tactile representation type '{tactile_rep_type}' not implemented"
        )
