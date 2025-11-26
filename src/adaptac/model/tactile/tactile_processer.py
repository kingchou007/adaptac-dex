import numpy as np
from scipy.spatial.transform import Rotation as R

from adaptac.model.tactile.robot_tactile_kdl import RobotTactileKDL
from adaptac.model.tactile.constants import (
    URDF_PATH,
    ARM_JOINTS_ORDER,
    HAND_JOINTS_ORDER,
    TACTILE_RAW_DATA_SCALE,
    TACTILE_DATA_TYPE,
)


robot_tactile_kdl = RobotTactileKDL(URDF_PATH, ARM_JOINTS_ORDER, HAND_JOINTS_ORDER)


def convert_force_from_paxini_to_link(forces):
    """Convert force from paxini coordinate system to link coordinate system.

    Args:
        forces: Force data with shape (N, 3)

    Returns:
        Transformed force data with same shape
    """
    forces = forces.copy()
    # Swap x and z axes (paxini coord: x,y,z -> link coord: z,y,x)
    forces[:, [0, 2]] = forces[:, [2, 0]]
    return forces


def compute_resultant(coord, forces, computation_type="force", return_per_point_force=False, scale=5):
    """Compute resultant force from tactile data.

    Args:
        coord: Coordinate data with shape (N, 6+) where last 3 elements are rotation
        forces: Force data with shape (N, 3)
        computation_type: Type of computation ('force' only supported currently)
        return_per_point_force: Whether to return per-point forces
        scale: Scaling factor for forces

    Returns:
        Resultant force vector (and per-point forces if requested)
    """
    if computation_type != "force":
        raise NotImplementedError(f"Computation type '{computation_type}' not supported")

    # Convert forces to link coordinate system
    force_in_link = convert_force_from_paxini_to_link(forces)

    # Extract rotation angles and convert to rotation matrices
    rot_rpy = coord[:, 3:6] * np.pi
    rot_matrix = R.from_euler("xyz", rot_rpy).as_matrix()

    # Transform forces to base frame and scale
    force_in_base = np.einsum("ijk,ik->ij", rot_matrix, force_in_link) / scale

    # Compute resultant force
    resultant_force = np.sum(force_in_base, axis=0)

    if return_per_point_force:
        return resultant_force, force_in_base
    return resultant_force


def process_raw_tactile_data(tactile_data, end_idx=0):
    """Process raw tactile data by reshaping and scaling.

    Args:
        tactile_data: Raw tactile data with shape [B, N*3]
        end_idx: End index for slicing (default 0 means no slicing)

    Returns:
        Processed tactile data with shape [B, N*3] after scaling
    """
    batch_size = tactile_data.shape[0]
    tactile_data = tactile_data.reshape(batch_size, -1, 3)
    tactile_data /= TACTILE_RAW_DATA_SCALE
    tactile_data = tactile_data.reshape(batch_size, -1)
    
    if end_idx != 0:
        return tactile_data[:len(tactile_data) + end_idx]
    return tactile_data


def _get_tactile_coordinates(robot_tactile_kdl, arm_joint_state, hand_joint_state, 
                             tactile_link_list, tactile_type, base="hand"):
    """Get tactile point coordinates based on type."""
    if tactile_type == "3d_raw_data":
        return robot_tactile_kdl.forward_kinematics(
            arm_joint_state,
            hand_joint_state,
            tactile_link_list,
            coords_type="full",
            base=base,
        )
    elif tactile_type == "3d_canonical_data":
        return robot_tactile_kdl.forward_kinematics(
            arm_joint_state,
            hand_joint_state,
            tactile_link_list,
            coords_type="original",
            coords_space="canonical",
            base=base,
        )
    else:
        raise ValueError(f"Unknown tactile_type: {tactile_type}")


def _concatenate_tactile_forces(tactile_force_dict):
    """Concatenate force data from all sensors."""
    forces = [tactile_force_dict[sensor_name] for sensor_name in tactile_force_dict]
    return np.concatenate(forces) / TACTILE_RAW_DATA_SCALE


def _slice_data(data, end_idx):
    """Slice data based on end_idx."""
    if end_idx != 0:
        return data[:len(data) + end_idx]
    return data


def process_3d_tactile_data(
    ori_tactile_data,
    state,
    robot_tactile_kdl,
    tactile_type,
    end_idx=0,
    resultant_type=None,
    base="hand",
):
    """Process 3D tactile data with forward kinematics.

    Args:
        ori_tactile_data: Original tactile data dictionary
        state: Robot state data with 'arm_abs_joint' and 'hand_abs_joint'
        robot_tactile_kdl: KDL robot model
        tactile_type: Type of tactile data ('3d_raw_data' or '3d_canonical_data')
        end_idx: End index for slicing (default 0 means no slicing)
        resultant_type: Type of resultant computation (None to skip)
        base: Base frame for coordinates

    Returns:
        Processed tactile data [B, N, 6] (xyz + fxfyfz)
        Optionally returns resultant forces if resultant_type is not None
    """
    tactile_link_list = list(ori_tactile_data[0].keys())
    arm_joint_states = state["arm_abs_joint"]
    hand_joint_states = state["hand_abs_joint"]

    processed_tactiles = []
    resultants = [] if resultant_type is not None else None

    for arm_joint_state, hand_joint_state, tactile_force in zip(
        arm_joint_states, hand_joint_states, ori_tactile_data
    ):
        # Get tactile point coordinates
        tactile_points = _get_tactile_coordinates(
            robot_tactile_kdl, arm_joint_state, hand_joint_state,
            tactile_link_list, tactile_type, base
        )
        tactile_xyz = np.concatenate(tactile_points)

        # Concatenate forces from all sensors
        tactile_forces = _concatenate_tactile_forces(tactile_force)

        # Compute resultant force if requested
        if resultant_type is not None:
            resultant_force = compute_resultant(tactile_xyz, tactile_forces, resultant_type)
            resultants.append(resultant_force)

        # Combine position and force data
        tactile_xyzfxfyfz = np.concatenate([tactile_xyz, tactile_forces], axis=1)
        processed_tactiles.append(tactile_xyzfxfyfz)

    # Stack and slice data
    processed_tactiles = np.stack(processed_tactiles, axis=0)
    processed_tactiles = _slice_data(processed_tactiles, end_idx)

    if resultant_type is not None:
        resultants = np.stack(resultants, axis=0)
        resultants = _slice_data(resultants, end_idx)
        return processed_tactiles, resultants

    return processed_tactiles


def tactile_process(
    ori_tactile_data,
    state,
    robot_tactile_kdl,
    end_idx=0,
    resultant_type=None,
    base="hand",
):
    """Process tactile data based on available type in TACTILE_DATA_TYPE.

    Args:
        ori_tactile_data: Original tactile data dictionary
        state: Robot state data with 'arm_abs_joint' and 'hand_abs_joint'
        robot_tactile_kdl: KDL robot model
        end_idx: End index for slicing (default 0 means no slicing)
        resultant_type: Type of resultant computation (None to skip)
        base: Base frame for coordinates

    Returns:
        Processed tactile data (and optionally resultant forces for 3D data)
        Returns None if no matching tactile type found
    """
    # Find first available tactile type
    for tactile_type in TACTILE_DATA_TYPE:
        if tactile_type not in ori_tactile_data:
            continue

        if tactile_type == "raw_data":
            tactile_data = ori_tactile_data[tactile_type].numpy().copy()
            return process_raw_tactile_data(tactile_data, end_idx)

        elif tactile_type in ("3d_raw_data", "3d_canonical_data"):
            return process_3d_tactile_data(
                ori_tactile_data,
                state,
                robot_tactile_kdl,
                tactile_type,
                end_idx,
                resultant_type,
                base,
            )

    return None
