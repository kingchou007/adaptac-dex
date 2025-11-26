import torch
from adaptac.dataset.utils.constants import *


def normalize_arm_hand(tcp_list, norm_trans=False):
    """tcp_list: [T, 3(trans) + 6(rot) + 16(hand joint)]"""
    if norm_trans:
        tcp_list[:, :3] = (tcp_list[:, :3] - TRANS_MIN) / (
            TRANS_MAX - TRANS_MIN
        ) * 2 - 1
    tcp_list[:, 9:] = (tcp_list[:, 9:] - HAND_JOINT_LOWER_LIMIT) / (
        HAND_JOINT_UPPER_LIMIT - HAND_JOINT_LOWER_LIMIT
    ) * 2 - 1
    return tcp_list


def denormalize_arm_hand(tcp_list, norm_trans=False):
    """tcp_list: [T, 3(trans) + 6(rot) + 16(hand joint)]"""
    if norm_trans:
        tcp_list[:, :3] = (tcp_list[:, :3] + 1) / 2 * (
            TRANS_MAX - TRANS_MIN
        ) + TRANS_MIN
    tcp_list[:, 9:] = (tcp_list[:, 9:] + 1) / 2 * (
        HAND_JOINT_UPPER_LIMIT - HAND_JOINT_LOWER_LIMIT
    ) + HAND_JOINT_LOWER_LIMIT
    return tcp_list


def normalize_tactile(tactile_data):
    """
    Normalize tactile sensor data.
    First two channels are x,y forces (-50 to 50) -> normalized to [-1, 1]
    Third channel is normal force (0 to 50) -> normalized to [0, 1]
    """
    tactile = tactile_data.copy()
    tactile[:, :, :2] = (
        np.clip(tactile[:, :, :2], -50, 50) / 50.0
    )  # x,y include shear forces to [-1,1]
    tactile[:, :, 2] = np.clip(tactile[:, :, 2], 0, 50) / 50.0  # normal force to [0,1]
    return tactile


def batched_rotation_difference_angle(A, B):
    """
    Compute the rotation difference angles (in radians) for a batch of rotation matrices.

    Args:
        A (torch.Tensor): First batch of rotation matrices, shape (batch_size, 3, 3).
        B (torch.Tensor): Second batch of rotation matrices, shape (batch_size, 3, 3).

    Returns:
        torch.Tensor: Rotation difference angles in radians, shape (batch_size,).
    """
    # Compute relative rotation matrices: R_diff = A^T * B
    R_diff = torch.matmul(A.transpose(1, 2), B)  # Shape: (batch_size, 3, 3)

    # Compute the trace of each relative rotation matrix
    trace = torch.diagonal(R_diff, dim1=1, dim2=2).sum(-1)  # Shape: (batch_size,)

    # Compute the rotation angles using the formula: theta = acos((trace - 1) / 2)
    theta = torch.acos(
        torch.clamp((trace - 1) / 2, -1, 1)
    )  # Clamp to avoid numerical instability

    return theta
