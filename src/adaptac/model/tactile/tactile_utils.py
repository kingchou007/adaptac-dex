import numpy as np
import torch
import torch.nn as nn

from torch_geometric.data import Data, Batch
from adaptac.model.tactile.constants import *


def data_to_gnn_batch(data, edge_type="four"):
    """Convert tensor data to PyTorch Geometric batch format.
    
    Args:
        data: [B, num_nodes, feature_dim] tensor
        edge_type: Type of edge connections ("four", "eight", "all", or combinations)
    
    Returns:
        batch_data: PyTorch Geometric Batch object
        num_batch: Batch size
        num_nodes: Number of nodes per sample
        num_feature_dim: Feature dimension
    """
    assert len(data.shape) == 3, f"Expected 3D tensor, got {len(data.shape)}D"
    
    num_batch, num_nodes, num_feature_dim = data.shape
    edge_index = create_edge_index(num_nodes, edge_type).to(data.device)
    
    data_list = [Data(x=x, edge_index=edge_index) for x in data]
    batch_data = Batch.from_data_list(data_list)
    
    return batch_data, num_batch, num_nodes, num_feature_dim


# Edge index definitions for tip and pulp sensors
# These define the connectivity patterns for 4-connected and 8-connected grids

_TIP_EDGE_INDEX_FOUR = np.array([
    [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 15],
    [1, 2, 6, 2, 1, 3, 7, 3, 2, 4, 8, 4, 3, 5, 5, 4, 9, 6, 1, 7, 11, 7, 2, 6, 8, 12, 8, 3, 7, 13, 9, 5, 10, 15, 10, 9, 11, 6, 12, 12, 7, 11, 13, 13, 8, 12, 14, 14, 13, 15, 15, 9, 14]
]) - 1

_PULP_EDGE_INDEX_FOUR = np.array([
    [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15],
    [1, 2, 6, 2, 1, 3, 7, 3, 2, 4, 8, 4, 3, 5, 9, 5, 4, 10, 6, 1, 7, 11, 7, 2, 6, 8, 12, 8, 3, 7, 9, 13, 9, 4, 8, 10, 14, 10, 5, 9, 15, 11, 6, 12, 12, 7, 11, 13, 13, 8, 12, 14, 14, 9, 13, 15, 15, 10, 14]
]) - 1

_TIP_EDGE_INDEX_EIGHT = np.array([
    [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15],
    [1, 2, 6, 7, 2, 1, 3, 6, 7, 8, 3, 2, 4, 7, 8, 4, 3, 5, 8, 9, 5, 4, 9, 10, 6, 1, 2, 7, 11, 12, 7, 1, 2, 3, 6, 8, 11, 12, 13, 8, 2, 3, 4, 7, 12, 13, 14, 9, 4, 5, 10, 14, 15, 10, 5, 9, 15, 11, 6, 7, 12, 12, 6, 7, 8, 11, 13, 13, 7, 8, 12, 14, 14, 8, 9, 13, 15, 15, 9, 10, 14]
]) - 1

_PULP_EDGE_INDEX_EIGHT = np.array([
    [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15],
    [1, 2, 6, 7, 2, 1, 3, 6, 7, 8, 3, 2, 4, 7, 8, 9, 4, 3, 5, 8, 9, 10, 5, 4, 9, 10, 6, 1, 2, 7, 11, 12, 7, 1, 2, 3, 6, 8, 11, 12, 13, 8, 2, 3, 4, 7, 9, 12, 13, 14, 9, 3, 4, 5, 8, 10, 13, 14, 15, 10, 4, 5, 9, 14, 15, 11, 6, 7, 12, 12, 6, 7, 8, 11, 13, 13, 7, 8, 9, 12, 14, 14, 8, 9, 10, 13, 15, 15, 9, 10, 14]
]) - 1

_SENSOR_EDGE_INDEX = np.array([
    [6, 25, 21, 21, 21, 36, 55, 51, 51, 51, 66, 85, 81, 81, 81, 96, 115, 111, 111, 111],
    [25, 6, 51, 81, 111, 55, 36, 21, 81, 111, 85, 66, 21, 51, 111, 115, 96, 21, 51, 81]
]) - 1


def _get_base_edge_indices(edge_type):
    """Get base edge indices for tip and pulp sensors based on edge type."""
    if "four" in edge_type:
        return _TIP_EDGE_INDEX_FOUR, _PULP_EDGE_INDEX_FOUR
    elif "eight" in edge_type:
        return _TIP_EDGE_INDEX_EIGHT, _PULP_EDGE_INDEX_EIGHT
    else:
        raise ValueError(f"Invalid edge_type: {edge_type}. Must contain 'four' or 'eight'.")


def _build_sensor_edge_index(tip_edge_index, pulp_edge_index, tip_points_number, pulp_points_number):
    """Build edge index for all sensors by concatenating tip and pulp patterns."""
    edge_index_list = []
    start_idx = 0
    
    expected_sensors = ["thumb_tip", "thumb_pulp", "index_tip", "index_pulp", 
                       "middle_tip", "middle_pulp", "ring_tip", "ring_pulp"]
    assert list(TACTILE_INFO.keys()) == expected_sensors, "TACTILE_INFO keys mismatch"

    for sensor in TACTILE_INFO:
        if "tip" in sensor:
            edge_index_list.append(start_idx + tip_edge_index)
            start_idx += tip_points_number
        elif "pulp" in sensor:
            edge_index_list.append(start_idx + pulp_edge_index)
            start_idx += pulp_points_number
    
    return edge_index_list


def create_edge_index(num_nodes, edge_type="four"):
    """Create edge index for tactile sensor graph.
    
    Args:
        num_nodes: Total number of nodes in the graph
        edge_type: Type of edge connections. Can be:
            - "four": 4-connected grid (up, down, left, right)
            - "eight": 8-connected grid (includes diagonals)
            - "all": Fully connected graph
            - Can include "sensor" suffix to add inter-sensor connections
    
    Returns:
        edge_index: [2, num_edges] tensor of edge connections
    """
    # Handle fully connected case
    if "all" in edge_type:
        row, col = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing="ij")
        return torch.stack([row.flatten(), col.flatten()], dim=0)

    # Get point counts
    tip_points_number = PAXINI_TIP_COORD.shape[0]
    pulp_points_number = PAXINI_PULP_COORD.shape[0]
    assert tip_points_number == pulp_points_number, "Tip and pulp must have same number of points"

    # Get base edge indices
    tip_edge_index, pulp_edge_index = _get_base_edge_indices(edge_type)

    # Build edge index for all sensors
    edge_index_list = _build_sensor_edge_index(
        tip_edge_index, pulp_edge_index, tip_points_number, pulp_points_number
    )

    # Add inter-sensor connections if requested
    if "sensor" in edge_type:
        edge_index_list.append(_SENSOR_EDGE_INDEX)

    # Concatenate and convert to tensor
    edge_index = np.concatenate(edge_index_list, axis=1)
    return torch.from_numpy(edge_index)


def create_activation(name):
    """Create activation function by name.
    
    Args:
        name: Name of activation function ("relu", "gelu", "prelu", "elu", or None)
    
    Returns:
        Activation function module
    """
    activation_map = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "prelu": nn.PReLU(),
        "elu": nn.ELU(),
        None: nn.Identity(),
    }
    
    if name not in activation_map:
        raise NotImplementedError(f"Activation '{name}' is not implemented. Available: {list(activation_map.keys())}")
    
    return activation_map[name]
