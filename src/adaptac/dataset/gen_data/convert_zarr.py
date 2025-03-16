import os
import cv2
import zarr
import click
import numpy as np
from tqdm import tqdm
from termcolor import cprint
from typing import Tuple, List, Optional, Dict, Union

from adaptac.dataset.utils.constants import *
from adaptac.dataset.gen_data.process_obs import (
    get_point_cloud,
    crop_pc,
    convert_tcp_data_to_camera,
    get_processed_tactile_data,
)
from adaptac.dataset.gen_data.convert_util import (
    save_tactile_debug_data,
    visualize_point_cloud,
    handle_existing_data,
    process_tactile_data,
)

# Window size parameters - sampling rate is 5hz
# ! set window num to 0 if train ours policy
WINDOW_SIZE = 5  # 4 before + current + 4 after
HALF_WINDOW = 2  # 2 steps before and after

# DEBUG=True to visualize point clouds
DEBUG = False
DEBUG_SSH = False
VIS_TACTILE_DIFF = False
# Compression settings (adjust clevel for size/speed tradeoff)
DEFAULT_COMPRESSOR = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
DEFAULT_CHUNK_SIZE = 100
DICT_SPACE = 30000

TACTILE_LEVEL = "hand"  # hand, pad, point


class DatasetConfig:
    """Configuration class for dataset conversion parameters."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        frame: str,
        voxel_size: float,
        tactile_rep_type: str,
        crop_frame: Optional[str] = None,
        gen_pc: bool = True,
        tactile_frame: Optional[str] = None,
        val_ratio: float = 0.0,
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.frame = frame
        self.voxel_size = voxel_size
        self.tactile_rep_type = tactile_rep_type
        self.crop_frame = crop_frame
        self.gen_pc = gen_pc
        self.tactile_frame = tactile_frame
        self.val_ratio = val_ratio


def initialize_data_containers() -> Dict[str, List]:
    """Initialize data containers for storing different types of data."""
    return {
        "img": [],
        "depth": [],
        "point_cloud": [],
        "input_coords": [],
        "input_feats": [],
        "input_index": [],
        "state": [],
        "action": [],
        "tactile": [],
        "time": [],
        "episode_ends": [],
    }


def load_demo_files(demo_dir: str) -> List[str]:
    """Load demonstration files from a directory, sorted by numeric value."""
    if not os.path.exists(demo_dir):
        raise FileNotFoundError(f"Demonstration directory not found: {demo_dir}")

    # Extract only files that contain digits and sort numerically
    demo_files = sorted(
        os.listdir(demo_dir), key=lambda f: int("".join(filter(str.isdigit, f)) or 0)
    )

    return demo_files


# TODO: add this to sensor process
def get_cropped_point_cloud(
    color_image: np.ndarray,
    depth_image: np.ndarray,
    voxel_size: float,
    frame: str,
    crop_frame: Optional[str] = None,
) -> np.ndarray:
    """Generate and crop point cloud data from color and depth images."""
    points, colors = get_point_cloud(color_image, depth_image, voxel_size, frame)
    return crop_pc(points, colors, frame, crop_frame)


# def extract_demo_data(demo: np.ndarray) -> Tuple:
#     """Extract and preprocess data from a demonstration step."""
#     # Check if the required keys are present in the demonstration data
#     try:
#         return (
#             demo["hand_joint_positions"],
#             demo["hand_commanded_joint_position"],
#             demo["arm_ee_pose"],
#             demo["arm_commanded_ee_pose"],
#             cv2.cvtColor(demo["camera_1_color_image"], cv2.COLOR_BGR2RGB),
#             demo["camera_1_depth_image"],
#             demo["arm_joint_positions"],
#             demo["time"],
#         )
#     except KeyError as e:
#         raise KeyError(f"Missing required key in demonstration data: {e}")


def transform_ee_pose_frame(ee_pose: np.ndarray, frame: str) -> np.ndarray:
    """Transform end effector pose to the specified reference frame."""
    if frame == "camera":
        return convert_tcp_data_to_camera(ee_pose)
    elif frame == "base":
        return ee_pose
    else:
        raise ValueError(f"Unsupported frame type: {frame}. Use 'camera' or 'base'.")


def process_demo_step(
    step_path: str,
    frame: str,
    voxel_size: float,
    tactile_rep_type: str,
    crop_frame: Optional[str] = None,
    tactile_frame: Optional[str] = None,
    gen_pc: bool = True,
) -> Tuple:
    """Process a single step of demonstration data."""
    try:
        demo = np.load(step_path, allow_pickle=True)

        # Extract real data from demo file
        hand_joint_positions = demo["hand_joint_positions"]  # (16,)
        hand_commanded_joint_position = demo["hand_commanded_joint_position"]  # (16,)
        color_image = cv2.cvtColor(demo["camera_1_color_image"], cv2.COLOR_BGR2RGB)
        time = demo["time"]

        # Convert arm data from tuples to arrays and extend 6D to 7D
        arm_ee_pose_6d = np.array(demo["arm_ee_pose"], dtype=np.float32)  # (6,)
        arm_joint_state_6d = np.array(demo["arm_joint_positions"], dtype=np.float32)  # (6,)

        # Extend to 7D by adding a dummy component
        arm_ee_pose = np.concatenate([arm_ee_pose_6d, [1.0]])  # (7,)
        arm_joint_state = np.concatenate([arm_joint_state_6d, [0.0]])  # (7,)

        # Use arm_ee_pose as commanded pose (since arm_commanded_ee_pose is missing)
        arm_commanded_ee_pose = arm_ee_pose.copy()

        # Create dummy depth image (same size as color image)
        h, w = color_image.shape[:2]
        depth_image = np.ones((h, w), dtype=np.float32) * 500.0  # Dummy depth at 500mm

        # Ensure data is in the correct format
        state_data = {
            "arm_abs_joint": np.array([arm_joint_state]),
            "hand_abs_joint": np.array([hand_joint_positions]),
        }
        input_tactile = np.array([demo["tactile_data"]])

        tactile_data = get_processed_tactile_data(
            input_tactile, state_data, tactile_rep_type, tactile_frame
        )

        # Transform poses to the specified frame
        transformed_ee_pose = transform_ee_pose_frame(arm_ee_pose, frame)
        transformed_action_ee_pose = transform_ee_pose_frame(
            arm_commanded_ee_pose, frame
        )

        # Construct state and action vectors
        state = np.concatenate([transformed_ee_pose, hand_joint_positions], axis=0)
        action = np.concatenate(
            [transformed_action_ee_pose, hand_commanded_joint_position], axis=0
        )

        # Generate point cloud data
        if gen_pc:
            cropped_pc = get_cropped_point_cloud(
                color_image, depth_image, voxel_size, frame, crop_frame
            )
            # If point cloud is all zeros, generate synthetic data in range [0, 1]
            if np.all(cropped_pc == 0):
                # Create random point cloud with 30000 points, 6 channels (xyz + rgb)
                # XYZ in range [0, 1], RGB normalized to [0, 1]
                cropped_pc = np.random.rand(30000, 6).astype(np.float32)
        else:
            cropped_pc = np.zeros((30000, 6), dtype=np.float32)

        return color_image, depth_image, tactile_data, state, action, cropped_pc, time

    except Exception as e:
        cprint(f"Error processing demo step {step_path}: {str(e)}", "red")
        raise


def get_val_mask(n_episodes: int, val_ratio: float, seed: int = 0) -> np.ndarray:
    """Generate a boolean mask for validation data selection."""
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # Have at least 1 episode for validation, and at least 1 episode for training
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def save_zarr_dataset(
    zarr_group: zarr.Group,
    name: str,
    data: Union[List, np.ndarray],
    chunks: Optional[Tuple] = None,
    dtype: Optional[str] = None,
    compressor: Optional[zarr.Blosc] = None,
) -> None:
    """Helper function to save dataset to zarr group with standard parameters."""
    # Skip empty datasets
    if (
        data is None
        or (isinstance(data, np.ndarray) and data.size == 0)
        or (isinstance(data, list) and len(data) == 0)
    ):
        return

    # Convert list to numpy array if needed
    if isinstance(data, list):
        try:
            data = np.stack(data, axis=0)
        except ValueError:
            cprint(f"Could not stack data for {name}, skipping", "red")
            return

    # Use data's dtype if not specified
    if dtype is None:
        dtype = data.dtype

    # Use default compressor if not specified
    if compressor is None:
        compressor = DEFAULT_COMPRESSOR

    # Set default chunks if not specified
    if chunks is None:
        chunks = (DEFAULT_CHUNK_SIZE,) + data.shape[1:]

    # Create dataset
    try:
        zarr_group.create_dataset(
            name, data=data, chunks=chunks, dtype=dtype, compressor=compressor
        )
    except Exception as e:
        cprint(f"Error saving {name} dataset: {str(e)}", "red")


def _preload_tactile_data(
    demo_dir: str,
    demo_files: List[str],
    frame: str,
    voxel_size: float,
    tactile_rep_type: str,
    crop_frame: Optional[str],
    tactile_frame: Optional[str],
) -> List:
    """Pre-load all tactile data for a demonstration."""
    tactile_data_list = []
    for step_idx, demo_file in enumerate(demo_files):
        step_path = os.path.join(demo_dir, demo_file)
        try:
            _, _, tactile_data, _, _, _, _ = process_demo_step(
                step_path,
                frame,
                voxel_size,
                tactile_rep_type,
                crop_frame,
                tactile_frame,
                gen_pc,
            )

            tactile_data_list.append(tactile_data)  # [(120, 12)]

        except Exception as e:
            cprint(
                f"Error pre-loading tactile data for {step_path}: {str(e)}", "yellow"
            )
            tactile_data_list.append(None)  # Placeholder for failed steps
    return tactile_data_list


def _generate_tactile_window(
    step_idx: int,
    total_steps: int,
    tactile_data_list: List,
    window_size: int,
    half_window: int,
    current_tactile_data,
) -> np.ndarray:
    """Generate a tactile window for the current step from pre-loaded data."""
    window_start = max(0, step_idx - half_window)
    window_end = min(total_steps, step_idx + half_window + 1)
    current_window = tactile_data_list[window_start:window_end]

    # Pad with zeros if window is incomplete
    if len(current_window) < window_size:
        padding_size = window_size - len(current_window)

        zero_padding = (
            np.zeros_like(current_tactile_data)
            if current_tactile_data is not None
            else np.zeros((120, 12), dtype=np.float32)
        )  # Default shape

        if step_idx < half_window:  # Pad at the beginning
            current_window = [zero_padding] * padding_size + current_window
        else:  # Pad at the end
            current_window.extend([zero_padding] * padding_size)

    return np.stack(current_window[:window_size], axis=0)


def convert_to_zarr(
    data: List[str],
    data_dir: str,
    output_dir: str,
    frame: str,
    voxel_size: float,
    tactile_rep_type: str,
    crop_frame: Optional[str],
    gen_pc: bool,
    tactile_frame: Optional[str],
) -> None:
    """Convert demonstration data to zarr format with optional windowed tactile data"""
    # Initialize data containers
    data_containers = initialize_data_containers()
    total_count = 0

    # Unpack data containers for easier access
    img_arrays = data_containers["img"]
    depth_arrays = data_containers["depth"]
    point_clouds = data_containers["point_cloud"]
    input_coords_list = data_containers["input_coords"]
    input_feats_list = data_containers["input_feats"]
    input_index = data_containers["input_index"]
    state_arrays = data_containers["state"]
    action_arrays = data_containers["action"]
    tactile_arrays = data_containers["tactile"]
    time_arrays = data_containers["time"]
    episode_ends_arrays = data_containers["episode_ends"]

    # Process each demonstration
    for demo_name in tqdm(data, desc="Processing demonstrations"):
        demo_dir = os.path.join(data_dir, demo_name)
        demo_files = load_demo_files(demo_dir)
        num_steps = len(demo_files)

        if VIS_TACTILE_DIFF:
            check_tactile_img_arrays = []
            check_tactile_tactile_arrays = []

        # Pre-load all tactile data for this demonstration if WINDOW_SIZE != 0
        all_tactile_data = []
        if WINDOW_SIZE != 0:
            all_tactile_data = _preload_tactile_data(
                demo_dir,
                demo_files,
                frame,
                voxel_size,
                tactile_rep_type,
                crop_frame,
                tactile_frame,
            )

        # Process all steps within the demonstration
        for step_idx in range(num_steps):
            total_count += 1
            step_path = os.path.join(demo_dir, demo_files[step_idx])
            try:
                # Process current step (full data)
                (
                    color_image,
                    depth_image,
                    tactile_data,
                    state,
                    action,
                    cropped_pc,
                    time,
                ) = process_demo_step(
                    step_path,
                    frame,
                    voxel_size,
                    tactile_rep_type,
                    crop_frame,
                    tactile_frame,
                    gen_pc,
                )

                # Debug visualization if enabled
                if DEBUG or DEBUG_SSH:
                    visualize_point_cloud(
                        cropped_pc,
                        tactile_data,
                        DEBUG=DEBUG,
                        DEBUG_SSH=DEBUG_SSH,
                    )

                # Store current step data
                img_arrays.append(color_image)
                depth_arrays.append(depth_image)
                state_arrays.append(state)
                action_arrays.append(action)
                if WINDOW_SIZE == 0:
                    tactile_arrays.append(tactile_data)
                time_arrays.append(time)

                if gen_pc:
                    pc_padding = np.zeros((DICT_SPACE, 6), dtype=np.float32)
                    pc_padding[: len(cropped_pc), :] = cropped_pc
                    point_clouds.append(pc_padding)
                    input_index.append(len(cropped_pc))

                # Generate tactile window if windowing is enabled
                if WINDOW_SIZE != 0:
                    window_array = _generate_tactile_window(
                        step_idx,
                        num_steps,
                        all_tactile_data,
                        WINDOW_SIZE,
                        HALF_WINDOW,
                        tactile_data,
                    )

                    tactile_arrays.append(window_array)

                    # TODO: debug visualization, need to be removed in the final version
                    if VIS_TACTILE_DIFF:
                        window_array, force_magnitudes, max_force_index = (
                            process_tactile_data(
                                window_array,
                                TACTILE_LEVEL,
                            )
                        )
                        check_tactile_img_arrays.append(color_image)
                        check_tactile_tactile_arrays.append(window_array)

            except Exception as e:
                cprint(f"Error processing step {step_path}: {str(e)}", "red")
                cprint("Skipping this step and continuing...", "yellow")
                continue

        # Mark the end of each episode
        episode_ends_arrays.append(total_count)

        if VIS_TACTILE_DIFF and len(check_tactile_img_arrays) > 0:
            save_tactile_debug_data(
                check_tactile_img_arrays, check_tactile_tactile_arrays, demo_name
            )

    # Create zarr file structure
    os.makedirs(output_dir, exist_ok=True)
    zarr_root = zarr.group(output_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")
    compressor = DEFAULT_COMPRESSOR

    # Save all datasets
    save_zarr_dataset(zarr_data, "img", img_arrays, None, "uint8", compressor)
    save_zarr_dataset(zarr_data, "depth", depth_arrays, None, "float64", compressor)
    save_zarr_dataset(zarr_data, "tactile", tactile_arrays, None, "float32", compressor)

    # Save point cloud data if generated
    if gen_pc and point_clouds:
        try:
            point_clouds_array = np.stack(point_clouds, axis=0)
            chunks = (DEFAULT_CHUNK_SIZE,) + point_clouds_array.shape[1:]
            save_zarr_dataset(
                zarr_data,
                "point_cloud",
                point_clouds_array,
                chunks,
                "float32",
                compressor,
            )
            input_index_array = np.array(input_index)
            save_zarr_dataset(
                zarr_data,
                "input_index",
                input_index_array,
                (DEFAULT_CHUNK_SIZE,),
                "int64",
                compressor,
            )
        except Exception as e:
            cprint(f"Error saving point cloud data: {str(e)}", "red")

    # Save coordinate and feature data if available
    save_zarr_dataset(
        zarr_data, "input_coords", input_coords_list, None, "int32", compressor
    )
    save_zarr_dataset(
        zarr_data, "input_feats", input_feats_list, None, "float32", compressor
    )

    # Convert remaining lists to arrays and save
    try:
        if state_arrays:
            state_array = np.stack(state_arrays, axis=0)
            chunks = (DEFAULT_CHUNK_SIZE, state_array.shape[1])
            save_zarr_dataset(
                zarr_data, "state", state_array, chunks, "float32", compressor
            )
        if action_arrays:
            action_array = np.stack(action_arrays, axis=0)
            chunks = (DEFAULT_CHUNK_SIZE, action_array.shape[1])
            save_zarr_dataset(
                zarr_data, "action", action_array, chunks, "float32", compressor
            )
        if time_arrays:
            time_array = np.array(time_arrays)
            save_zarr_dataset(
                zarr_data, "time", time_array, None, "float32", compressor
            )
        if episode_ends_arrays:
            episode_ends_array = np.array(episode_ends_arrays)
            save_zarr_dataset(
                zarr_meta,
                "episode_ends",
                episode_ends_array,
                (DEFAULT_CHUNK_SIZE,),
                "int64",
                compressor,
            )
    except Exception as e:
        cprint(f"Error saving arrays: {str(e)}", "red")

    # Print summary of dataset
    print_dataset_summary(data_containers, gen_pc)
    cprint(f"Saved zarr file to {output_dir}", "green")


def print_dataset_summary(data_containers: Dict[str, List], gen_pc: bool) -> None:
    """Print summary of dataset shapes."""
    cprint("Final dataset shapes:", "cyan")

    # Helper function to print shape if data exists
    def print_shape(name, data):
        if isinstance(data, list) and len(data) > 0:
            try:
                shape = np.array(data).shape
                cprint(f"  {name}: {shape}", "cyan")
            except:
                cprint(f"  {name}: {len(data)} items (varied shapes)", "cyan")
        elif isinstance(data, np.ndarray) and data.size > 0:
            cprint(f"  {name}: {data.shape}", "cyan")

    # Print shapes for all data types
    for key, data in data_containers.items():
        if key == "point_cloud" and not gen_pc:
            continue
        print_shape(key, data)


@click.command()
@click.option(
    "--data_dir",
    type=str,
    required=True,
    help="Path to the directory containing the data.",
)
@click.option(
    "--output_dir",
    type=str,
    required=True,
    help="Path to save the data in zarr format.",
)
@click.option(
    "--frame", type=str, default="camera", help="Frame to transform the point cloud to."
)
@click.option(
    "--voxel_size", type=float, default=0.005, help="Voxel size for voxelization."
)
@click.option("--crop_frame", type=str, default=None, help="Crop in which frame.")
@click.option("--gen_pc", type=bool, default=True, help="Generate point cloud.")
@click.option("--tactile_frame", type=str, default=None, help="Tactile frame.")
@click.option("--val_ratio", type=float, default=0.1, help="Validation dataset ratio.")
@click.option(
    "--tactile_rep_type",
    type=str,
    default="sensor",
    help="Type of tactile representation to use.",
)
@click.option(
    "--force_overwrite", type=bool, default=True, help="Force overwrite existing data."
)
def main(
    data_dir: str,
    output_dir: str,
    frame: str,
    voxel_size: float,
    tactile_rep_type: str,
    crop_frame: str,
    gen_pc: bool,
    tactile_frame: str,
    val_ratio: float,
    force_overwrite: bool,
):
    """Convert demonstration data to zarr format."""
    try:
        # Create configuration object
        config = DatasetConfig(
            data_dir=data_dir,
            output_dir=output_dir,
            frame=frame,
            voxel_size=voxel_size,
            tactile_rep_type=tactile_rep_type,
            crop_frame=crop_frame,
            gen_pc=gen_pc,
            tactile_frame=tactile_frame,
            val_ratio=val_ratio,
        )

        # List and sort demonstration directories
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        data = os.listdir(data_dir)
        data.sort(key=lambda f: int("".join(filter(str.isdigit, f)) or 0))

        if not data:
            raise ValueError(f"No demonstration directories found in {data_dir}")

        # Handle existing data at the output location
        handle_existing_data(output_dir, force_overwrite)

        # Split data into training and validation sets
        val_mask = get_val_mask(len(data), val_ratio)
        train_data = np.array(data)[~val_mask].tolist()
        val_data = np.array(data)[val_mask].tolist()

        cprint(f"Processing {len(train_data)} training demonstrations", "blue")

        # Process training data
        convert_to_zarr(
            train_data,
            data_dir,
            os.path.join(output_dir, "train"),
            frame,
            voxel_size,
            tactile_rep_type,
            crop_frame,
            gen_pc,
            tactile_frame,
        )

        # Process validation data if needed
        if val_ratio > 0:
            cprint(f"Processing {len(val_data)} validation demonstrations", "blue")
            convert_to_zarr(
                val_data,
                data_dir,
                os.path.join(output_dir, "val"),
                frame,
                voxel_size,
                tactile_rep_type,
                crop_frame,
                gen_pc,
                tactile_frame,
            )

        cprint("Dataset conversion completed successfully!", "green")

    except Exception as e:
        cprint(f"Error in dataset conversion: {str(e)}", "red")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    main()
