import os
import cv2
import numpy as np
from typing import Optional, Tuple
from adaptac.dataset.utils.constants import IMG_MEAN, IMG_STD
from adaptac.model.tactile.tactile_processer import compute_resultant
import open3d as o3d
import visualizer
from termcolor import cprint


def process_tactile_data(
    window_array: np.ndarray,
    tactile_level: str,
) -> Tuple[np.ndarray, float, int]:
    """Process tactile data based on specified tactile level."""

    max_force_magnitude = None
    max_force_index = None

    # if tactile_level == "hand":
    # Process hand-level tactile data (compute single resultant force per frame)
    resultant_forces = []

    # # 5,120,12
    if tactile_level == "hand":
        for frame_idx in range(window_array.shape[0]):
            coords = window_array[frame_idx][:, :6]
            forces = window_array[frame_idx][:, 9:12]
            resultant = compute_resultant(coords, forces, "force")
            resultant_forces.append(resultant)

        window_array = np.stack(resultant_forces, axis=0)

        # Compute max force magnitude and index
        force_magnitudes = np.linalg.norm(window_array, axis=1)
        max_force_magnitude = np.max(force_magnitudes)
        max_force_index = np.argmax(force_magnitudes)

    elif tactile_level == "point":
        for frame_idx in range(window_array.shape[0]):
            coords = window_array[frame_idx][:, :6]
            forces = window_array[frame_idx][:, 9:12]
            point_force_magnitude = np.linalg.norm(forces, axis=1)
            max_value = np.max(point_force_magnitude)
            resultant_forces.append(max_value)

        window_array = np.stack(resultant_forces, axis=0)
        max_force_magnitude = np.max(window_array)
        max_force_index = np.argmax(window_array)

    else:
        raise ValueError(
            f"Unsupported tactile level: {tactile_level}. Use 'hand' or 'point'."
        )

    return window_array, max_force_magnitude, max_force_index


def handle_existing_data(output_dir: str, force_overwrite: bool = True) -> None:
    """Handle existing data at the output directory."""
    if os.path.exists(output_dir):
        print("")
        cprint(f"Data already exists at {output_dir}", "yellow")

        if force_overwrite:
            cprint(f"Overwriting existing data at {output_dir}", "red")
            os.system(f"rm -rf {output_dir}")
            print("")
        else:
            cprint(
                "If you want to overwrite, delete the existing directory first.",
                "yellow",
            )
            user_input = input("Overwrite? (y/n): ")
            if user_input.lower() == "y":
                cprint(f"Overwriting existing data at {output_dir}", "red")
                os.system(f"rm -rf {output_dir}")
                print("")
            else:
                cprint("Exiting", "yellow")
                exit()

    os.makedirs(output_dir, exist_ok=True)
    cprint(f"Creating new data at {output_dir}", "green")


def visualize_point_cloud(
    cropped_pc: np.ndarray,
    tactile_data: Optional[np.ndarray] = None,
    DEBUG: bool = False,
    DEBUG_SSH: bool = False,
) -> None:
    """Visualize point cloud data for debugging."""
    if DEBUG:
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(cropped_pc[:, :3])
        o3d_pc.colors = o3d.utility.Vector3dVector(cropped_pc[:, 3:6])
        o3d.visualization.draw_geometries([o3d_pc])
    elif DEBUG_SSH and tactile_data is not None:
        cropped_pc[:, 3:6] = cropped_pc[:, 3:6] * IMG_STD + IMG_MEAN
        cropped_pc[:, 3:6] *= 255
        tactile_points = tactile_data[:, :3]
        tactile_colors = np.zeros_like(tactile_points)
        tactile_colors[:, 0] = 255
        tactile_pc = np.append(tactile_points, tactile_colors, 1)
        all_pc = np.append(cropped_pc, tactile_pc, 0)
        visualizer.visualize_pointcloud(all_pc)


def save_tactile_debug_data(
    check_tactile_img_arrays,
    check_tactile_tactile_arrays,
    demo_name,
    base_dir="tactile_debug1",
):
    """
    Save tactile data and corresponding images as visualization debug files.

    Args:
        check_tactile_img_arrays (list): List containing RGB images
        check_tactile_tactile_arrays (list): List containing force values
        demo_name (str): Demonstration name, used to create subdirectory
        base_dir (str): Base directory name, defaults to 'tactile_debug'
    """
    if not check_tactile_img_arrays or len(check_tactile_img_arrays) == 0:
        print("No visualization data to save")
        return

    try:
        # Create demonstration-specific directories
        tactile_debug_dir = os.path.join(base_dir)
        os.makedirs(tactile_debug_dir, exist_ok=True)

        demo_tactile_dir = os.path.join(tactile_debug_dir, demo_name)
        os.makedirs(demo_tactile_dir, exist_ok=True)

        print(f"Saving tactile debug data for {demo_name} to {demo_tactile_dir}")

        for j in range(len(check_tactile_tactile_arrays)):
            try:
                # Get force value and image for current frame
                total_force = np.max(check_tactile_tactile_arrays[j])
                debug_img = check_tactile_img_arrays[j].copy()

                # Convert color space and add text annotation
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
                cv2.putText(
                    debug_img,
                    f"Force: {total_force:.4f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                # Build save paths
                img_path = os.path.join(demo_tactile_dir, f"{j}.png")
                npy_path = os.path.join(demo_tactile_dir, f"{j}.npy")

                # Save data and image
                np.save(npy_path, check_tactile_tactile_arrays[j])
                cv2.imwrite(img_path, debug_img)

            except Exception as e:
                from termcolor import cprint

                cprint(
                    f"Error saving tactile debug data for frame {j}: {str(e)}", "red"
                )

        print(
            f"Successfully saved {len(check_tactile_tactile_arrays)} frames of tactile debug data"
        )

    except Exception as e:
        from termcolor import cprint

        cprint(f"Error saving tactile debug data: {str(e)}", "red")
        import traceback

        traceback.print_exc()
