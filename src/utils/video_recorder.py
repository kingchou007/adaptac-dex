import os
from PIL import Image
import imageio
import numpy as np
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor
import logging


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("video_conversion.log"), logging.StreamHandler()],
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert image sequences to video/gif")
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Input directory path"
    )
    parser.add_argument("--output_dir", type=str, help="Output directory path")
    parser.add_argument(
        "--format",
        type=str,
        choices=["gif", "mp4"],
        default="mp4",
        help="Output format",
    )
    parser.add_argument("--fps", type=int, default=5, help="Frames per second")
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        default=[485, 125, 1035, 675],
        help="Crop parameters: left top right bottom",
    )
    parser.add_argument("--threads", type=int, default=4, help="Number of threads")
    return parser.parse_args()


def process_image(image_path, crop_params, macro_block_size=16):
    """single image processing"""
    try:
        image = Image.open(image_path).crop(crop_params)
        width, height = image.size
        # adjust image size to be multiple of macro block size
        width = (width + macro_block_size - 1) // macro_block_size * macro_block_size
        height = (height + macro_block_size - 1) // macro_block_size * macro_block_size
        return image.resize((width, height))
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return None


def process_folder(folder_path, output_dir, output_format, fps, crop_params):
    """process all images in a folder"""
    try:
        folder_name = os.path.basename(folder_path)
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])

        if not image_files:
            logging.warning(f"No PNG files found in {folder_path}")
            return

        # use thread pool to parallelize image processing
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_image, os.path.join(folder_path, img), crop_params
                )
                for img in image_files
            ]
            images = [f.result() for f in futures if f.result() is not None]

        if not images:
            logging.error(f"No valid images processed in {folder_path}")
            return

        output_file = os.path.join(output_dir, f"{folder_name}.{output_format}")

        if output_format == "gif":
            images[0].save(
                output_file,
                save_all=True,
                append_images=images[1:],
                loop=0,
                optimize=True,
            )
        elif output_format == "mp4":
            imageio.mimsave(output_file, images, fps=fps)

        logging.info(f"Successfully processed {folder_name}")

    except Exception as e:
        logging.error(f"Error processing folder {folder_path}: {e}")


def main():
    args = parse_arguments()
    setup_logging()
    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)
    folders = [
        f
        for f in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, f))
    ]

    logging.info(f"Found {len(folders)} folders to process")
    for folder in tqdm(folders, desc="Processing folders"):
        folder_path = os.path.join(args.input_dir, folder)
        process_folder(folder_path, output_dir, args.format, args.fps, tuple(args.crop))


if __name__ == "__main__":
    main()
