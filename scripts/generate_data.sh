#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=24

# Change to the script directory
cd "$PROJECT_ROOT/src/adaptac/dataset/gen_data"

# comment depth_mask to use full depth image
# comment crop frame to crop in original frame
python convert_zarr.py \
    --data_dir "$PROJECT_ROOT/test_data/filtered_data" \
    --output_dir "$PROJECT_ROOT/data/test" \
    --frame camera \
    --voxel_size 0.005 \
    --tactile_rep_type 3d_canonical_data \
    --gen_pc True\
    --tactile_frame camera \
    --val_ratio 0.0 \
