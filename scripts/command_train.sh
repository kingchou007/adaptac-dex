#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled

cd "$PROJECT_ROOT/src"

# For single GPU training on localhost
OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=0 torchrun \
    --standalone \
    --nproc_per_node=1 \
    train.py \
    eval_interval=50 \
    save_epochs=1000 \
    batch_size=16 \
    tasks=open_box \
    tasks.dataset_root_path="$PROJECT_ROOT/data" \
    tasks.dataset_name=test \
    tasks.use_color=True \
    tasks.aug=True \
    gen_pc=True \
    aug_jitter=True \
    obs_net_force=False \
    exp_config=111 \
