[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2505.13982-b31b1b.svg)](https://arxiv.org/abs/2505.13982)
[![Docs](https://img.shields.io/badge/docs-available-brightgreen)](https://adaptac-dex.github.io/)
[![GitHub stars](https://img.shields.io/github/stars/kingchou007/adaptac-dex?style=social)](https://github.com/kingchou007/adaptac-dex)

# AdapTac-Dex: Adaptive Visuo-Tactile Fusion with Predictive Force Attention for Dexterous Manipulation


<p align="center">
  <img src="docs/static/iros.png" alt="IROS Logo" height="100"/>
</p>

<p align="center">
  <b>Authors:</b><br>
  <a href="https://kingchou007.github.io/">Jinzhou Li*</a> ·
  <a href="https://tianhaowuhz.github.io/">Tianhao Wu*</a> ·
  <a href="https://jiyao06.github.io/">Jiyao Zhang**</a> ·
  <a href="https://chenzyn.github.io/">Zeyuan Chen**</a> ·
  <a href="">Haotian Jin</a> ·
  <a href="https://aaronanima.github.io/">Mingdong Wu</a> ·
  <a href="https://shenyujun.github.io/">Yujun Shen</a> ·
  <a href="https://www.yangyaodong.com/">Yaodong Yang</a> ·
  <a href="https://zsdonghao.github.io/">Hao Dong†</a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2505.13982"><b>Paper</b></a>&nbsp;|&nbsp;
  <a href="https://adaptac-dex.github.io/"><b>Website</b></a>&nbsp;|&nbsp;
  <a href="https://www.youtube.com/watch?v=Aq34cDWNBE8"><b>Video</b></a>&nbsp;|&nbsp;
  <a href="https://drive.google.com/drive/folders/1dJnF192aBb8VxeNBQRBstrCrC3GcPKho"><b>Hardware</b></a>&nbsp;|&nbsp;
  <a href="https://github.com/real-dex-suite/REAL-ROBO"><b>Teleoperation</b></a>
</p>


## Abstract

Effectively utilizing multi-sensory data is important for robots to generalize across diverse tasks. However, the heterogeneous nature of these modalities makes fusion challenging. Existing methods propose strategies to obtain comprehensively fused features but often ignore the fact that each modality requires different levels of attention at different manipulation stages. To address this, we propose a force-guided attention fusion module that adaptively adjusts the weights of visual and tactile features without human labeling. We also introduce a self-supervised future force prediction auxiliary task to reinforce the tactile modality, improve data imbalance, and encourage proper adjustment. Our method achieves an average success rate of 93% across three fine-grained, contact-rich tasks in real-world experiments. Further analysis shows that our policy appropriately adjusts attention to each modality at different manipulation stages.

## TODO

- [x] Add the hardware setup, [here](https://drive.google.com/drive/folders/1dJnF192aBb8VxeNBQRBstrCrC3GcPKho)
- [x] Released the pre-trained [code](https://github.com/tianhaowuhz/3dtacdex) and [checkpoint](https://huggingface.co/kingchou007/3dTacDex/tree/main)
  - We recommend training your own model using the provided code.

- [x] Release the init code
- [ ] Further clean up the code
- [ ] Release the dataset
- [ ] Clean Teleopeartion Code
- [ ] Release the tutorial

## Quick Start

1. **Clone the repository** with submodules:
   ```bash
   git clone --recursive https://github.com/kingchou007/adaptac-dex.git
   cd adaptac-dex
   ```

2. **Install dependencies** (see [Installation](#installation) section below)
3. **Generate training dataset**: Configure parameters in `scripts/generate_data.sh` and run:
   ```bash
   bash scripts/generate_data.sh
   ```
4. **Train your policy**: Modify configs in `src/adaptac/configs/tasks/` and run:
   ```bash
   bash scripts/command_train.sh
   ```
5. **Evaluate your policy**: Run:
   ```bash
   bash scripts/command_eval.sh
   ```

For detailed instructions, see the [Training Tutorial](#training-tutorial) section.

---

# Installation

## Prerequisites

- **OS**: Ubuntu 20.04 (tested) or compatible Linux distribution
- **CUDA**: 11.8 (recommended to avoid compatibility issues)
- **Python**: 3.8
- **Conda**: Anaconda or Miniconda
- **Git**: For cloning repositories

## Clone Repository

Clone the repository with submodules:

```bash
git clone --recursive https://github.com/kingchou007/adaptac-dex.git
cd adaptac-dex
```

If you've already cloned the repository without submodules, initialize them:

```bash
git submodule update --init --recursive
```

## Environment Setup

Please follow the instructions to install the conda environments and the dependencies of the codebase. We recommend using CUDA 11.8 during installations to avoid compatibility issues.

1. Create a new conda environment and activate the environment.
    ```bash
    conda create -n adaptac python=3.8
    conda activate adaptac
    ```

2. Install necessary dependencies.
    ```bash
    conda install cudatoolkit=11.8
    pip install -r requirements.txt
    ```

3. Install [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) manually following [the official installation instructions](https://github.com/NVIDIA/MinkowskiEngine?tab=readme-ov-file#cuda-11x).
    
    **Note:** MinkowskiEngine is included as a Git submodule. If you cloned with `--recursive`, it should already be available. Otherwise, initialize submodules first (see [Clone Repository](#clone-repository) section).
    
    ```bash
    cd dependencies
    conda install openblas-devel -c anaconda
    export CUDA_HOME=/usr/local/cuda-11.8
    git clone https://github.com/NVIDIA/MinkowskiEngine.git
    cd MinkowskiEngine
    python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
    cd ../..
    ```

4. Install [Pytorch3D](https://github.com/facebookresearch/pytorch3d) manually.
    
    **Note:** Pytorch3D is included as a Git submodule. If you cloned with `--recursive`, it should already be available. Otherwise, initialize submodules first (see [Clone Repository](#clone-repository) section).
    
    ```bash
    cd dependencies/pytorch3d
    pip install -e .
    cd ../..
    ```
5. (Optional) If you'd like to visualize point clouds in service, install the visualizer package:

    ```bash
    # Install Plotly and Kaleido for point cloud visualization
    pip install kaleido plotly

    cd dependencies
    cd visualizer && pip install -e .
    cd ../..
    ```

## Real Robot Setup

### Hardware Requirements
- [Flexiv Rizon 4](https://www.flexiv.com/products/rizon?reload=1764163581143) Robotic Arm
- Leap Hand: Please refer to the [LEAP Hand API repository](https://github.com/leap-hand/LEAP_Hand_API) for installation and setup instructions
- Intel [RealSense](https://www.intel.com/content/www/us/en/architecture-and-technology/realsense-overview.html) RGB-D Camera (D415/D435/L515)
- [Paxini](https://omnisharingdb.paxini.com/) Tactile sensor

### Software Requirements
- Ubuntu 20.04 (tested) with previous environment installed
- If you are using Intel RealSense RGB-D camera, install the python wrapper `pyrealsense2` of `librealsense` according to [the official installation instructions](https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python#installation).


## Training Tutorial

### 1. Generate Training Dataset

Before running the script, please configure these key parameters in `scripts/generate_data.sh`:

- `data_dir`: Directory for raw dataset
- `output_dir`: Directory for processed dataset
- `frame`: Transform data in 'camera' or 'base' frame
- `tactile_rep_type`: Type of tactile representation

Then run:
```bash
bash scripts/generate_data.sh
```

### 2. Train Your Policy

Modify the configuration file in `src/adaptac/configs/tasks` directory, then launch the training script:

```bash
bash scripts/command_train.sh
```

### 3. Test Your Policy

Run evaluation:

```bash
bash scripts/command_eval.sh
```

## Quick Navigation

| Task | Script |
|------|--------|
| Dataset Generation | `scripts/generate_data.sh` |
| Training | `scripts/command_train.sh` |
| Evaluation | `scripts/command_eval.sh` |

## Tips

- Ensure all paths are correct in the scripts before running them
- Make sure the robot is reset to the initial position before starting evaluation
- Check that all hardware devices are properly connected and configured
- Verify USB port assignments match your hardware setup

## Common Issues

### Git Submodule Issues

**Submodules not initialized:**
- If dependencies are missing, make sure you cloned with `--recursive` flag
- Or run: `git submodule update --init --recursive`

**Submodule out of sync:**
- Update submodules to the latest commit: `git submodule update --remote`

### USB Port Configuration

We assign USB 0 for tactile sensor 1, USB 1 for tactile sensor 2, and USB 2 for the hand. Please make sure the USB port is correct.

**Check USB ports:**
```bash
ls /dev/ttyUSB*
```

**Grant permissions to USB ports:**
```bash
sudo chmod 777 /dev/ttyUSB*  # Replace * with the specific USB port number
```

## Contact

If you have any questions, please contact [Jinzhou Li](https://kingchou007.github.io/) or [Tianhao Wu](https://tianhaowuhz.github.io/).

## Citation

```bibtex
@article{li2025adaptive,
  title={Adaptive Visuo-Tactile Fusion with Predictive Force Attention for Dexterous Manipulation},
  author={Li, Jinzhou and Wu, Tianhao and Zhang, Jiyao and Chen, Zeyuan and Jin, Haotian and Wu, Mingdong and Shen, Yujun and Yang, Yaodong and Dong, Hao},
  journal={arXiv preprint arXiv:2505.13982},
  year={2025}
}
```

## Acknowledgments

We acknowledge the [RISE](https://github.com/rise-policy/RISE)/[FoAR](https://github.com/Alan-Heoooh/FoAR) authors for their open-source codebase. If you find our work beneficial, please consider citing us.

## License

This repository is released under the MIT License. See the [LICENSE](LICENSE) file for more details.
