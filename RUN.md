# Training Tutorial

### 1. Generate Training Dataset
Before running the script, please configure these key parameters
```bash
data_dir          # Directory for raw dataset
output_dir        # Directory for processed dataset
frame             # Transform data in 'camera' or 'base' frame
tactile_rep_type  # Type of tactile representation

bash scripts/generate_dataset.sh
```

### 2. Train Your Policy
mofiy the configuration file in `src/adaptac/configs/tasks` directory.

Launch training script:
```bash
bash scripts/command_train.sh
```

### 3. Test Your Policy
Run evaluation:

```bash
bash scripts/command_eval.sh

# press h to reset the robot to the initial position
h
```

### 4. Generate Test Results Video
Configure video generation parameters:

```bash
input_dir   # Directory containing test results
output_dir  # Directory for saving generated videos

bash scripts/generate_video.sh
```

## Quick Navigation
Dataset Generation: scripts/generate_dataset.sh
Training: scripts/command_train.sh
Testing: scripts/command_test.sh


## Tips
- Ensure the path is correct in the scripts before running them.
- Make sure the robot is reset to the initial position before starting.


## Common Issues
We assign the USB 0 for tactile sensor 1, USB 1 for tactile sensor 2, and USB 2 for the hand. Please make sure the USB port is correct. Once you connect the USB port, you can check the USB port by running the following command:
```bash
ls /dev/ttyUSB*
```

Also, give the permission to the USB port by running the following command:
```bash
sudo chmod 777 /dev/ttyUSB*(USB port)
```