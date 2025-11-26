import numpy as np
import os
import pathlib

INTRINSICS_MATRIX_515 = {
    "fx": 898.22,
    "fy": 898.491,
    "cx": 653.722,
    "cy": 383.427,
}

# 12.15
EXTRINSIC_MATRIX = np.array(
    [
        [-0.89740872, 0.23855237, -0.37114734, 1.18276854],
        [0.43178102, 0.30196547, -0.84993059, 0.74104137],
        [-0.09067927, -0.9229895, -0.37398884, 0.54812949],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# flexiv home
FLEXIV_POSITIONS = {
    "home_js": np.array(
        [
            -0.20539183914661407,
            -0.6288893818855286,
            0.2788938581943512,
            1.8674005270004272,
            -0.2044251412153244,
            0.9303941130638123,
            0.27979201078414917,
        ]
    )
    * 57.3
    # 'home_js': np.array([-0.2133278250694275, -0.5433886051177979, 0.29640617966651917, 1.838704228401184, -0.1605091542005539, 0.8208202719688416, 0.2522362470626831]) * 57.3 # open light pos

}

# imagenet statistics for image normalization
IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD = np.array([0.229, 0.224, 0.225])

# tcp normalization and gripper width normalization
TRANS_MIN, TRANS_MAX = np.array([-0.25, -0.25, 0.8]), np.array(
    [0.5, 0.35, 1.4]
)  # camera
# TRANS_MIN, TRANS_MAX = np.array([0.3, -0.25, -0.2]), np.array([1.2, 0.3, 0.5])       # base

# workspace in camera coordinate
# CAM_WORKSPACE_MIN = np.array([-0.25, -0.22, 0.87])
CAM_WORKSPACE_MIN = np.array([-0.2, -0.2, 0.85])
CAM_WORKSPACE_MAX = np.array([0.45, 0.3, 1.35])

# workspace in base coordinate
BASE_WORKSPACE_MIN = np.array([0.5, -0.4, 0.0])
BASE_WORKSPACE_MAX = np.array([1, 0.1, 0.5])

# safe workspace in base coordinate
SAFE_EPS = 0.002
SAFE_WORKSPACE_MIN = np.array([0.2, -0.4, -0.05])
SAFE_WORKSPACE_MAX = np.array([0.8, 0.4, 0.4])

# hand joint limits
HAND_JOINT_LOWER_LIMIT = np.array(
    [
        -1.047,
        -0.314,
        -0.506,
        -0.366,
        -1.047,
        -0.314,
        -0.506,
        -0.366,
        -1.047,
        -0.314,
        -0.506,
        -0.366,
        -0.349,
        -0.47,
        -1.2,
        -1.34,
    ]
)

HAND_JOINT_UPPER_LIMIT = np.array(
    [
        1.047,
        2.23,
        1.885,
        2.042,
        1.047,
        2.23,
        1.885,
        2.042,
        1.047,
        2.23,
        1.885,
        2.042,
        2.094,
        2.443,
        1.9,
        1.88,
    ]
)

# Graph MAEGAT ckpt path - relative path
MAEGAT_CKPT = os.path.join(
    str(pathlib.Path(__file__).parent.parent.parent), "ckpt/4_maegat_v2_poseaug_10000.pt"
)
