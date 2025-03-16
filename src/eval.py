import os
import time
import json
import torch
import argparse
import numpy as np
import open3d as o3d
import MinkowskiEngine as ME
import cv2

from copy import deepcopy
from pynput import keyboard
from termcolor import cprint
from easydict import EasyDict as edict

from adaptac.policy.FFG_policy import FFG
from adaptac.dataset.utils.projector import Projector
from eval_agent import Agent
from utils.constants import *
from utils.training import set_seed, print_args
from utils.ensemble import EnsembleBuffer
from utils.transformation import xyz_rot_transform
from adaptac.model.tactile.tactile_processer import compute_resultant

STOP_EVAL = False
DELTA_Y = 0.0
DYNAMIC_INFERENCE = False

def _on_press(key):
    global STOP_EVAL
    global DELTA_Y
    try:
        if key.char == "q":
            cprint("Stop evaluation!", "red")
            STOP_EVAL = True
        elif key.char == "d":
            DELTA_Y = 0.05
    except AttributeError:
        # Handle special keys or other exceptions
        pass


keyboard_listener = keyboard.Listener(on_press=_on_press)
keyboard_listener.start()

default_args = edict(
    {
        "ckpt": None,
        "calib": "calib/",
        "num_action": 20,
        "num_inference_step": 20,
        "voxel_size": 0.005,
        "obs_feature_dim": 512,
        "hidden_dim": 512,
        "nheads": 8,
        "num_encoder_layers": 4,
        "num_decoder_layers": 1,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "max_steps": 300,
        "seed": 233,
        "vis": False,
        "discretize_rotation": True,
        "ensemble_mode": "new",
        "enable_state": False,
        "state_dim": 25,
        "state_feat_dims": (64, 64),
        "enable_state_mlp": False,
        "num_obs": 1,
        "repr_frame": "camera",  # camera or base
        "norm_trans": False,
        "use_color": False,
        "backbones": "resnet14",
        "enable_tactile": False,
        "tactile_frame": "camera",
        "tactile_rep_type": "3d_canonical_data",
        "tactile_backbone": "maegat'",
        "depth_mask": None,
        "diffusion_head": "unet",
        "fuse_type": "concat",
        "num_force_prediction": 0,
        "tactile_mask_ratio": 0,
        "tactile_weight_schedule": False,
        "force_prediction_type": "force",
        "predictor_feat_type": "all",
        "fuse_attention": [],
        "obs_net_force": True,
        "net_force_scale": 5.0,
    }
)


def get_states(arm_ee_pose, hand_joint_pos, max_xyz_range, min_xyz_range):
    arm_6d_ee_pose = xyz_rot_transform(
        arm_ee_pose, from_rep="quaternion", to_rep="rotation_6d"
    )
    states = np.concatenate([arm_6d_ee_pose, hand_joint_pos], axis=-1)
    states_normalized = normalize_state(states, max_xyz_range, min_xyz_range)
    return states_normalized


def read_xyz_range(frame, calib_path):
    xyz_range = np.load(
        os.path.join(calib_path, frame, "translation_range.npy"), allow_pickle=True
    ).item()
    return xyz_range["max_translation_range"], xyz_range["min_translation_range"]


def create_point_cloud(
    colors, depths, cam_intrinsics, voxel_size=0.005, repr_frame="camera"
):
    """
    color, depth => point cloud
    """
    h, w = depths.shape
    fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
    cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]

    colors = o3d.geometry.Image(colors.astype(np.uint8))
    depths = o3d.geometry.Image(depths.astype(np.float32))

    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        colors, depths, depth_scale=1.0, convert_rgb_to_intensity=False
    )
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
    cloud = cloud.voxel_down_sample(voxel_size)
    points = np.array(cloud.points).astype(np.float32)
    colors = np.array(cloud.colors).astype(np.float32)

    if repr_frame == "base":
        ws_min = BASE_WORKSPACE_MIN
        ws_max = BASE_WORKSPACE_MAX
        R = EXTRINSIC_MATRIX[:3, :3]
        t = EXTRINSIC_MATRIX[:3, 3]
        Rt = np.hstack((R, t.reshape(-1, 1)))
        point_homogeneous_base = np.hstack((points, np.ones((points.shape[0], 1))))
        point_homogeneous_camera = np.dot(Rt, point_homogeneous_base.T).T
        points_transformed = point_homogeneous_camera[:, :3]
        points = points_transformed
    else:
        ws_min = CAM_WORKSPACE_MIN
        ws_max = CAM_WORKSPACE_MAX

    x_mask = (points[:, 0] >= ws_min[0]) & (points[:, 0] <= ws_max[0])
    y_mask = (points[:, 1] >= ws_min[1]) & (points[:, 1] <= ws_max[1])
    z_mask = (points[:, 2] >= ws_min[2]) & (points[:, 2] <= ws_max[2])
    mask = x_mask & y_mask & z_mask
    points = points[mask]
    colors = colors[mask]
    # imagenet normalization
    colors = (colors - IMG_MEAN) / IMG_STD
    # final cloud
    cloud_final = np.concatenate([points, colors], axis=-1).astype(np.float32)
    return cloud_final


def create_batch(coords_batch, feats_batch):
    """
    coords, feats => batch coords, batch feats (batch size = 1)
    """
    coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)
    return coords_batch, feats_batch


def create_input(
    colors,
    depths,
    cam_intrinsics,
    voxel_size=0.005,
    repr_frame="camera",
    use_color=True,
):
    """
    colors, depths => batch coords, batch feats
    """
    num_obs = colors.shape[0]
    coords = []
    feats = []

    for frame_idx in range(num_obs):
        cloud = create_point_cloud(
            colors[frame_idx],
            depths[frame_idx],
            cam_intrinsics,
            voxel_size=voxel_size,
            repr_frame=repr_frame,
        )
        if not use_color:
            cloud = cloud[:, :3]
        coord = np.ascontiguousarray(cloud[:, :3] / voxel_size, dtype=np.int32)

        coords.append(coord)
        feats.append(cloud)

    coords_batch, feats_batch = create_batch(coords, feats)

    return coords_batch, feats_batch, cloud


def normalize_state(state, max_xyz_range, min_xyz_range):
    state[:, 9:] = (state[:, 9:] - HAND_JOINT_LOWER_LIMIT) / (
        HAND_JOINT_UPPER_LIMIT - HAND_JOINT_LOWER_LIMIT
    ) * 2 - 1
    return state


def unnormalize_action(action, norm_trans):
    if norm_trans:
        action[..., :3] = (action[..., :3] + 1) / 2.0 * (
            TRANS_MAX - TRANS_MIN
        ) + TRANS_MIN
    action[..., 9:] = (action[:, 9:] + 1) / 2.0 * (
        HAND_JOINT_UPPER_LIMIT - HAND_JOINT_LOWER_LIMIT
    ) + HAND_JOINT_LOWER_LIMIT
    return action


def discretize_translation(pos_begin, pos_end, step_size):
    vector = pos_end - pos_begin
    distance = np.linalg.norm(vector)
    n_step = int(distance // step_size) + 1
    pos_steps = []
    for i in range(n_step):
        pos_i = pos_begin * (n_step - 1 - i) / n_step + pos_end * (i + 1) / n_step
        pos_steps.append(pos_i)
    return pos_steps


def evaluate(args_override):
    global STOP_EVAL
    global DELTA_Y

    # load default arguments
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value

    # load args from training ckpt
    ckpt_dir = os.path.dirname(args.ckpt)
    training_args = json.load(open(os.path.join(ckpt_dir, "args.json"), "r"))
    cprint(f"Training args: {training_args}", "yellow")
    print_args(training_args)

    # override args
    for key, value in training_args.items():
        if key in args:
            args[key] = value

    # print args
    cprint(f"Eval args: {args}", "yellow")
    print_args(args)

    # set up device
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.depth_mask is not None:
        from adaptac.dataset.gen_data.process_obs import GetMaksedDepth

        depth_masker = GetMaksedDepth(mask_type=args.depth_mask, device=device)

    # policy
    print("Loading policy ...")
    policy = FFG(
        num_action=args.num_action,
        input_dim=6 if args.use_color else 3,
        obs_feature_dim=args.obs_feature_dim,
        action_dim=9 + 16,
        hidden_dim=args.hidden_dim,
        nheads=args.nheads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        enable_state=args.enable_state,
        state_dim=args.state_dim,
        num_obs=args.num_obs,
        backbone=args.backbones,
        enable_tactile=args.enable_tactile,
        tactile_backbone=args.tactile_backbone,
        fuse_type=args.fuse_type,
        diffusion_head=args.diffusion_head,
        num_force_prediction=args.num_force_prediction,
        force_prediction_type=args.force_prediction_type,
        predictor_feat_type=args.predictor_feat_type,
        args=args,
    ).to(device)
    n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))

    # load checkpoint
    assert args.ckpt is not None, "Please provide the checkpoint to evaluate."
    policy.load_state_dict(torch.load(args.ckpt, map_location=device), strict=True)
    print("Checkpoint {} loaded.".format(args.ckpt))

    # TODO: modify this part to fit your own flexiv robot
    agent = Agent(
        robot_sn="Rizon4-062521",
        camera_serial="f1230963",
        frame=args.repr_frame,
        obs_num=args.num_obs,
        enable_tactile=args.enable_tactile,
        tactile_rep_type=args.tactile_rep_type,
        tactile_frame=args.tactile_frame,
    )

    projector = Projector()
    ensemble_buffer = EnsembleBuffer(mode=args.ensemble_mode)

    # read xyz range
    max_xyz_range, min_xyz_range = read_xyz_range(args.repr_frame, args.calib)

    # VIDEO RECORDING
    recording_dir = os.path.join(
        os.path.dirname(args.ckpt),
        "videos",
        time.strftime("%m-%d-%H-%M-%S")
        + f"_step:{args.num_inference_step}_obs:{args.num_obs}",
    )
    os.makedirs(recording_dir, exist_ok=True)

    # pre-define
    prev_step_hand = None
    prev_step_tcp = None

    # check the hand
    agent.home_hand()
    register = input("\nPress a key to perform an action...")
    while register == "h":
        cprint("Reseting the Robot! If not, press h again...", "green")
        agent.home_hand()
        register = input("\nPress a key to start...")

    # fill obs buffer
    for _ in range(args.num_obs):
        colors, depths, arm_ee_pose, hand_joint_pos, tactile_data = (
            agent.get_observation()
        )

    if args.enable_tactile:
        register = input(
            "\nPress a key to start eval, if max tactile value is less than 2..."
        )

    with torch.inference_mode():
        policy.eval()
        for t in range(args.max_steps):
            if STOP_EVAL:
                break

            if t % args.num_inference_step == 0:
                # mask depth
                if args.depth_mask is not None:
                    arm_ee_pose2base = agent.robot.get_tcp_position()
                    arm_joint_pos = agent.robot.get_joint_position()
                    depths = depth_masker.get_masked_depths(
                        arm_ee_pose2base, hand_joint_pos, arm_joint_pos, depths
                    )

                coords, feats, cloud = create_input(
                    colors,
                    depths,
                    cam_intrinsics=agent.intrinsics,
                    voxel_size=args.voxel_size,
                    repr_frame=args.repr_frame,
                    use_color=args.use_color,
                )
                feats, coords = feats.to(device), coords.to(device)
                cloud_data = ME.SparseTensor(feats, coords)
                # get states
                states_data = get_states(
                    arm_ee_pose, hand_joint_pos, max_xyz_range, min_xyz_range
                )
                states_data = (
                    torch.tensor(states_data, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(device)
                )
                # predict
                if args.enable_tactile:
                    if args.obs_net_force:
                        obs_net_forces = []
                        for each_tact in tactile_data:
                            obs_net_force = compute_resultant(
                                each_tact[:, :6],
                                each_tact[:, 9:12],
                                "force",
                                scale = args.net_force_scale,
                            )
                            obs_net_forces.append(obs_net_force)
                        obs_net_forces = torch.tensor(obs_net_forces).to(device).float()
                        obs_net_forces = obs_net_forces.unsqueeze(
                            0
                        )
                    else:
                        obs_net_forces = None
                    tactile_data = torch.tensor(tactile_data).to(device).float()
                    tactile_data = tactile_data.unsqueeze(
                        0
                    )  # n_obs , tactile_dim => batch_size , n_obs , tactile_dim
                else:
                    tactile_data = None

                policy_return_dict = (
                    policy(
                        cloud_data,
                        actions=None,
                        batch_size=1,
                        tactiles=tactile_data,
                        states=states_data if args.enable_state else None,
                        mask_ratio=0,
                        obs_net_force=obs_net_forces,
                        return_prop=DYNAMIC_INFERENCE,
                    )
                )
                pred_raw_action = policy_return_dict['action_pred'].squeeze(0).cpu().numpy()

                if DYNAMIC_INFERENCE:
                    prop = policy_return_dict['prop']
                    args.num_inference_step = max(int((1 -prop.item()) * 4),1)
                    cprint(f"num_inference_step: {args.num_inference_step}", "red")

                # unnormalize predicted actions
                action = unnormalize_action(pred_raw_action, norm_trans=args.norm_trans)

                if args.vis:
                    import open3d as o3d

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
                    if args.use_color:
                        pcd.colors = o3d.utility.Vector3dVector(
                            cloud[:, 3:]  # * IMG_STD + IMG_MEAN
                        )
                    else:
                        pcd.colors = o3d.utility.Vector3dVector(
                            np.ones_like(cloud[:, :3]) * IMG_STD + IMG_MEAN
                        )

                    tcp_vis_list = []
                    for raw_tcp in action:
                        tcp_vis = o3d.geometry.TriangleMesh.create_sphere(
                            0.01
                        ).translate(raw_tcp[:3])
                        tcp_vis_list.append(tcp_vis)

                    tactile_vis_list = []
                    if args.enable_tactile:
                        tactile_points = tactile_data[0][0].cpu().numpy()[:, :3]
                        tactile_vis = o3d.geometry.PointCloud()
                        tactile_vis.points = o3d.utility.Vector3dVector(tactile_points)
                        tactile_vis.colors = o3d.utility.Vector3dVector(
                            np.ones_like(tactile_points) * [0, 0, 1]
                        )
                        tactile_vis_list.append(tactile_vis)

                    o3d.visualization.draw_geometries(
                        [pcd, *tcp_vis_list, *tactile_vis_list]
                    )

                # project action to base coordinate
                if args.repr_frame == "camera":
                    action_tcp = projector.project_tcp_to_base_coord(
                        action[..., :9], rotation_rep="rotation_6d"
                    )
                else:
                    action_tcp = action[..., :9]
                action_hand = action[..., 9:]
                action = np.concatenate([action_tcp, action_hand], axis=-1)

                ensemble_buffer.add_action(action, t)

            # get step action from ensemble buffer
            step_action = ensemble_buffer.get_action()
            if step_action is None:  # no action in the buffer => no movement.
                continue

            step_tcp = step_action[:9]
            step_hand = step_action[9:]

            quat_tcp = xyz_rot_transform(
                step_tcp, from_rep="rotation_6d", to_rep="quaternion"
            )

            # send tcp pose to robot
            DELTA_MAX_TRANSLATION = 0.1
            if prev_step_tcp is not None:
                step_tcp[:3] = np.clip(
                    step_tcp[:3],
                    prev_step_tcp - DELTA_MAX_TRANSLATION,
                    prev_step_tcp + DELTA_MAX_TRANSLATION,
                )
            prev_step_tcp = step_tcp[:3]

            num_interp = 4
            if prev_step_hand is not None:
                for interp_i in range(num_interp):
                    interpolated_hand_pos = (
                        prev_step_hand
                        + (step_hand - prev_step_hand) * interp_i / num_interp
                    )
                    agent.set_hand_joint_pos(interpolated_hand_pos, blocking=False)
                    rospy.sleep(0.05)

            trans_step = 0.05
            if prev_step_tcp is not None:
                trans_steps = discretize_translation(
                    prev_step_tcp, step_tcp[:3], trans_step
                )
                # cprint(f"interp tcp step len: {len(trans_steps)}", "red")
                for interp_step in trans_steps:
                    agent.set_tcp_pose(
                        np.concatenate([interp_step, step_tcp[3:]]),
                        rotation_rep="rotation_6d",
                        blocking=False,
                    )
            else:
                agent.set_tcp_pose(step_tcp, rotation_rep="rotation_6d", blocking=False)

            prev_step_tcp = step_tcp[:3]
            prev_step_hand = step_hand.copy()

            # update observation
            colors, depths, arm_ee_pose, hand_joint_pos, tactile_data = (
                agent.get_observation()
            )

            # save image
            img_path = os.path.join(recording_dir, f"{t:03d}.png")
            img = cv2.cvtColor(colors[-1], cv2.COLOR_BGR2RGB)
            cv2.imwrite(img_path, img)

    agent.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", action="store", type=str, help="checkpoint path", required=True
    )
    parser.add_argument(
        "--calib", action="store", type=str, help="calibration path", required=True
    )
    parser.add_argument(
        "--num_action",
        action="store",
        type=int,
        help="number of action steps",
        required=False,
        default=20,
    )
    parser.add_argument(
        "--num_inference_step",
        action="store",
        type=int,
        help="number of inference query steps",
        required=False,
        default=20,
    )
    parser.add_argument(
        "--voxel_size",
        action="store",
        type=float,
        help="voxel size",
        required=False,
        default=0.005,
    )
    parser.add_argument(
        "--obs_feature_dim",
        action="store",
        type=int,
        help="observation feature dimension",
        required=False,
        default=512,
    )
    parser.add_argument(
        "--hidden_dim",
        action="store",
        type=int,
        help="hidden dimension",
        required=False,
        default=512,
    )
    parser.add_argument(
        "--nheads",
        action="store",
        type=int,
        help="number of heads",
        required=False,
        default=8,
    )
    parser.add_argument(
        "--num_encoder_layers",
        action="store",
        type=int,
        help="number of encoder layers",
        required=False,
        default=4,
    )
    parser.add_argument(
        "--num_decoder_layers",
        action="store",
        type=int,
        help="number of decoder layers",
        required=False,
        default=1,
    )
    parser.add_argument(
        "--dim_feedforward",
        action="store",
        type=int,
        help="feedforward dimension",
        required=False,
        default=2048,
    )
    parser.add_argument(
        "--dropout",
        action="store",
        type=float,
        help="dropout ratio",
        required=False,
        default=0.1,
    )
    parser.add_argument(
        "--max_steps",
        action="store",
        type=int,
        help="max steps for evaluation",
        required=False,
        default=300,
    )
    parser.add_argument(
        "--seed", action="store", type=int, help="seed", required=False, default=233
    )
    parser.add_argument(
        "--vis", action="store_true", help="add visualization during evaluation"
    )
    parser.add_argument(
        "--discretize_rotation",
        action="store_true",
        help="whether to discretize rotation process.",
    )
    parser.add_argument(
        "--ensemble_mode",
        action="store",
        type=str,
        help="temporal ensemble mode",
        required=False,
        default="new",
    )
    parser.add_argument(
        "--enable_state", action="store_true", help="whether to use state information"
    )
    parser.add_argument(
        "--state_dim",
        action="store",
        type=int,
        help="state dimension",
        required=False,
        default=25,
    )
    parser.add_argument(
        "--state_feat_dims",
        action="store",
        type=int,
        nargs="+",
        help="state feature dimensions",
        required=False,
        default=(64, 64),
    )
    parser.add_argument(
        "--enable_state_mlp", action="store_true", help="whether to use state mlp"
    )
    parser.add_argument(
        "--num_obs",
        action="store",
        type=int,
        help="number of observations",
        required=False,
        default=1,
    )
    parser.add_argument(
        "--repr_frame",
        action="store",
        type=str,
        help="representation frame",
        required=False,
        default="camera",
    )
    parser.add_argument(
        "--norm_trans", action="store_true", help="whether to normalize translation"
    )
    parser.add_argument(
        "--use_color", action="store_true", help="whether to use color information"
    )
    parser.add_argument(
        "--backbones",
        action="store",
        type=str,
        help="backbone architecture",
        required=False,
        default="resnet14",
    )
    parser.add_argument(
        "--enable_tactile",
        action="store_true",
        help="whether to use tactile information",
    )
    parser.add_argument(
        "--depth_mask",
        action="store",
        type=str,
        help="depth mask type",
        required=False,
        default=None,
    )

    import rospy

    rospy.init_node("eval_agent")

    evaluate(vars(parser.parse_args()))
