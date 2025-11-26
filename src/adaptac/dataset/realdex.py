import os
import json
import torch
import numpy as np
import open3d as o3d
import MinkowskiEngine as ME
import torchvision.transforms as T
import collections.abc as container_abcs

from PIL import Image
from tqdm import tqdm
from termcolor import cprint
from torch.utils.data import Dataset

# from replay_buffer import ReplayBuffer
import zarr
from adaptac.dataset.utils.constants import *
from adaptac.dataset.utils.projector import Projector
from adaptac.dataset.utils.rgb_augmentation import GaussianBlur
from adaptac.dataset.gen_data.convert_zarr import get_cropped_point_cloud

from utils.transformation import (
    rot_trans_mat,
    apply_mat_to_pose,
    apply_mat_to_pcd,
    xyz_rot_transform,
)
from utils.normalization import normalize_arm_hand, normalize_tactile
from adaptac.model.tactile.tactile_processer import compute_resultant


class RealDexDataset(Dataset):
    """
    Real-world Dataset.
    """

    def __init__(
        self,
        path,
        split="train",
        num_obs=1,
        num_action=20,
        voxel_size=0.005,
        cam_ids=["515"],
        aug=False,
        aug_trans_min=[-0.2, -0.2, -0.2],
        aug_trans_max=[0.2, 0.2, 0.2],
        aug_rot_min=[-30, -30, -30],
        aug_rot_max=[30, 30, 30],
        aug_jitter=False,
        aug_jitter_params=[0.4, 0.4, 0.2, 0.1],
        aug_jitter_prob=0.2,
        with_cloud=False,
        norm_trans=False,
        use_color=True,
        keys=["action", "tactile", "state", "point_cloud", "input_index"],
        gen_pc=False,
        frame="camera",
        crop_frame=None,
        action_type="abs",
        tactile_frame="camera",
        num_force_prediction=0,
        args=None,
    ):
        assert split in ["train", "val", "all"]

        self.path = path
        self.split = split
        self.data_path = os.path.join(path, split)
        self.calib_path = "data/pick_basketball_v1/calib"
        self.num_obs = num_obs
        self.num_action = num_action
        self.voxel_size = voxel_size
        self.aug = aug
        self.aug_trans_min = np.array(aug_trans_min)
        self.aug_trans_max = np.array(aug_trans_max)
        self.aug_rot_min = np.array(aug_rot_min)
        self.aug_rot_max = np.array(aug_rot_max)
        self.aug_jitter = aug_jitter
        self.aug_jitter_params = np.array(aug_jitter_params)
        self.aug_jitter_prob = aug_jitter_prob
        self.with_cloud = with_cloud
        self.norm_trans = norm_trans
        self.use_color = use_color
        self.keys = keys  # ['action', 'state', 'point_cloud', 'input_index']
        self.gen_pc = gen_pc
        self.frame = frame
        self.crop_frame = crop_frame
        self.action_type = action_type
        self.tactile_frame = tactile_frame
        self.num_force_prediction = num_force_prediction

        # Handle args being None
        if args is None:
            self.obs_net_force = False
            self.with_rgb = False
            self.net_force_scale = 1.0
        else:
            self.obs_net_force = args.obs_net_force
            self.with_rgb = args.with_rgb
            self.net_force_scale = args.net_force_scale

        zarr_root = zarr.open(self.data_path)
        for key in self.keys:
            if gen_pc and key == "point_cloud":
                continue
            setattr(self, "data_" + key, zarr_root["data"][key][:])

        self.episode_ends = zarr_root["meta/episode_ends"][:]
        self.num_demos = len(self.episode_ends)
        self.data_paths = []
        self.cam_ids = []
        self.calib_timestamp = []
        self.obs_frame_ids = []
        self.action_frame_ids = []
        self.tactile_frame_ids = []
        if self.num_force_prediction > 0:
            self.net_force_frame_ids = []
        self.projectors = {}

        start_frame_idx = 0
        for i in range(self.num_demos):
            for cam_id in cam_ids:
                frame_ids = list(range(start_frame_idx, self.episode_ends[i]))
                start_frame_idx = self.episode_ends[i]
                # get samples according to num_obs and num_action
                obs_frame_ids_list = []
                action_frame_ids_list = []
                padding_mask_list = []
                tactile_data_ids_list = []
                if self.num_force_prediction > 0:
                    net_force_frame_ids_list = []

                for cur_idx in range(len(frame_ids) - 1):
                    obs_pad_before = max(0, num_obs - cur_idx - 1)
                    action_pad_after = max(
                        0, num_action - (len(frame_ids) - 1 - cur_idx)
                    )
                    frame_begin = max(0, cur_idx - num_obs + 1)
                    frame_end = min(len(frame_ids), cur_idx + num_action + 1)
                    obs_frame_ids = (
                        frame_ids[:1] * obs_pad_before
                        + frame_ids[frame_begin : cur_idx + 1]
                    )
                    action_frame_ids = (
                        frame_ids[cur_idx + 1 : frame_end]
                        + frame_ids[-1:] * action_pad_after
                    )
                    obs_frame_ids_list.append(obs_frame_ids)
                    action_frame_ids_list.append(action_frame_ids)

                    tactile_frame_ids = obs_frame_ids
                    tactile_data_ids_list.append(tactile_frame_ids)

                    if self.num_force_prediction > 0:
                        net_force_pad_after = max(
                            0,
                            self.num_force_prediction - (len(frame_ids) - 1 - cur_idx),
                        )
                        net_frame_end = min(
                            len(frame_ids), cur_idx + self.num_force_prediction + 1
                        )
                        net_force_frame_ids = (
                            frame_ids[cur_idx + 1 : net_frame_end]
                            + frame_ids[-1:] * net_force_pad_after
                        )
                        net_force_frame_ids_list.append(net_force_frame_ids)

                self.cam_ids += [cam_id] * len(obs_frame_ids_list)
                self.obs_frame_ids += obs_frame_ids_list
                self.action_frame_ids += action_frame_ids_list
                self.tactile_frame_ids += tactile_data_ids_list
                if self.num_force_prediction > 0:
                    self.net_force_frame_ids += net_force_frame_ids_list

        # create color jitter
        if self.split == "train" and self.aug_jitter:
            s = 1.0
            color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.5 * s)
            self.rgb_transforms = T.Compose(
                [
                    T.RandomApply([color_jitter], p=0.8),
                    T.RandomGrayscale(p=0.2),
                ]
            )


    def __len__(self):
        return len(self.obs_frame_ids)

    def _augmentation(self, clouds, tcps, tactiles=None, net_force_tactiles=None):
        # translation and rotation augmentation
        translation_offsets = (
            np.random.rand(3) * (self.aug_trans_max - self.aug_trans_min)
            + self.aug_trans_min
        )
        rotation_angles = (
            np.random.rand(3) * (self.aug_rot_max - self.aug_rot_min) + self.aug_rot_min
        )
        rotation_angles = (
            rotation_angles / 180 * np.pi
        )  # tranform from degree to radius
        aug_mat = rot_trans_mat(translation_offsets, rotation_angles)
        for cloud in clouds:
            cloud = apply_mat_to_pcd(cloud, aug_mat)
        tcps = apply_mat_to_pose(tcps, aug_mat, rotation_rep="quaternion")

        if tactiles is not None and self.tactile_frame != "hand":
            for tactile in tactiles:
                tactile_origin_pose = tactile[:, :6].copy()
                tactile_origin_pose[:, 3:6] *= np.pi
                tactile_origin_pose = apply_mat_to_pose(
                    tactile_origin_pose, aug_mat, "euler_angles", "XYZ"
                )
                tactile_origin_pose[:, 3:6] /= np.pi
                tactile[:, :6] = tactile_origin_pose

        if net_force_tactiles is not None and self.tactile_frame != "hand":
            for net_force_tactile in net_force_tactiles:
                net_force_tactile_origin_pose = net_force_tactile[:, :6].copy()
                net_force_tactile_origin_pose[:, 3:6] *= np.pi
                net_force_tactile_origin_pose = apply_mat_to_pose(
                    net_force_tactile_origin_pose, aug_mat, "euler_angles", "XYZ"
                )
                net_force_tactile_origin_pose[:, 3:6] /= np.pi
                net_force_tactile[:, :6] = net_force_tactile_origin_pose

        return clouds, tcps, tactiles, net_force_tactiles

    def __getitem__(self, index):
        cam_id = self.cam_ids[index]
        obs_frame_ids = self.obs_frame_ids[index]
        action_frame_ids = self.action_frame_ids[index]
        tactile_frame_ids = (
            self.tactile_frame_ids[index] if "tactile" in self.keys else None
        )
        if self.num_force_prediction > 0:
            net_force_frame_ids = self.net_force_frame_ids[index]

        # load camera projector by calib timestamp
        timestamp = "515"
        if timestamp not in self.projectors:
            self.projectors[timestamp] = Projector()
        projector = self.projectors[timestamp]

        # state
        state_tcps = []
        state_hands = []
        for frame_id in obs_frame_ids:
            state_data = self.data_state[frame_id].copy()  # n_obs, 25
            tcp = state_data[:7].astype(np.float32)
            projected_tcp = tcp
            hand_joint = state_data[7:].astype(np.float32)
            state_tcps.append(projected_tcp)
            state_hands.append(hand_joint)
        state_tcps = np.stack(state_tcps)
        state_hands = np.stack(state_hands)

        # actions
        action_tcps = []
        action_hands = []
        for frame_id in action_frame_ids:
            action_data = self.data_action[frame_id].copy()  # n_action, 25
            tcp = action_data[:7].astype(np.float32)
            projected_tcp = tcp
            hand_joint = action_data[7:].astype(np.float32)
            action_tcps.append(projected_tcp)
            action_hands.append(hand_joint)
        action_tcps = np.stack(action_tcps)
        action_hands = np.stack(action_hands)

        # get point cloud
        if self.gen_pc:
            clouds = []
            # load colors and depths
            for frame_id in obs_frame_ids:
                color = self.data_img[frame_id].copy()
                if self.split == "train" and self.aug_jitter:
                    color = Image.fromarray(color)
                    color = self.rgb_transforms(color)
                    color = np.array(color)
                depth = self.data_depth[frame_id].copy()

                clouds.append(
                    get_cropped_point_cloud(
                        color, depth, self.voxel_size, self.frame, self.crop_frame
                    )
                )
        else:
            clouds = []
            for frame_id in obs_frame_ids:
                clouds.append(
                    self.data_point_cloud[frame_id][
                        : self.data_input_index[frame_id]
                    ].copy()
                )

        # get tactile data
        tactiles = []
        if "tactile" in self.keys:
            for frame_id in tactile_frame_ids:
                # assume it already has 3d canonical data
                tactiles.append(self.data_tactile[frame_id].copy())
        else:
            tactiles = None

        if "tactile" in self.keys and self.num_force_prediction > 0:
            net_force_tactiles = []
            for net_force_frame_id in net_force_frame_ids:
                net_force_tactiles.append(self.data_tactile[net_force_frame_id].copy())
        else:
            net_force_tactiles = None

        # augmentation
        if self.aug and self.split == "train":
            clouds, action_tcps, tactiles, net_force_tactiles = self._augmentation(
                clouds, action_tcps, tactiles, net_force_tactiles
            )

        if self.action_type == "abs":
            # rotation transformation (to 6d)
            action_tcps = xyz_rot_transform(
                action_tcps, from_rep="quaternion", to_rep="rotation_6d"
            )
            actions = np.concatenate((action_tcps, action_hands), axis=-1)

            state_tcps = xyz_rot_transform(
                state_tcps, from_rep="quaternion", to_rep="rotation_6d"
            )
            states = np.concatenate((state_tcps, state_hands), axis=-1)

        # normalization
        actions_normalized = normalize_arm_hand(
            actions.copy(), norm_trans=self.norm_trans
        )
        states_normalized = normalize_arm_hand(
            states.copy(), norm_trans=self.norm_trans
        )

        # make voxel input
        input_coords_list = []
        input_feats_list = []
        for cloud in clouds:
            coords = np.ascontiguousarray(
                cloud[:, :3] / self.voxel_size, dtype=np.int32
            )
            input_coords_list.append(coords)
            if self.use_color:
                feats = cloud.astype(np.float32)
            else:
                feats = cloud.astype(np.float32)[:, :3]
            input_feats_list.append(feats)

        # convert to torch
        actions = torch.from_numpy(actions).float()
        actions_normalized = torch.from_numpy(actions_normalized).float()
        states_normalized = torch.from_numpy(states_normalized).float()

        ret_dict = {
            "input_coords_list": input_coords_list,
            "input_feats_list": input_feats_list,
            "action": actions,
            "action_normalized": actions_normalized,
            "state_normalized": states_normalized,
        }

        if "tactile" in self.keys:
            tactile = np.stack(tactiles)
            tactile = torch.from_numpy(tactile).float()
            ret_dict["tactile"] = tactile

        if self.num_force_prediction > 0:
            net_forces = []
            for net_force_idx in range(len(net_force_frame_ids)):
                net_force = compute_resultant(
                    net_force_tactiles[net_force_idx][:, :6],
                    net_force_tactiles[net_force_idx][:, 9:12],
                    "force",
                    scale = self.net_force_scale,
                )
                net_forces.append(net_force)
            net_force = np.stack(net_forces)
            net_force = torch.from_numpy(net_force).float()
            ret_dict["net_force"] = net_force

        if self.obs_net_force:
            obs_net_forces = []
            for obs_net_force_idx in range(len(obs_frame_ids)):
                obs_net_force = compute_resultant(
                    tactiles[obs_net_force_idx][:, :6],
                    tactiles[obs_net_force_idx][:, 9:12],
                    "force",
                    scale = self.net_force_scale,
                )
                obs_net_forces.append(obs_net_force)
            obs_net_force = np.stack(obs_net_forces)
            obs_net_force = torch.from_numpy(obs_net_force).float()
            ret_dict["obs_net_force"] = obs_net_force

        if (
            self.with_cloud
        ):  # warning: this may significantly slow down the training process.
            ret_dict["clouds_list"] = clouds

        if self.with_rgb:
            ret_dict["rgb_list"] = [self.data_img[frame_id].copy() for frame_id in obs_frame_ids]
        return ret_dict


def collate_fn(batch):
    if type(batch[0]).__module__ == "numpy":
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif torch.is_tensor(batch[0]):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        ret_dict = {}
        for key in batch[0]:
            if key in TO_TENSOR_KEYS:
                ret_dict[key] = collate_fn([d[key] for d in batch])
            else:
                ret_dict[key] = [d[key] for d in batch]
        coords_batch = ret_dict["input_coords_list"]
        feats_batch = ret_dict["input_feats_list"]
        coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)
        ret_dict["input_coords_list"] = coords_batch
        ret_dict["input_feats_list"] = feats_batch
        return ret_dict
    elif isinstance(batch[0], container_abcs.Sequence):
        return [sample for b in batch for sample in b]

    raise TypeError(
        "batch must contain tensors, dicts or lists; found {}".format(type(batch[0]))
    )
