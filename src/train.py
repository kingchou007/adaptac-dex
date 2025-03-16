import os
import json
import wandb
import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist

from tqdm import tqdm
from easydict import EasyDict as edict
from diffusers.optimization import get_cosine_schedule_with_warmup

from adaptac.policy.FFG_policy import FFG
from adaptac.dataset.realdex import RealDexDataset, collate_fn
from utils.training import set_seed, plot_history, sync_loss, print_args
from utils.json_logger import JsonLogger
from utils.optimizer import get_optimizer
from adaptac.policy.utils import preprocess_data

import hydra
from omegaconf import DictConfig, OmegaConf

from utils.constants import MAEGAT_CKPT

class TactileWeightScheduler:
    def __init__(self, total_epoch):
        """
        Initialize the weight scheduler.

        Args:
        - start_epoch (int): The epoch when the tactile weight starts increasing (default: 1000).
        - end_epoch (int): The epoch when the tactile weight reaches 1 (default: 3000).
        """
        self.start_epoch = int(total_epoch * 0.5)
        self.end_epoch = int(total_epoch * 0.75)

    def get_weight(self, epoch):
        """
        Calculate the tactile weight based on the current epoch.

        Args:
        - epoch (int): The current training epoch.

        Returns:
        - float: The dynamically adjusted tactile weight.
        """
        if epoch < self.start_epoch:
            return 0.0  # Keep the weight at 0 before the start epoch
        elif epoch > self.end_epoch:
            return 1.0  # Keep the weight at 1 after the end epoch
        else:
            # Linearly interpolate weight from 0 to 1
            return (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)


@hydra.main(
    version_base=None,
    config_path="adaptac/configs",
    config_name="train_config",
)
def main(cfg: DictConfig):
    # Convert DictConfig to EasyDict for consistency
    args = edict(OmegaConf.to_container(cfg, resolve=True))

    log_path = os.path.join(args.output_dir, "logs.json.txt")

    # Prepare distributed training
    torch.multiprocessing.set_sharing_strategy("file_system")
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    RANK = int(os.environ["RANK"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    os.environ["NCCL_P2P_DISABLE"] = "1"
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=WORLD_SIZE, rank=RANK
    )

    # set up device
    set_seed(args.seed)
    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print arguments
    if RANK == 0:
        print_args(args)

    # Set up device
    set_seed(args.seed)
    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    if args.wandb and RANK == 0:
        wandb_run = wandb.init(
            project="dex-sense", config=args, name=args.exp_name, reinit=True
        )

    # Dataset & Dataloader
    if RANK == 0:
        print("Loading dataset ...")

    if "val" in os.listdir(args.data_path):
        has_val_dataset = True
    else:
        has_val_dataset = False

    dataset = RealDexDataset(
        path=args.data_path,
        split="train",
        num_obs=args.num_obs,
        num_action=args.num_action,
        voxel_size=args.voxel_size,
        aug=args.aug,
        aug_jitter=args.aug_jitter,
        with_cloud=False,
        norm_trans=args.norm_trans,
        use_color=args.use_color,
        aug_trans_min=args.aug_trans_min,
        aug_trans_max=args.aug_trans_max,
        aug_rot_min=args.aug_rot_min,
        aug_rot_max=args.aug_rot_max,
        keys=args.keys,
        gen_pc=args.gen_pc,
        action_type=args.action_type,
        tactile_frame=args.tactile_frame,
        num_force_prediction=args.num_force_prediction,
        args=args,
    )
    if RANK == 0:
        print(f"Dataset size: {len(dataset)}")

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size // WORLD_SIZE,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        sampler=sampler,
    )

    if has_val_dataset:
        val_dataset = RealDexDataset(
            path=args.data_path,
            split="val",
            num_obs=args.num_obs,
            num_action=args.num_action,
            voxel_size=args.voxel_size,
            aug=False,
            aug_jitter=False,
            with_cloud=False,
            norm_trans=args.norm_trans,
            use_color=args.use_color,
            aug_trans_min=args.aug_trans_min,
            aug_trans_max=args.aug_trans_max,
            aug_rot_min=args.aug_rot_min,
            aug_rot_max=args.aug_rot_max,
            keys=args.keys,
            gen_pc=False,
            action_type=args.action_type,
            tactile_frame=args.tactile_frame,
            num_force_prediction=args.num_force_prediction,
            args=args,
        )

        if RANK == 0:
            print(f"Validation Dataset size: {len(val_dataset)}")

        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=False
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size // WORLD_SIZE,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            sampler=val_sampler,
        )

    # Policy
    if RANK == 0:
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
            num_obs=args.num_obs,
            backbone=args.backbones,
            enable_tactile=args.enable_tactile,
            tactile_backbone=args.tactile_backbone,
            fuse_type=args.fuse_type,
            num_force_prediction=args.num_force_prediction,
            force_prediction_type=args.force_prediction_type,
            predictor_feat_type=args.predictor_feat_type,
            args=args,
            device=device,
        ).to(device)

    if RANK == 0:
        n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f"Number of parameters: {n_parameters / 1e6:.2f}M")
    policy = nn.parallel.DistributedDataParallel(
        policy,
        device_ids=[LOCAL_RANK],
        output_device=LOCAL_RANK,
        find_unused_parameters=True,
    )

    if args.pretrain_sparse_encoder_ckpt is not None:
        policy.module.sparse_encoder.load_state_dict(
            torch.load(args.pretrain_sparse_encoder_ckpt, map_location=device),
            strict=True,
        )
        sparse_encoder_lr = args.tactile_encoder_learning_rate
    else:
        sparse_encoder_lr = args.lr
    if args.pretrain_sparse_encoder_transformer_ckpt is not None:
        policy.module.transformer.load_state_dict(
            torch.load(
                args.pretrain_sparse_encoder_transformer_ckpt, map_location=device
            ),
            strict=True,
        )
        policy.module.readout_embed.load_state_dict(
            torch.load(
                args.pretrain_sparse_encoder_transformer_readout_embed_ckpt,
                map_location=device,
            ),
            strict=True,
        )
        sparse_encoder_transformer_lr = args.tactile_encoder_learning_rate
    else:
        sparse_encoder_transformer_lr = args.lr

    # Load checkpoint
    if args.resume_ckpt is not None:
        policy.module.load_state_dict(
            torch.load(args.resume_ckpt, map_location=device), strict=True
        )
        if RANK == 0:
            print("Checkpoint {} loaded.".format(args.resume_ckpt))

    # Checkpoint directory
    if RANK == 0 and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Save args
    if RANK == 0:
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(dict(args), f, indent=4)

    # Optimizer and LR Scheduler
    if RANK == 0:
        print("Loading optimizer and scheduler ...")

    optimizer = get_optimizer(
        policy,
        learning_rate=args.lr,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
        args=args,
        sparse_encoder_lr=sparse_encoder_lr,
        sparse_encoder_transformer_lr=sparse_encoder_transformer_lr,
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=1000,
        num_training_steps=len(dataloader) * args.num_epochs,
    )
    lr_scheduler.last_epoch = len(dataloader) * (args.resume_epoch + 1) - 1

    if args.tactile_weight_schedule:
        tactile_weight_scheduler = TactileWeightScheduler(args.num_epochs)

    # Training Loop
    train_history = []
    policy.train()
    for epoch in range(args.resume_epoch + 1, args.num_epochs):
        step_log = dict()

        if RANK == 0:
            print(f"Epoch {epoch}")
        if WORLD_SIZE > 1:
            sampler.set_epoch(epoch)
        optimizer.zero_grad()
        num_steps = len(dataloader)
        pbar = tqdm(dataloader) if RANK == 0 else dataloader
        avg_loss = 0
        if args.num_force_prediction > 0:
            avg_predict_loss = 0

        for data in pbar:
            cloud_data, action_data, tactile_tensor, net_force, obs_net_force = (
                preprocess_data(data, args, device)
            )

            if args.tactile_weight_schedule:
                tactile_weight = tactile_weight_scheduler.get_weight(epoch)
            else:
                tactile_weight = 1.0

            policy_return_dict = policy(
                cloud_data,
                action_data,
                batch_size=action_data.shape[0],
                tactiles=tactile_tensor if args.enable_tactile else None,
                net_force=net_force,
                tactile_weight=tactile_weight,
                obs_net_force=obs_net_force,
            )
            loss = policy_return_dict["loss"]

            # check nan
            has_nan = torch.isnan(loss)
            has_nan_tensor = torch.tensor(float(has_nan), dtype=torch.float32, device='cuda')
            dist.all_reduce(has_nan_tensor, op=dist.ReduceOp.MAX)
            if has_nan_tensor.item() == 1.0:
                print(f"Rank {dist.get_rank()}: get NaN, skip update")
                continue

            if args.num_force_prediction > 0:
                predict_loss = policy_return_dict["predict_loss"]
                loss += args.force_prediction_alpha * predict_loss

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_value_(policy.parameters(), clip_value=5.0)

            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            avg_loss += loss.item()
            if args.num_force_prediction > 0:
                avg_predict_loss += predict_loss.item()

        avg_loss /= num_steps
        if args.num_force_prediction > 0:
            avg_predict_loss /= num_steps

        if RANK == 0:
            step_log["epoch"] = epoch
            step_log["train_loss"] = avg_loss
            if args.num_force_prediction > 0:
                step_log["predict_loss"] = avg_predict_loss

        if RANK == 0:
            with JsonLogger(log_path) as json_logger:
                json_logger.log(step_log)

        # save logs
        if args.wandb and RANK == 0:
            wandb_run.log(step_log, epoch)

        if WORLD_SIZE > 1:
            avg_loss = sync_loss(avg_loss, device)
        train_history.append(avg_loss)

        if RANK == 0:
            print("Train loss: {:.6f}".format(avg_loss))
            if (epoch + 1) % args.save_epochs == 0:
                torch.save(
                    policy.module.state_dict(),
                    os.path.join(
                        args.output_dir,
                        "policy_epoch_{}_seed_{}.ckpt".format(epoch + 1, args.seed),
                    ),
                )
                if args.pretrain_sparse_encoder_ckpt is None:
                    torch.save(
                        policy.module.sparse_encoder.state_dict(),
                        os.path.join(
                            args.output_dir,
                            "sparse_encoder_epoch_{}_seed_{}.ckpt".format(
                                epoch + 1, args.seed
                            ),
                        ),
                    )
                if args.pretrain_sparse_encoder_transformer_ckpt is None:
                    torch.save(
                        policy.module.transformer.state_dict(),
                        os.path.join(
                            args.output_dir,
                            "sparse_encoder_transformer_epoch_{}_seed_{}.ckpt".format(
                                epoch + 1, args.seed
                            ),
                        ),
                    )
                if args.pretrain_sparse_encoder_transformer_readout_embed_ckpt is None:
                    torch.save(
                        policy.module.readout_embed.state_dict(),
                        os.path.join(
                            args.output_dir,
                            "sparse_encoder_transformer_readout_embed_epoch_{}_seed_{}.ckpt".format(
                                epoch + 1, args.seed
                            ),
                        ),
                    )
                plot_history(train_history, epoch, args.output_dir, args.seed)

    if RANK == 0:
        torch.save(
            policy.module.state_dict(),
            os.path.join(args.output_dir, "policy_last.ckpt"),
        )


if __name__ == "__main__":
    main()
