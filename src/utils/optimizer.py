from typing import Dict, Tuple
import torch


def get_optimizer(
    policy,
    learning_rate: float,
    betas: Tuple[float, float],
    weight_decay: float,
    args,
    sparse_encoder_lr,
    sparse_encoder_transformer_lr,
) -> torch.optim.Optimizer:

    optim_groups = []

    # sparse_encoder
    optim_groups.append(
        {
            "params": policy.module.sparse_encoder.parameters(),
            "weight_decay": weight_decay,
            "lr": sparse_encoder_lr,
        }
    )

    # transformer
    optim_groups.append(
        {
            "params": policy.module.transformer.parameters(),
            "weight_decay": weight_decay,
            "lr": sparse_encoder_transformer_lr,
        }
    )
    # readout_embed
    optim_groups.append(
        {
            "params": policy.module.readout_embed.parameters(),
            "weight_decay": weight_decay,
            "lr": sparse_encoder_transformer_lr,
        }
    )

    if args.enable_tactile:
        # tactile_encoder
        optim_groups.append(
            {
                "params": policy.module.tactile_encoder.parameters(),
                "weight_decay": weight_decay,
                "lr": args.tactile_encoder_learning_rate,
            }
        )

        if args.tactile_backbone == "maegat":
            # dim_reducer
            optim_groups.append(
                {
                    "params": policy.module.dim_reducer.parameters(),
                    "weight_decay": weight_decay,
                    "lr": args.tactile_encoder_learning_rate,
                }
            )

    if args.enable_state:
        raise NotImplementedError

    if args.num_force_prediction > 0:
        # force_predictor
        optim_groups.append(
            {
                "params": policy.module.force_predictor.parameters(),
                "weight_decay": weight_decay,
                "lr": learning_rate,
            }
        )

    if args.fuse_type == "adaptive":
        # flat_embed
        optim_groups.append(
            {
                "params": policy.module.flat_embed.parameters(),
                "weight_decay": weight_decay,
                "lr": learning_rate,
            }
        )
    elif args.fuse_type == "implicit_adaptive":
        optim_groups.append(
            {
                "params": policy.module.force_guided_attn.parameters(),
                "weight_decay": weight_decay,
                "lr": learning_rate,
            }
        )

    if "global" in args.fuse_attention or "global_self" in args.fuse_attention:
        # global_embed
        optim_groups.append(
            {
                "params": policy.module.global_attention_fuser.parameters(),
                "weight_decay": weight_decay,
                "lr": learning_rate,
            }
        )

    # action decoder
    optim_groups.append(
        {
            "params": policy.module.action_decoder.parameters(),
            "weight_decay": weight_decay,
            "lr": learning_rate,
        }
    )

    print(len(optim_groups))

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer
