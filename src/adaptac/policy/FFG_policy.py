import torch
import torch.nn as nn

from utils.constants import MAEGAT_CKPT
from adaptac.model.tokenizer import Sparse3DEncoder
from adaptac.model.transformer import Transformer
from adaptac.model.unet_diffusion import DiffusionUNetPolicy
from adaptac.model.transformer_diffusion import DiffusionTransformer
from adaptac.model.tactile.tactile_encoder import MAEGAT, DimReducer
from adaptac.model.fuse_network import CrossAttentionFusion, SelfAttentionFusion, ForceGuidedCrossAttention


class FFG(nn.Module):
    """Force Guided Fusion Policy."""

    def __init__(
        self,
        num_action=20,
        input_dim=6,
        obs_feature_dim=512,
        action_dim=25,
        hidden_dim=512,
        nheads=8,
        num_encoder_layers=4,
        num_decoder_layers=1,
        dim_feedforward=2048,
        dropout=0.1,
        num_obs=1,
        backbone="resnet14",
        enable_tactile=False,
        tactile_backbone="maegat",
        fuse_type="concat",
        num_force_prediction=0,
        force_prediction_type="force",
        predictor_feat_type="all",
        args=None,
        device=None,
    ):
        super().__init__()
        if args is None:
            raise ValueError("args is required to configure FFG.")

        self.enable_tactile = enable_tactile
        self.tactile_backbone = tactile_backbone
        self.num_obs = num_obs
        self.fuse_type = fuse_type
        self.fuse_attention = args.fuse_attention
        self.num_force_prediction = num_force_prediction
        self.predictor_feat_type = predictor_feat_type
        self.force_prediction_subtype = args.force_prediction_subtype
        self.obs_net_force = args.obs_net_force
        self.obs_force_guide_only = args.obs_force_guide_only
        self.net_force_scale = args.net_force_scale
        self.use_global_attention = "global" in self.fuse_attention
        self.use_global_self_attention = (
            not self.use_global_attention and "global_self" in self.fuse_attention
        )

        self.sparse_encoder = Sparse3DEncoder(input_dim, obs_feature_dim, backbone=backbone)
        self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.readout_embed = nn.Embedding(1, hidden_dim)

        action_decoder_obs_dim = obs_feature_dim
        if self.enable_tactile:
            self.tactile_encoder = MAEGAT(pretrain_ckpt_path=MAEGAT_CKPT, device=device)
            self.dim_reducer = DimReducer(output_dim=obs_feature_dim)
            action_decoder_obs_dim += obs_feature_dim

        self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, action_decoder_obs_dim)

        if self.num_force_prediction > 0:
            pred_force_dim = action_decoder_obs_dim if self.predictor_feat_type == "all" else obs_feature_dim
            self.force_predictor = DiffusionTransformer(
                action_dim=3,
                horizon=self.num_force_prediction,
                num_obs=num_obs,
                obs_dim=pred_force_dim,
                prediction_type="sample",
                clip_sample=(self.net_force_scale >= 5),
            )

            if self.obs_force_guide_only:
                force_dim = num_obs * 3
            elif self.obs_net_force:
                force_dim = (self.num_force_prediction + num_obs) * 3
            else:
                force_dim = self.num_force_prediction * 3

            self.force_guided_attn = ForceGuidedCrossAttention(
                visual_dim=obs_feature_dim,
                tactile_dim=obs_feature_dim,
                force_dim=force_dim,
                hidden_dim=hidden_dim,
            )

        if self.use_global_attention:
            self.global_attention_fuser = CrossAttentionFusion(
                embed_dim=obs_feature_dim,
                num_heads=nheads,
                ff_dim=dim_feedforward,
                dropout=dropout,
                activation="relu",
            )
        elif self.use_global_self_attention:
            self.global_attention_fuser = SelfAttentionFusion(
                visual_dim=obs_feature_dim * num_obs,
                tactile_dim=obs_feature_dim * num_obs,
                hidden_dim=hidden_dim,
            )

    def get_prediction_features(self, readout, tactile_feat, batch_size):
        """Get features for force prediction."""
        if self.predictor_feat_type == "visual":
            return readout
        elif tactile_feat is None:
            raise ValueError("Tactile features required for force prediction.")
        elif self.use_global_attention:
            return self.global_attention_fuser(
                readout.view(batch_size, self.num_obs, -1),
                tactile_feat.view(batch_size, self.num_obs, -1),
            )
        elif self.use_global_self_attention:
            return self.global_attention_fuser(
                readout.view(batch_size, -1),
                tactile_feat.view(batch_size, -1),
            )
        else:
            return torch.cat([readout, tactile_feat], dim=1)

    def predict_future_force(self, readout, tactile_feat, net_force, obs_net_force, actions, batch_size):
        """Predict future force from visual and tactile features."""
        pred_feat = self.get_prediction_features(readout, tactile_feat, batch_size)
        pred_feat = pred_feat.view(batch_size, self.num_obs, -1)
        gt_force = net_force if self.force_prediction_subtype == "future" else obs_net_force
        
        if actions is not None:
            predict_loss, predict_force = self.force_predictor.compute_loss(
                pred_feat, gt_force, return_pred=True
            )
            return predict_loss, predict_force
        else:
            predict_force = self.force_predictor.predict_action(pred_feat)
            zero_loss = readout.new_tensor(0.0)
            return zero_loss, predict_force

    def apply_force_guided_attention(self, readout, tactile_feat, obs_net_force, predict_force, batch_size):
        """Apply force-guided attention to fuse visual and tactile features."""
        # Get guide force
        if self.obs_force_guide_only:
            guide_force = obs_net_force
        elif self.obs_net_force:
            if self.num_force_prediction > 0:
                guide_force = torch.cat([obs_net_force, predict_force], dim=1)
            else:
                guide_force = obs_net_force
        elif self.num_force_prediction > 0:
            guide_force = predict_force

        visual_feat = readout.view(batch_size * self.num_obs, -1)
        tactile_feat_reshaped = tactile_feat.view(batch_size * self.num_obs, -1)
        guide_force_reshaped = guide_force.view(batch_size, -1)
        
        result = self.force_guided_attn(visual_feat, tactile_feat_reshaped, guide_force_reshaped)
        return result["attended_feat"]

    def forward(
        self,
        cloud,
        actions=None,
        batch_size=24,
        tactiles=None,
        net_force=None,
        return_pred=False,
        tactile_weight=1.0,
        obs_net_force=None
    ):
        # Encode visual input
        src, pos, src_padding_mask = self.sparse_encoder(cloud, batch_size=batch_size * self.num_obs)
        readout = self.transformer(src, src_padding_mask, self.readout_embed.weight, pos)[-1][:, 0]

        # Encode tactile input
        tactile_feat = None
        if tactiles is not None:
            tactile_feat = self.tactile_encoder(tactiles.reshape(-1, *tactiles.shape[2:]))
            tactile_feat = self.dim_reducer(torch.flatten(tactile_feat, start_dim=1))

        # Predict force if needed
        predict_loss = readout.new_tensor(0.0)
        predict_force = None
        
        if self.num_force_prediction > 0:
            predict_loss, predict_force = self.predict_future_force(
                readout, tactile_feat, net_force, obs_net_force, actions, batch_size
            )

        # Fuse features with force-guided attention
        final_feat = readout
        if tactile_feat is not None and hasattr(self, 'force_guided_attn'):
            final_feat = self.apply_force_guided_attention(
                readout, tactile_feat, obs_net_force, predict_force, batch_size
            )

        final_feat = final_feat.view(batch_size, self.num_obs, -1)

        # Compute output
        return_dict = {}
        if actions is not None:
            loss = self.action_decoder.compute_loss(final_feat, actions)
            return_dict["loss"] = loss
            return_dict["predict_loss"] = predict_loss * tactile_weight
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(final_feat)
            return_dict["action_pred"] = action_pred
            if return_pred:
                return_dict["predict_force"] = predict_force

        return return_dict
