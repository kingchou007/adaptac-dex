import torch
import torch.nn as nn


class ForceGuidedCrossAttention(nn.Module):
    """Force-guided cross-attention for fusing visual and tactile features."""

    def __init__(
        self,
        visual_dim,
        tactile_dim,
        force_dim,
        hidden_dim=256,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.tactile_proj = nn.Linear(tactile_dim, hidden_dim)
        self.force_proj = nn.Linear(force_dim, hidden_dim)

        self.cross_attn = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )

    def _expand_force_for_observations(self, force, num_obs):
        """Expand force from [B, hidden_dim] to [B*n_obs, 1, hidden_dim]."""
        return force.unsqueeze(1).repeat(1, num_obs, 1).view(-1, 1, self.hidden_dim)

    def forward(self, visual_feat, tactile_feat, predicted_force, num_obs):
        """
        Args:
            visual_feat: [B*n_obs, visual_dim]
            tactile_feat: [B*n_obs, tactile_dim]
            predicted_force: [B, force_dim]
            num_obs: number of observations per batch

        Returns:
            dict with key "attended_feat": [B*n_obs, hidden_dim]
        """
        visual = self.visual_proj(visual_feat)
        tactile = self.tactile_proj(tactile_feat)
        force = self.force_proj(predicted_force)

        query = self._expand_force_for_observations(force, num_obs)
        memory = torch.stack([visual, tactile], dim=1)

        attended_feat = self.cross_attn(tgt=query, memory=memory)
        return {"attended_feat": attended_feat.squeeze(1)}


class SelfAttentionFusion(nn.Module):
    """Self-attention fusion for combining visual and tactile features."""

    def __init__(
        self,
        visual_dim,
        tactile_dim,
        hidden_dim=256,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.tactile_proj = nn.Linear(tactile_dim, hidden_dim)

        self.self_attn_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, visual, tactile):
        """
        Args:
            visual: [B, visual_dim]
            tactile: [B, tactile_dim]

        Returns:
            [B, 2, hidden_dim] - fused features
        """
        visual_feat = self.visual_proj(visual)
        tactile_feat = self.tactile_proj(tactile)

        stacked_features = torch.stack([visual_feat, tactile_feat], dim=1)
        return self.self_attn_layer(stacked_features)


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion using tactile as query and visual as key/value."""

    def __init__(
        self, embed_dim=512, num_heads=8, ff_dim=1024, dropout=0.1, activation="relu"
    ):
        super().__init__()
        self.cross_attn_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

    def forward(self, visual_feat, tactile_feat):
        """
        Args:
            visual_feat: [B, num_obs, embed_dim]
            tactile_feat: [B, num_obs, embed_dim]

        Returns:
            [B, num_obs, embed_dim] - fused features
        """
        return self.cross_attn_layer(tgt=tactile_feat, memory=visual_feat)