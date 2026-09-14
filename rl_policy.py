"""Attention-based feature extractor for the RL policy.

The observation is a flat vector of shape ``window * n_feat + 1`` (a window of
technical features plus the current position). This extractor reshapes the
window into a sequence of ``window`` tokens and applies multi-head attention
pooling with a learnable query, then concatenates the current position and
projects to a fixed-size feature vector consumed by the PPO actor/critic heads.
"""

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AttentionExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        window: int = 20,
        n_feat: int = 10,
        d_model: int = 64,
        nhead: int = 4,
        features_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim)
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.window = window
        self.n_feat = n_feat

        self.proj = nn.Linear(n_feat, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, window, d_model))
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        self.out = nn.Linear(d_model + 1, features_dim)

        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch = observations.shape[0]
        seq = observations[:, : self.window * self.n_feat].view(batch, self.window, self.n_feat)
        pos = observations[:, -1:]

        x = self.proj(seq) + self.pos_emb
        query = self.query.expand(batch, -1, -1)
        x, _ = self.attn(query, x, x)
        x = x[:, 0]

        return self.out(torch.cat([x, pos], dim=-1))
