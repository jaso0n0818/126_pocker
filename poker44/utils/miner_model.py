"""Shared Torch model definition for the train script and miner runtime."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from poker44.utils.hand_features import CHUNK_FEATURE_NAMES


class MinerNet(nn.Module):
    def __init__(self, input_dim: int | None = None, hidden_dim: int = 96):
        super().__init__()
        input_dim = input_dim or len(CHUNK_FEATURE_NAMES)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(p=0.15)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x.squeeze(1)
