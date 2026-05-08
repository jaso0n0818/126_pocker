"""Shared Torch model definition for the train script and miner runtime."""

from __future__ import annotations

import torch
import torch.nn as nn

from poker44.utils.hand_features import CHUNK_FEATURE_NAMES


MODEL_ARCHITECTURE_VERSION = 4


def logit(value: float, eps: float = 1e-6) -> float:
    value = max(eps, min(1.0 - eps, float(value)))
    return torch.logit(torch.tensor(value)).item()


def apply_score_shift(scores: torch.Tensor, score_shift: float = 0.0) -> torch.Tensor:
    if not score_shift:
        return scores
    logits = torch.logit(scores.clamp(1e-6, 1.0 - 1e-6))
    return torch.sigmoid(logits + float(score_shift))


class MinerNet(nn.Module):
    def __init__(self, input_dim: int | None = None, hidden_dim: int = 128):
        super().__init__()
        input_dim = input_dim or len(CHUNK_FEATURE_NAMES)
        sequence_dim = sum(1 for name in CHUNK_FEATURE_NAMES if name.startswith("chunk_"))
        stat_dim = input_dim - sequence_dim
        self.stat_dim = stat_dim
        self.sequence_dim = sequence_dim
        self.stat_branch = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.18),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.sequence_branch = nn.Sequential(
            nn.Linear(sequence_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.12),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear((hidden_dim // 2) + (hidden_dim // 4), hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.10),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stat_x = x[:, : self.stat_dim]
        sequence_x = x[:, self.stat_dim :]
        stat_repr = self.stat_branch(stat_x)
        sequence_repr = self.sequence_branch(sequence_x)
        logits = self.head(torch.cat([stat_repr, sequence_repr], dim=1)).squeeze(1)
        return torch.sigmoid(logits)
