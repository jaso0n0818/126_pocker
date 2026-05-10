"""Shared Torch model definition for the train script and miner runtime."""

from __future__ import annotations

import torch
import torch.nn as nn

from poker44.utils.hand_features import CHUNK_FEATURE_NAMES, extract_hand_features


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


class LegacyMinerNetV1(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).squeeze(1)


class LegacyMinerNetV2(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, mid_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, mid_dim)
        self.fc3 = nn.Linear(mid_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x)).squeeze(1)


def normalize_state_dict(state: dict) -> dict:
    return {str(k).replace("module.", ""): v for k, v in state.items()}


def build_model_for_state_dict(state: dict) -> nn.Module:
    if "stat_branch.0.weight" in state:
        return MinerNet()

    if "fc1.weight" in state and "fc2.weight" in state and "fc3.weight" in state:
        input_dim = int(state["fc1.weight"].shape[1])
        hidden_dim = int(state["fc1.weight"].shape[0])
        mid_dim = int(state["fc2.weight"].shape[0])
        return LegacyMinerNetV2(input_dim=input_dim, hidden_dim=hidden_dim, mid_dim=mid_dim)

    if "fc1.weight" in state and "fc2.weight" in state:
        input_dim = int(state["fc1.weight"].shape[1])
        hidden_dim = int(state["fc1.weight"].shape[0])
        return LegacyMinerNetV1(input_dim=input_dim, hidden_dim=hidden_dim)

    raise ValueError(f"Unsupported checkpoint keys: {sorted(state.keys())[:8]}")


def model_input_dim(model: nn.Module) -> int | None:
    fc1 = getattr(model, "fc1", None)
    if fc1 is not None and hasattr(fc1, "in_features"):
        return int(fc1.in_features)
    return None


def legacy_chunk_features(chunk, input_dim: int) -> list[float]:
    if input_dim not in {13, 52}:
        raise ValueError(f"Unsupported legacy input_dim={input_dim}")

    if isinstance(chunk, dict):
        hand_features = [extract_hand_features(chunk)[:13]]
    elif isinstance(chunk, list):
        hand_features = [
            extract_hand_features(hand)[:13]
            for hand in chunk
            if isinstance(hand, dict)
        ]
    else:
        hand_features = []

    if not hand_features:
        return [0.0] * input_dim

    values = torch.tensor(hand_features, dtype=torch.float32)
    means = values.mean(dim=0)
    if input_dim == 13:
        return means.tolist()

    stds = values.std(dim=0, unbiased=False)
    mins = values.min(dim=0).values
    maxs = values.max(dim=0).values
    return torch.cat([means, stds, mins, maxs]).tolist()
