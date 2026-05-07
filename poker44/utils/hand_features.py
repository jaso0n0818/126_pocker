"""Extract poker hand features for bot/human behavioral signature modeling.

This feature set is designed to capture risk-seeking and risk-averse
behavioral patterns from poker hand records, including aggression,
normalized betting scale, showdown behavior, and profit/stack signals.
These signals align with poker AI research that distinguishes human and
algorithmic decision-making via behavioral signatures.
"""

from collections import Counter
from typing import Any, Dict, List

FEATURE_NAMES = [
    "num_actions",
    "street_depth",
    "showdown",
    "aggression_ratio",
    "raise_ratio",
    "call_ratio",
    "check_ratio",
    "fold_ratio",
    "avg_normalized_amount_bb",
    "max_normalized_amount_bb",
    "pot_growth_bb",
    "hero_stack_bb",
    "hero_profit_bb",
]

CHUNK_FEATURE_NAMES = [
    f"{stat}_{name}"
    for stat in ("mean", "std", "min", "max")
    for name in FEATURE_NAMES
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _convert_label(label: Any) -> int:
    raw = str(label).strip().lower()
    if raw in {"bot", "ai", "algorithm", "computer", "machine"}:
        return 1
    return 0


def extract_hand_features(hand: Dict[str, Any]) -> List[float]:
    metadata = hand.get("metadata") or {}
    actions = hand.get("actions") or []
    outcome = hand.get("outcome") or {}
    players = hand.get("players") or []

    num_actions = len(actions)
    streets = {action.get("street") for action in actions if action.get("street") is not None}
    street_depth = _clamp01(len(streets) / 4.0)
    showdown = 1.0 if bool(outcome.get("showdown")) else 0.0

    action_counts = Counter(action.get("action_type") for action in actions)
    meaningful_actions = max(
        1,
        sum(
            action_counts.get(kind, 0)
            for kind in ("call", "check", "bet", "raise", "fold")
        ),
    )

    call_ratio = action_counts.get("call", 0) / meaningful_actions
    check_ratio = action_counts.get("check", 0) / meaningful_actions
    bet_ratio = action_counts.get("bet", 0) / meaningful_actions
    raise_ratio = action_counts.get("raise", 0) / meaningful_actions
    fold_ratio = action_counts.get("fold", 0) / meaningful_actions
    aggression_ratio = (action_counts.get("bet", 0) + action_counts.get("raise", 0)) / meaningful_actions

    normalized_amounts = [
        _safe_float(action.get("normalized_amount_bb"), 0.0)
        for action in actions
        if action.get("normalized_amount_bb") is not None
    ]
    avg_normalized_amount_bb = float(sum(normalized_amounts) / len(normalized_amounts)) if normalized_amounts else 0.0
    max_normalized_amount_bb = float(max(normalized_amounts)) if normalized_amounts else 0.0

    bb = max(1.0, _safe_float(metadata.get("bb", 1.0)))
    pot_growth_bb = _safe_float(outcome.get("total_pot", 0.0)) / bb

    hero_seat = metadata.get("hero_seat")
    hero_uid = None
    hero_stack = 0.0
    for player in players:
        if player.get("seat") == hero_seat:
            hero_uid = player.get("player_uid")
            hero_stack = _safe_float(player.get("starting_stack", 0.0))
            break

    hero_stack_bb = hero_stack / bb
    payouts = outcome.get("payouts") or {}
    hero_profit_bb = _safe_float(payouts.get(hero_uid, 0.0)) / bb

    return [
        float(num_actions),
        street_depth,
        showdown,
        _clamp01(aggression_ratio),
        _clamp01(raise_ratio),
        _clamp01(call_ratio),
        _clamp01(check_ratio),
        _clamp01(fold_ratio),
        _clamp01(avg_normalized_amount_bb / 10.0),
        _clamp01(max_normalized_amount_bb / 20.0),
        _clamp01(pot_growth_bb / 20.0),
        _clamp01(hero_stack_bb / 50.0),
        _clamp01((hero_profit_bb + 5.0) / 10.0),
    ]


def extract_chunk_features(chunk: Any) -> List[float]:
    """Return one fixed feature vector for a validator scoring chunk.

    Live validator payloads are shaped as ``list[list[hand]]``: one score per
    chunk, and each chunk can contain one or many hands. Some older local
    training files contain plain hand dicts, so this accepts both shapes.
    """
    if isinstance(chunk, dict):
        features = extract_hand_features(chunk)
        return features + [0.0] * len(FEATURE_NAMES) + features + features

    if not isinstance(chunk, list) or not chunk:
        return [0.0] * len(CHUNK_FEATURE_NAMES)

    hand_features = [
        extract_hand_features(hand)
        for hand in chunk
        if isinstance(hand, dict)
    ]
    if not hand_features:
        return [0.0] * len(CHUNK_FEATURE_NAMES)

    means = []
    stds = []
    mins = []
    maxs = []
    for values in zip(*hand_features):
        values = list(values)
        mean = float(sum(values) / len(values))
        variance = float(sum((value - mean) ** 2 for value in values) / len(values))
        means.append(mean)
        stds.append(variance ** 0.5)
        mins.append(float(min(values)))
        maxs.append(float(max(values)))

    return means + stds + mins + maxs


def normalize_label(label: Any) -> int:
    return _convert_label(label)
