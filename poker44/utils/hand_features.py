"""Extract poker hand features for bot/human behavioral signature modeling.

This feature set is designed to capture risk-seeking and risk-averse
behavioral patterns from poker hand records, including aggression,
normalized betting scale, showdown behavior, and profit/stack signals.
These signals align with poker AI research that distinguishes human and
algorithmic decision-making via behavioral signatures.
"""

import math
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
    "action_entropy",
    "actor_entropy",
    "nonzero_amount_ratio",
    "amount_std_bb",
    "aggressive_street_coverage",
    "hero_action_ratio",
    "hero_aggression_ratio",
    "active_actor_ratio",
    "repeated_action_ratio",
    "passive_ratio",
    "pot_volatility_bb",
    "large_bet_ratio",
    "street_switch_ratio",
    "actor_switch_ratio",
    "amount_entropy",
    "bet_sizing_cv",
    "hero_amount_share",
    "opening_aggression",
    "actor_action_edge_entropy",
    "street_action_edge_entropy",
    "actor_degree_concentration",
    "hero_response_ratio",
    "amount_delta_cv",
    "sizing_repetition_ratio",
]

CHUNK_FEATURE_NAMES = [
    f"{stat}_{name}"
    for stat in ("mean", "std", "min", "max")
    for name in FEATURE_NAMES
] + [
    "chunk_hand_count_norm",
    "chunk_action_count_cv",
    "chunk_hero_seat_entropy",
    "chunk_first_actor_entropy",
    "chunk_aggression_stability",
    "chunk_amount_stability",
    "chunk_passivity_stability",
    "chunk_actor_mix_stability",
    "chunk_actor_transition_entropy",
    "chunk_street_transition_entropy",
    "chunk_opening_aggression_stability",
    "chunk_sizing_repetition_stability",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalized_entropy(values: List[Any]) -> float:
    values = [value for value in values if value is not None]
    if len(values) <= 1:
        return 0.0
    counts = Counter(values)
    entropy = 0.0
    for count in counts.values():
        probability = count / len(values)
        entropy -= probability * math.log(probability)
    return _clamp01(entropy / max(math.log(len(counts)), 1e-9))


def _switch_ratio(values: List[Any]) -> float:
    values = [value for value in values if value is not None]
    if len(values) <= 1:
        return 0.0
    switches = sum(1 for left, right in zip(values, values[1:]) if left != right)
    return _clamp01(switches / (len(values) - 1))


def _coefficient_of_variation(values: List[float]) -> float:
    values = [float(value) for value in values if value is not None]
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if abs(mean) < 1e-9:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return (variance ** 0.5) / abs(mean)


def _concentration(values: List[Any]) -> float:
    values = [value for value in values if value is not None]
    if not values:
        return 0.0
    counts = Counter(values)
    return _clamp01(max(counts.values()) / len(values))


def _mean_abs_delta(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return sum(abs(right - left) for left, right in zip(values, values[1:])) / (len(values) - 1)


def _pair_values(left_values: List[Any], right_values: List[Any]) -> List[tuple[Any, Any]]:
    return [
        (left, right)
        for left, right in zip(left_values, right_values)
        if left is not None and right is not None
    ]


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
    action_types = [action.get("action_type") for action in actions]
    action_entropy = _normalized_entropy(action_types)
    actor_seats = [action.get("actor_seat") for action in actions]
    actor_entropy = _normalized_entropy(actor_seats)
    nonzero_amount_ratio = (
        sum(1 for amount in normalized_amounts if amount > 0.0) / len(normalized_amounts)
        if normalized_amounts
        else 0.0
    )
    if normalized_amounts:
        amount_mean = sum(normalized_amounts) / len(normalized_amounts)
        amount_variance = sum((amount - amount_mean) ** 2 for amount in normalized_amounts) / len(normalized_amounts)
        amount_std_bb = amount_variance ** 0.5
    else:
        amount_std_bb = 0.0
    aggressive_streets = {
        action.get("street")
        for action in actions
        if action.get("action_type") in {"bet", "raise"}
    }
    aggressive_street_coverage = _clamp01(len(aggressive_streets) / 4.0)
    hero_actions = [action for action in actions if action.get("actor_seat") == hero_seat]
    hero_action_ratio = len(hero_actions) / max(1, len(actions))
    hero_meaningful = [
        action
        for action in hero_actions
        if action.get("action_type") in {"call", "check", "bet", "raise", "fold"}
    ]
    hero_aggression_ratio = (
        sum(1 for action in hero_meaningful if action.get("action_type") in {"bet", "raise"})
        / max(1, len(hero_meaningful))
    )
    active_actor_ratio = len(set(actor_seats)) / max(1, len(players))
    repeated_action_ratio = max(action_counts.values(), default=0) / max(1, len(actions))
    passive_ratio = (action_counts.get("call", 0) + action_counts.get("check", 0)) / meaningful_actions
    pot_deltas = [
        abs(_safe_float(action.get("pot_after"), 0.0) - _safe_float(action.get("pot_before"), 0.0)) / bb
        for action in actions
        if action.get("pot_after") is not None and action.get("pot_before") is not None
    ]
    pot_volatility_bb = sum(pot_deltas) / len(pot_deltas) if pot_deltas else 0.0
    large_bet_ratio = (
        sum(1 for amount in normalized_amounts if amount >= 10.0) / len(normalized_amounts)
        if normalized_amounts
        else 0.0
    )
    street_switch_ratio = _switch_ratio([action.get("street") for action in actions])
    actor_switch_ratio = _switch_ratio(actor_seats)
    amount_bins = [
        "zero" if amount <= 0.0 else "small" if amount < 3.0 else "medium" if amount < 10.0 else "large"
        for amount in normalized_amounts
    ]
    amount_entropy = _normalized_entropy(amount_bins)
    bet_sizing_cv = _coefficient_of_variation([amount for amount in normalized_amounts if amount > 0.0])
    hero_amount_total = sum(
        _safe_float(action.get("normalized_amount_bb"), 0.0)
        for action in hero_actions
        if action.get("normalized_amount_bb") is not None
    )
    amount_total = sum(abs(amount) for amount in normalized_amounts)
    hero_amount_share = hero_amount_total / max(amount_total, 1e-9) if amount_total > 0.0 else 0.0
    opening_action = next(
        (
            action.get("action_type")
            for action in actions
            if action.get("action_type") in {"call", "check", "bet", "raise", "fold"}
        ),
        None,
    )
    opening_aggression = 1.0 if opening_action in {"bet", "raise"} else 0.0
    actor_action_edge_entropy = _normalized_entropy(_pair_values(actor_seats, action_types))
    street_action_edge_entropy = _normalized_entropy(
        _pair_values([action.get("street") for action in actions], action_types)
    )
    actor_degree_concentration = _concentration(actor_seats)
    hero_response_ratio = (
        len(hero_actions[1:]) / max(1, len(actions) - 1)
        if len(actions) > 1
        else hero_action_ratio
    )
    positive_amounts = [amount for amount in normalized_amounts if amount > 0.0]
    amount_delta_cv = _coefficient_of_variation(
        [
            abs(right - left)
            for left, right in zip(positive_amounts, positive_amounts[1:])
        ]
    )
    sizing_repetition_ratio = _concentration(
        [round(amount, 2) for amount in positive_amounts]
    )

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
        action_entropy,
        actor_entropy,
        _clamp01(nonzero_amount_ratio),
        _clamp01(amount_std_bb / 20.0),
        aggressive_street_coverage,
        _clamp01(hero_action_ratio),
        _clamp01(hero_aggression_ratio),
        _clamp01(active_actor_ratio),
        _clamp01(repeated_action_ratio),
        _clamp01(passive_ratio),
        _clamp01(pot_volatility_bb / 20.0),
        _clamp01(large_bet_ratio),
        _clamp01(street_switch_ratio),
        _clamp01(actor_switch_ratio),
        amount_entropy,
        _clamp01(bet_sizing_cv / 2.0),
        _clamp01(hero_amount_share),
        opening_aggression,
        actor_action_edge_entropy,
        street_action_edge_entropy,
        actor_degree_concentration,
        _clamp01(hero_response_ratio),
        _clamp01(amount_delta_cv / 2.0),
        sizing_repetition_ratio,
    ]


def extract_chunk_features(chunk: Any) -> List[float]:
    """Return one fixed feature vector for a validator scoring chunk.

    Live validator payloads are shaped as ``list[list[hand]]``: one score per
    chunk, and each chunk can contain one or many hands. Some older local
    training files contain plain hand dicts, so this accepts both shapes.
    """
    if isinstance(chunk, dict):
        features = extract_hand_features(chunk)
        return features + [0.0] * len(FEATURE_NAMES) + features + features + [0.0] * 12

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

    action_counts = [feature[FEATURE_NAMES.index("num_actions")] for feature in hand_features]
    aggression = [feature[FEATURE_NAMES.index("aggression_ratio")] for feature in hand_features]
    amount_means = [feature[FEATURE_NAMES.index("avg_normalized_amount_bb")] for feature in hand_features]
    passivity = [feature[FEATURE_NAMES.index("passive_ratio")] for feature in hand_features]
    actor_entropy = [feature[FEATURE_NAMES.index("actor_entropy")] for feature in hand_features]
    opening_aggression = [
        feature[FEATURE_NAMES.index("opening_aggression")]
        for feature in hand_features
    ]
    sizing_repetition = [
        feature[FEATURE_NAMES.index("sizing_repetition_ratio")]
        for feature in hand_features
    ]
    hero_seats = [
        hand.get("metadata", {}).get("hero_seat")
        for hand in chunk
        if isinstance(hand, dict)
    ]
    first_actors = [
        (hand.get("actions") or [{}])[0].get("actor_seat")
        for hand in chunk
        if isinstance(hand, dict) and (hand.get("actions") or [])
    ]
    sequence_features = [
        _clamp01(len(hand_features) / 64.0),
        _clamp01(_coefficient_of_variation(action_counts)),
        _normalized_entropy(hero_seats),
        _normalized_entropy(first_actors),
        _clamp01(1.0 - _mean_abs_delta(aggression)),
        _clamp01(1.0 - _mean_abs_delta(amount_means)),
        _clamp01(1.0 - _mean_abs_delta(passivity)),
        _clamp01(1.0 - _mean_abs_delta(actor_entropy)),
        _normalized_entropy(list(zip(first_actors, first_actors[1:]))),
        _normalized_entropy(
            [
                (
                    (hand.get("actions") or [{}])[0].get("street"),
                    (hand.get("actions") or [{}])[0].get("action_type"),
                )
                for hand in chunk
                if isinstance(hand, dict) and (hand.get("actions") or [])
            ]
        ),
        _clamp01(1.0 - _mean_abs_delta(opening_aggression)),
        _clamp01(1.0 - _mean_abs_delta(sizing_repetition)),
    ]

    return means + stds + mins + maxs + sequence_features


def normalize_label(label: Any) -> int:
    return _convert_label(label)
