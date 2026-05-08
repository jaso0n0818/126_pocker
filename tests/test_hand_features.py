import unittest

from poker44.utils.hand_features import (
    CHUNK_FEATURE_NAMES,
    FEATURE_NAMES,
    extract_chunk_features,
    extract_hand_features,
)


class HandFeatureTests(unittest.TestCase):
    def test_single_hand_chunk_matches_hand_features(self):
        hand = {
            "metadata": {"bb": 0.02, "hero_seat": 1},
            "players": [{"seat": 1, "player_uid": "hero", "starting_stack": 2.0}],
            "actions": [
                {"street": "preflop", "action_type": "raise", "normalized_amount_bb": 2.0},
                {"street": "preflop", "action_type": "fold", "normalized_amount_bb": 0.0},
            ],
            "outcome": {
                "showdown": False,
                "total_pot": 0.06,
                "payouts": {"hero": 0.04},
            },
        }

        hand_features = extract_hand_features(hand)
        expected = (
            hand_features
            + [0.0] * len(FEATURE_NAMES)
            + hand_features
            + hand_features
            + [0.0] * 12
        )

        self.assertEqual(extract_chunk_features(hand), expected)

    def test_multi_hand_chunk_averages_hand_features(self):
        hand_a = {
            "metadata": {"bb": 0.02, "hero_seat": 1},
            "players": [{"seat": 1, "player_uid": "hero_a", "starting_stack": 2.0}],
            "actions": [
                {"street": "preflop", "action_type": "raise", "normalized_amount_bb": 2.0},
            ],
            "outcome": {"showdown": False, "total_pot": 0.04, "payouts": {}},
        }
        hand_b = {
            "metadata": {"bb": 0.02, "hero_seat": 2},
            "players": [{"seat": 2, "player_uid": "hero_b", "starting_stack": 1.0}],
            "actions": [
                {"street": "preflop", "action_type": "call", "normalized_amount_bb": 1.0},
                {"street": "flop", "action_type": "check", "normalized_amount_bb": 0.0},
            ],
            "outcome": {"showdown": True, "total_pot": 0.08, "payouts": {}},
        }

        means = [
            (a + b) / 2.0
            for a, b in zip(extract_hand_features(hand_a), extract_hand_features(hand_b))
        ]
        stds = [
            (((a - mean) ** 2 + (b - mean) ** 2) / 2.0) ** 0.5
            for a, b, mean in zip(extract_hand_features(hand_a), extract_hand_features(hand_b), means)
        ]
        mins = [
            min(a, b)
            for a, b in zip(extract_hand_features(hand_a), extract_hand_features(hand_b))
        ]
        maxs = [
            max(a, b)
            for a, b in zip(extract_hand_features(hand_a), extract_hand_features(hand_b))
        ]
        expected = means + stds + mins + maxs + [
            2 / 64,
            1 / 3,
            1.0,
            0.0,
            0.0,
            0.85,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
        ]

        self.assertEqual(extract_chunk_features([hand_a, hand_b]), expected)

    def test_empty_chunk_returns_zero_vector(self):
        self.assertEqual(extract_chunk_features([]), [0.0] * len(CHUNK_FEATURE_NAMES))


if __name__ == "__main__":
    unittest.main()
