from __future__ import annotations

import unittest

from rubric_order.data import normalize_records
from rubric_order.sampling import assert_order_invariants, build_calibration_blocks


class SamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        raw = [
            {
                "prompt": f"prompt-{score}-{index}",
                "response": f"response-{score}-{index}",
                "label": score,
            }
            for score in range(5)
            for index in range(5)
        ]
        self.rows = normalize_records(
            raw,
            {"prompt": "prompt", "response": "response", "label": "label"},
            "test",
        )

    def test_balanced_disjoint_blocks_and_orders(self) -> None:
        blocks, metadata = build_calibration_blocks(self.rows, 2, 2, seed=42)
        self.assertTrue(metadata["all_blocks_disjoint"])
        seen: set[str] = set()
        for block in blocks:
            counts = {score: 0 for score in range(5)}
            for row in block["examples"]:
                counts[row["human_score"]] += 1
                self.assertNotIn(row["example_id"], seen)
                seen.add(row["example_id"])
            self.assertEqual(counts, {score: 2 for score in range(5)})
            assert_order_invariants(block["orders"])
            evidence_sets = [set(order) for order in block["orders"].values()]
            self.assertTrue(all(value == evidence_sets[0] for value in evidence_sets))
            self.assertEqual(
                block["orders"]["reverse_random"],
                list(reversed(block["orders"]["random"])),
            )

    def test_invariant_rejects_changed_evidence(self) -> None:
        blocks, _ = build_calibration_blocks(self.rows, 1, 2, seed=42)
        orders = blocks[0]["orders"]
        orders["descending"][0] = "different"
        with self.assertRaises(ValueError):
            assert_order_invariants(orders)


if __name__ == "__main__":
    unittest.main()
