from __future__ import annotations

import copy
import unittest

from rubric_order.config import DEFAULTS
from rubric_order.exclusions import apply_exclusions
from rubric_order.judge_validation import (
    choose_judge_candidate,
    estimate_judge_validation,
    judge_gate,
    select_judge_subsets,
)


class JudgeValidationV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {
                "example_id": f"e-{score}-{index}",
                "prompt_id": f"p-{score}-{index}",
                "prompt": f"prompt {score} {index}",
                "response": f"response {score} {index}",
                "human_score": score,
            }
            for score in range(5)
            for index in range(35)
        ]

    def test_balanced_prompt_disjoint_selection_is_deterministic(self) -> None:
        first = select_judge_subsets(self.rows, dev_per_score=10, validation_per_score=20, seed=42)
        second = select_judge_subsets(self.rows, dev_per_score=10, validation_per_score=20, seed=42)
        self.assertEqual(first, second)
        dev, validation = first
        self.assertEqual(len(dev), 50)
        self.assertEqual(len(validation), 100)
        self.assertEqual(
            {score: sum(row["human_score"] == score for row in dev) for score in range(5)},
            {score: 10 for score in range(5)},
        )
        self.assertEqual(
            {score: sum(row["human_score"] == score for row in validation) for score in range(5)},
            {score: 20 for score in range(5)},
        )
        self.assertFalse(
            {row["prompt_id"] for row in dev} & {row["prompt_id"] for row in validation}
        )

    def test_prompt_exclusion_removes_associated_response(self) -> None:
        paired = [
            {"example_id": "a", "prompt_id": "same"},
            {"example_id": "b", "prompt_id": "same"},
            {"example_id": "c", "prompt_id": "other"},
        ]
        kept, report = apply_exclusions(paired, {"a"}, {"same"})
        self.assertEqual([row["example_id"] for row in kept], ["c"])
        self.assertEqual(report["removed_rows"], 2)

    def test_selection_enforces_distinct_prompts_within_each_score(self) -> None:
        duplicated = [dict(row) for row in self.rows]
        for score in range(5):
            duplicated.append(
                {
                    **duplicated[score * 35],
                    "example_id": f"duplicate-{score}",
                    "response": f"alternative response {score}",
                }
            )
        dev, validation = select_judge_subsets(
            duplicated,
            dev_per_score=10,
            validation_per_score=20,
            seed=42,
        )
        selected = dev + validation
        self.assertEqual(len({row["prompt_id"] for row in selected}), len(selected))

    def test_deterministic_candidate_selection(self) -> None:
        candidates = [
            {
                "model": "llama",
                "candidate": "b",
                "quadratic_weighted_kappa": 0.4,
                "spearman": 0.5,
                "mean_absolute_error": 1.0,
            },
            {
                "model": "llama",
                "candidate": "a",
                "quadratic_weighted_kappa": 0.4,
                "spearman": 0.5,
                "mean_absolute_error": 0.8,
            },
        ]
        self.assertEqual(choose_judge_candidate(candidates)["candidate"], "a")

    def test_gate_rejects_constant_and_undefined_metrics(self) -> None:
        thresholds = {
            "minimum_qwk": 0.3,
            "minimum_spearman": 0.3,
            "maximum_mae": 1.25,
        }
        result = judge_gate(
            {
                "quadratic_weighted_kappa": None,
                "spearman": None,
                "mean_absolute_error": 0.0,
                "constant_predictions": True,
                "missing_predictions": 0,
                "unresolved_schema_errors": 0,
            },
            thresholds,
        )
        self.assertFalse(result["passed"])
        self.assertIn("undefined_qwk", result["reasons"])
        self.assertIn("constant_predictions", result["reasons"])

    def test_call_estimate_is_200_plus_explicit_fallback(self) -> None:
        config = copy.deepcopy(DEFAULTS)
        config["judge_validation"] = {
            "dev_per_score": 10,
            "validation_per_score": 20,
            "candidates": ["judge_absolute", "judge_anchor_first"],
        }
        normal = estimate_judge_validation(config)
        fallback = estimate_judge_validation(config, include_qwen_fallback=True)
        self.assertEqual(normal["normal_total_calls"], 200)
        self.assertEqual(normal["optional_qwen_dev_calls"], 0)
        self.assertEqual(fallback["optional_qwen_dev_calls"], 100)
        self.assertEqual(fallback["total_with_requested_fallback"], 300)


if __name__ == "__main__":
    unittest.main()
