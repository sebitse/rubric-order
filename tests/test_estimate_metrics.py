from __future__ import annotations

import copy
import unittest
from pathlib import Path

from rubric_order.config import DEFAULTS, load_config
from rubric_order.estimate import estimate_calls
from rubric_order.metrics import (
    aggregate_agreement_table,
    agreement_metrics,
    calculate_agreement_table,
    common_tfidf_cosines,
    primary_metrics,
    quadratic_weighted_kappa,
    repeat_variation_metrics,
    spearman_correlation,
)


def prediction_rows(scores: dict[str, list[int]]) -> list[dict[str, object]]:
    orders = ["ascending", "descending", "random", "reverse_random"]
    rows: list[dict[str, object]] = []
    for example_id, values in scores.items():
        for order, score in zip(orders, values, strict=True):
            rows.append(
                {
                    "example_id": example_id,
                    "prompt_id": "shared",
                    "order": order,
                    "predicted_score": score,
                    "human_score": score,
                }
            )
    return rows


class EstimateAndMetricTests(unittest.TestCase):
    def test_pilot_call_count(self) -> None:
        config = copy.deepcopy(DEFAULTS)
        result = estimate_calls(config, heldout_response_count=20)
        self.assertEqual(result["sequential_updater_calls"], 80)
        self.assertEqual(result["batch_updater_calls"], 8)
        self.assertEqual(result["judge_calls"], 340)
        self.assertEqual(result["repeat_control_calls"], 0)
        self.assertEqual(result["total_inference_calls"], 428)

    def test_main_repeat_control_count(self) -> None:
        config = copy.deepcopy(DEFAULTS)
        config["sampling"]["calibration_blocks"] = 10
        config["sampling"]["heldout_prompts"] = 25
        config["repeat_control"].update({"enabled": True, "blocks": [0, 4, 9], "replicates": [1]})
        result = estimate_calls(config, heldout_response_count=50)
        self.assertEqual(result["repeat_control_updater_calls"], 33)
        self.assertEqual(result["repeat_control_judge_calls"], 300)
        self.assertEqual(result["repeat_control_calls"], 333)

    def test_exact_v2_call_counts(self) -> None:
        root = Path(__file__).parents[1]
        pilot = load_config(root / "configs" / "pilot_v2.yaml")
        main = load_config(root / "configs" / "main_v2.yaml")
        self.assertEqual(estimate_calls(pilot)["total_inference_calls"], 632)
        self.assertEqual(estimate_calls(main)["total_inference_calls"], 5156)

    def test_primary_metric_known_flip_example(self) -> None:
        result = primary_metrics(prediction_rows({"a": [0, 1, 0, 1], "b": [1, 0, 1, 0]}))
        self.assertAlmostEqual(result["calibration_order_instability"], 0.5)
        self.assertEqual(result["score_change_rate"], 1.0)
        self.assertEqual(result["mean_score_range"], 1.0)
        self.assertAlmostEqual(result["pairwise_ranking_flip_rate"], 4 / 6)
        self.assertEqual(result["tie_change_rate"], 0.0)

    def test_tie_change_metric(self) -> None:
        result = primary_metrics(prediction_rows({"a": [1, 1, 1, 1], "b": [1, 2, 1, 2]}))
        self.assertEqual(result["pairwise_ranking_flip_rate"], 0.0)
        self.assertAlmostEqual(result["tie_change_rate"], 4 / 6)

    def test_agreement_metrics_known_examples(self) -> None:
        rows = [{"human_score": score, "predicted_score": score} for score in range(5)]
        result = agreement_metrics(rows)
        self.assertEqual(result["quadratic_weighted_kappa"], 1.0)
        self.assertEqual(result["spearman"], 1.0)
        self.assertEqual(result["mean_absolute_error"], 0.0)
        self.assertEqual(result["mean_signed_error"], 0.0)
        self.assertEqual(quadratic_weighted_kappa([0, 1, 2], [0, 1, 2]), 1.0)
        self.assertEqual(spearman_correlation([0, 1, 2], [0, 1, 2]), 1.0)

    def test_agreement_condition_mean_and_static_deduplication(self) -> None:
        rows = []
        for block in (0, 1):
            for order in ("ascending", "descending"):
                for score in range(5):
                    rows.append(
                        {
                            "method": "batch",
                            "block_id": block,
                            "order": order,
                            "replicate": 0,
                            "example_id": f"e{score}",
                            "human_score": score,
                            "predicted_score": score if block == 0 else 4 - score,
                        }
                    )
        for order in ("ascending", "descending"):
            for score in range(5):
                rows.append(
                    {
                        "method": "static",
                        "block_id": 0,
                        "order": order,
                        "replicate": 0,
                        "example_id": f"e{score}",
                        "human_score": score,
                        "predicted_score": score,
                    }
                )
        table = calculate_agreement_table(rows)
        aggregate = aggregate_agreement_table(table)
        self.assertEqual(aggregate["batch"]["condition_count"], 4)
        self.assertEqual(aggregate["static"]["condition_count"], 1)

    def test_common_tfidf_and_repeat_variation(self) -> None:
        pair_only = common_tfidf_cosines(["alpha beta", "alpha gamma"])[(0, 1)]
        common = common_tfidf_cosines(["alpha beta", "alpha gamma", "beta gamma delta"])[(0, 1)]
        self.assertNotEqual(pair_only, common)
        rows = []
        scores = {0: (1, 2), 1: (1, 2), 2: (2, 1)}
        for replicate, values in scores.items():
            for index, score in enumerate(values):
                rows.append(
                    {
                        "method": "sequential",
                        "block_id": 0,
                        "order": "random",
                        "replicate": replicate,
                        "example_id": f"e{index}",
                        "prompt_id": "p",
                        "predicted_score": score,
                    }
                )
        result = repeat_variation_metrics(rows)
        self.assertGreater(result["repeat_instability"], 0)
        self.assertLess(result["exact_replicate_agreement"], 1)


if __name__ == "__main__":
    unittest.main()
