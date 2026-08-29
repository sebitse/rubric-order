from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rubric_order.api import (
    build_rubrics,
    estimate_experiment,
    experiment_completeness,
    judge_responses,
    load_experiment_config,
    plan_judging,
    plan_rubric_generation,
    prepare_experiment,
)
from rubric_order.run import read_json
from tests.test_pipeline import MockOllama


class ApiTests(unittest.TestCase):
    def test_pilot_estimate_and_filtered_plan(self) -> None:
        config = load_experiment_config("configs/pilot.yaml", "HelpSteer2")
        estimate = estimate_experiment(config)
        self.assertEqual(estimate["total_inference_calls"], 428)
        self.assertEqual(estimate["sequential_updater_calls"], 80)

        with tempfile.TemporaryDirectory() as temporary:
            prepared = self._prepared_config(Path(temporary))
            plan = plan_rubric_generation(
                prepared,
                "api-run",
                methods=["sequential"],
                blocks=[0],
                orders=["random"],
                replicates=[0],
            )
            self.assertEqual(plan["planned_calls"], 10)
            self.assertEqual(len(plan["jobs"]), 1)
            self.assertEqual(plan["jobs"][0]["order"], "random")

    def test_disabled_execution_and_budget_guards_make_no_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self._prepared_config(Path(temporary))
            client = MockOllama()
            filters = {
                "methods": ["sequential"],
                "blocks": [0],
                "orders": ["random"],
                "replicates": [0],
            }
            dry = build_rubrics(
                config,
                "api-run",
                execute_model_calls=False,
                max_calls=0,
                client=client,
                **filters,
            )
            self.assertFalse(dry["executed"])
            self.assertEqual(dry["actual_calls"], 0)
            self.assertEqual(client.generate_calls, 0)
            self.assertFalse((Path(temporary) / "runs" / "api-run" / "calls.sqlite3").exists())
            with self.assertRaises(ValueError):
                build_rubrics(
                    config,
                    "api-run",
                    execute_model_calls=True,
                    max_calls=9,
                    client=client,
                    **filters,
                )
            self.assertEqual(client.generate_calls, 0)

            judge_plan = plan_judging(config, "api-run", **filters)
            self.assertFalse(judge_plan["ready"])
            judge_dry = judge_responses(
                config,
                "api-run",
                execute_model_calls=False,
                max_calls=0,
                client=client,
                **filters,
            )
            self.assertFalse(judge_dry["executed"])
            with self.assertRaises(ValueError):
                judge_responses(
                    config,
                    "api-run",
                    execute_model_calls=True,
                    max_calls=100,
                    client=client,
                    **filters,
                )
            self.assertEqual(client.generate_calls, 0)

    def test_public_execution_and_completeness_reporting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self._prepared_config(Path(temporary))
            filters = {
                "methods": ["sequential"],
                "blocks": [0],
                "orders": ["random"],
                "replicates": [0],
            }
            updater = MockOllama()
            result = build_rubrics(
                config,
                "api-run",
                execute_model_calls=True,
                max_calls=10,
                client=updater,
                **filters,
            )
            self.assertTrue(result["executed"])
            self.assertEqual(result["actual_calls"], 10)
            judge_plan = plan_judging(config, "api-run", **filters)
            self.assertTrue(judge_plan["ready"])
            self.assertEqual(judge_plan["planned_calls"], 6)

            judge = MockOllama()
            with self.assertRaises(ValueError):
                judge_responses(
                    config,
                    "api-run",
                    execute_model_calls=True,
                    max_calls=5,
                    client=judge,
                    **filters,
                )
            self.assertEqual(judge.generate_calls, 0)
            judged = judge_responses(
                config,
                "api-run",
                execute_model_calls=True,
                max_calls=6,
                client=judge,
                **filters,
            )
            self.assertTrue(judged["executed"])
            self.assertEqual(judged["judgment_summary"]["completed"], 6)
            manifest = read_json(Path(temporary) / "runs" / "api-run" / "manifest.json")
            self.assertGreater(manifest["model_calls_made"], 0)
            completeness = experiment_completeness(config, "api-run")
            self.assertFalse(completeness["complete"])
            self.assertGreater(len(completeness["missing_rubrics"]), 0)

    def test_explicit_partial_judge_canary_is_resume_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self._prepared_config(Path(temporary))
            filters = {
                "methods": ["sequential"],
                "blocks": [0],
                "orders": ["random"],
                "replicates": [0],
            }
            build_rubrics(
                config,
                "api-run",
                execute_model_calls=True,
                max_calls=10,
                client=MockOllama(),
                **filters,
            )
            canary_client = MockOllama()
            canary = judge_responses(
                config,
                "api-run",
                execute_model_calls=True,
                max_calls=1,
                allow_partial_execution=True,
                client=canary_client,
                **filters,
            )
            self.assertEqual(canary["status"], "partial_budget_stop")
            self.assertEqual(canary["actual_calls"], canary_client.generate_calls)
            self.assertEqual(canary["actual_calls"], 1)
            resume_client = MockOllama()
            resumed = judge_responses(
                config,
                "api-run",
                execute_model_calls=True,
                max_calls=5,
                client=resume_client,
                **filters,
            )
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(resumed["actual_calls"], resume_client.generate_calls)
            self.assertEqual(resumed["actual_calls"], 5)
            self.assertEqual(resumed["judgment_summary"]["completed"], 6)

    def test_single_updater_canary_is_reused_without_changing_global_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self._prepared_config(Path(temporary))
            filters = {
                "methods": ["batch"],
                "blocks": [0],
                "orders": ["ascending"],
                "replicates": [0],
            }
            first_client = MockOllama()
            first = build_rubrics(
                config,
                "api-run",
                execute_model_calls=True,
                max_calls=1,
                client=first_client,
                **filters,
            )
            self.assertEqual(first["actual_calls"], 1)
            self.assertEqual(first_client.generate_calls, 1)
            resume_client = MockOllama()
            resumed = build_rubrics(
                config,
                "api-run",
                execute_model_calls=True,
                max_calls=1,
                client=resume_client,
                **filters,
            )
            self.assertEqual(resumed["actual_calls"], 0)
            self.assertEqual(resume_client.generate_calls, 0)
            self.assertEqual(resumed["final_phase_plan"]["execution_fresh_calls"], 0)

    def _prepared_config(self, root: Path) -> dict[str, object]:
        data_path = root / "synthetic.jsonl"
        rows = [
            {
                "prompt": f"calibration-{score}-{index}",
                "response": f"response-{score}-{index}",
                "helpfulness": score,
            }
            for score in range(5)
            for index in range(8)
        ]
        rows.extend(
            {
                "prompt": f"paired-{prompt_index}",
                "response": f"choice-{response_index}",
                "helpfulness": (prompt_index + response_index) % 5,
            }
            for prompt_index in range(4)
            for response_index in range(2)
        )
        data_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        config = load_experiment_config(
            "configs/pilot.yaml",
            data_path,
            overrides={
                "output_root": str(root / "runs"),
                "sampling.calibration_blocks": 1,
                "sampling.heldout_prompts": 3,
                "analysis.bootstrap_repetitions": 5,
                "analysis.make_plots": False,
            },
        )
        prepare_experiment(config, "api-run", resume=False)
        return config


if __name__ == "__main__":
    unittest.main()
