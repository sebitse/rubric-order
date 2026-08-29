from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from rubric_order.api import (
    build_rubrics,
    judge_responses,
    plan_judging,
    plan_rubric_generation,
)
from rubric_order.config import load_config
from rubric_order.context import sha256_text
from rubric_order.estimate import estimate_calls
from rubric_order.run import prepare_run, read_json, read_jsonl, write_jsonl

ROOT = Path(__file__).parents[1].resolve()


class AuditedMockClient:
    def __init__(self, run_dir: Path, rubric: dict[str, Any]) -> None:
        self.run_dir = run_dir
        self.rubric = rubric
        self.generate_calls = 0
        self.updater_calls = 0
        self.judge_calls = 0
        self.health_checks = 0

    def health_and_models(self, required_models: list[str]) -> dict[str, Any]:
        self.health_checks += 1
        return {"ok": True, "models": required_models, "missing": []}

    def generate(self, **kwargs: Any) -> dict[str, Any]:
        self.generate_calls += 1
        prompt_hash = sha256_text(kwargs["prompt"])
        if "score" in kwargs["response_schema"].get("properties", {}):
            self.judge_calls += 1
            plans = read_jsonl(self.run_dir / "artifacts" / "request_plans" / "judge.jsonl")
            self.assert_materialized(plans, prompt_hash)
            response = json.dumps({"score": self.judge_calls % 5})
        else:
            self.updater_calls += 1
            plans = read_jsonl(self.run_dir / "artifacts" / "request_plans" / "updater.jsonl")
            self.assert_materialized(plans, prompt_hash)
            response = json.dumps(self.rubric)
        return {
            "response": response,
            "latency_seconds": 0.001,
            "prompt_eval_count": 50,
            "eval_count": 2,
        }

    def unload(self, _model: str) -> None:
        return None

    def assert_materialized(self, plans: list[dict[str, Any]], prompt_hash: str) -> None:
        matches = [row for row in plans if row["rendered_prompt_hash"] == prompt_hash]
        if not matches or not matches[-1].get("materialized_before_client_call"):
            raise AssertionError("request plan was not persisted before the client call")


class PhaseAwarePreflightTests(unittest.TestCase):
    def test_updater_phases_hash_parity_chain_bounds_and_judging_deduplication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, run_dir = self._prepared(Path(temporary), "phase-run")
            initial_plan = plan_rubric_generation(config, "phase-run")
            self.assertTrue(initial_plan["ready"])
            self.assertEqual(initial_plan["exact_requests"], 8)
            self.assertEqual(initial_plan["deferred_requests"], 36)
            self.assertEqual(initial_plan["planned_physical_updater_calls"], 44)
            self.assertEqual(initial_plan["batch_exact_requests"], 4)
            self.assertEqual(initial_plan["sequential_exact_requests"], 4)
            deferred = [
                row
                for row in initial_plan["context"]["requests"]
                if row["request_status"] == "deferred"
            ]
            self.assertTrue(all(row["request_hash"] is None for row in deferred))
            self.assertTrue(
                all(row["deferred_reason"] == "depends_on_previous_rubric" for row in deferred)
            )
            self.assertTrue(all(row["context_bound_is_conservative"] for row in deferred))
            self.assertFalse((run_dir / "calls.sqlite3").exists())

            blocked = plan_judging(config, "phase-run")
            self.assertEqual(blocked["status"], "blocked_by_rubric_generation")
            self.assertFalse(blocked["ready"])
            self.assertEqual(blocked["exact_requests"], 0)

            exact_before = {
                self._request_key(row): row["request_hash"]
                for row in initial_plan["context"]["requests"]
                if row["request_status"] == "exact"
            }
            deferred_bounds = {
                self._request_key(row): row["estimated_input_tokens"] for row in deferred
            }
            updater = AuditedMockClient(run_dir, read_json(config["rubric"]["initial"]))
            completed = build_rubrics(
                config,
                "phase-run",
                execute_model_calls=True,
                max_calls=44,
                client=updater,
            )
            self.assertEqual(completed["actual_calls"], 44)
            self.assertEqual(updater.generate_calls, 44)
            self.assertEqual(completed["exact_requests"], 44)
            self.assertEqual(completed["deferred_requests"], 0)
            self.assertEqual(completed["materialized_requests"], 44)
            self.assertEqual(completed["materialization_mismatches"], 0)
            final_rows = completed["context"]["requests"]
            final_by_key = {self._request_key(row): row for row in final_rows}
            for key, request_hash in exact_before.items():
                self.assertEqual(final_by_key[key]["request_hash"], request_hash)
            for key, bound in deferred_bounds.items():
                self.assertLessEqual(final_by_key[key]["estimated_input_tokens"], bound)

            sequence_dir = (
                run_dir
                / "artifacts"
                / "rubrics"
                / "sequential"
                / "block_00"
                / "random"
                / "replicate_0"
            )
            metas = [
                read_json(sequence_dir / f"step_{step:02d}.meta.json") for step in range(1, 11)
            ]
            self.assertEqual(len(metas), 10)
            for previous, current in zip(metas, metas[1:], strict=False):
                self.assertEqual(
                    current["input_rubric_hash"], previous["output_canonical_rubric_hash"]
                )
                self.assertTrue(current["materialized_before_client_call"])

            judge_plan = plan_judging(config, "phase-run")
            self.assertEqual(judge_plan["status"], "ready")
            self.assertEqual(judge_plan["exact_requests"], 48)
            self.assertEqual(judge_plan["logical_predictions"], 48)
            self.assertEqual(judge_plan["physical_requests"], 36)
            self.assertEqual(judge_plan["fresh_requests"], 36)
            judge_hashes = {
                self._judge_key(row): row["request_hash"] for row in judge_plan["requests"]
            }
            judge = AuditedMockClient(run_dir, read_json(config["rubric"]["initial"]))
            judged = judge_responses(
                config,
                "phase-run",
                execute_model_calls=True,
                max_calls=36,
                client=judge,
            )
            self.assertEqual(judged["actual_calls"], 36)
            self.assertEqual(judge.generate_calls, 36)
            self.assertEqual(judged["logical_predictions"], 48)
            self.assertEqual(judged["physical_requests"], 36)
            self.assertEqual(judged["cache_hits"], 36)
            persisted = read_jsonl(run_dir / "artifacts" / "request_plans" / "judge.jsonl")
            self.assertEqual(len(persisted), 48)
            self.assertEqual(sum(row["physical_request"] for row in persisted), 36)
            self.assertEqual(
                {self._judge_key(row): row["request_hash"] for row in persisted},
                judge_hashes,
            )

            rubric_row = read_jsonl(run_dir / "artifacts" / "rubrics.jsonl")[0]
            rubric_path = run_dir / rubric_row["rubric_path"]
            rubric_path.write_text(rubric_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            invalid = plan_judging(config, "phase-run")
            self.assertEqual(invalid["status"], "blocked_by_rubric_generation")
            self.assertEqual(invalid["invalid_requests"], 1)

    def test_sequential_resume_starts_at_first_missing_and_revalidates_cached_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, run_dir = self._prepared(Path(temporary), "resume-run")
            filters = {
                "methods": ["sequential"],
                "blocks": [0],
                "orders": ["random"],
                "replicates": [0],
            }
            first_client = AuditedMockClient(run_dir, read_json(config["rubric"]["initial"]))
            first = build_rubrics(
                config,
                "resume-run",
                execute_model_calls=True,
                max_calls=10,
                client=first_client,
                **filters,
            )
            self.assertEqual(first["actual_calls"], 10)
            sequence_dir = (
                run_dir
                / "artifacts"
                / "rubrics"
                / "sequential"
                / "block_00"
                / "random"
                / "replicate_0"
            )
            for step in range(5, 11):
                (sequence_dir / f"step_{step:02d}.json").unlink()
                (sequence_dir / f"step_{step:02d}.meta.json").unlink()
            partial = plan_rubric_generation(config, "resume-run", **filters)
            self.assertEqual(partial["exact_requests"], 5)
            self.assertEqual(partial["deferred_requests"], 5)

            resumed_client = AuditedMockClient(run_dir, read_json(config["rubric"]["initial"]))
            resumed = build_rubrics(
                config,
                "resume-run",
                execute_model_calls=True,
                max_calls=6,
                client=resumed_client,
                **filters,
            )
            self.assertEqual(resumed["actual_calls"], 0)
            self.assertEqual(resumed["execution_cache_hits"], 6)
            self.assertEqual(resumed_client.generate_calls, 0)
            self.assertEqual(resumed["exact_requests"], 10)
            self.assertEqual(resumed["deferred_requests"], 0)

            broken_meta = sequence_dir / "step_04.meta.json"
            metadata = read_json(broken_meta)
            metadata["input_rubric_hash"] = "0" * 64
            broken_meta.write_text(json.dumps(metadata), encoding="utf-8")
            chain_client = AuditedMockClient(run_dir, read_json(config["rubric"]["initial"]))
            with self.assertRaisesRegex(ValueError, "broken canonical chain"):
                build_rubrics(
                    config,
                    "resume-run",
                    execute_model_calls=True,
                    max_calls=0,
                    client=chain_client,
                    **filters,
                )
            self.assertEqual(chain_client.generate_calls, 0)

    def test_context_failure_refuses_trajectory_before_client_contact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, run_dir = self._prepared(
                Path(temporary), "context-run", safety_margin_tokens=20_000
            )
            filters = {
                "methods": ["sequential"],
                "blocks": [0],
                "orders": ["random"],
                "replicates": [0],
            }
            plan = plan_rubric_generation(config, "context-run", **filters)
            self.assertFalse(plan["ready"])
            self.assertGreater(plan["invalid_requests"], 0)
            client = AuditedMockClient(run_dir, read_json(config["rubric"]["initial"]))
            with self.assertRaisesRegex(ValueError, "preflight is not ready"):
                build_rubrics(
                    config,
                    "context-run",
                    execute_model_calls=True,
                    max_calls=10,
                    client=client,
                    **filters,
                )
            self.assertEqual(client.health_checks, 0)
            self.assertEqual(client.generate_calls, 0)

    def test_pilot_002_required_phase_counts_without_model_contact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = load_config(ROOT / "configs" / "pilot_v2.yaml")
            config["output_root"] = str(Path(temporary) / "runs")
            manifest = prepare_run(config, "count-check")
            estimate = estimate_calls(config, heldout_response_count=20)
            self.assertEqual(estimate["rubric_stage_calls"], 132)
            self.assertEqual(estimate["judge_stage_calls"], 500)
            self.assertEqual(estimate["total_inference_calls"], 632)
            updater = plan_rubric_generation(config, "count-check")
            self.assertEqual(updater["exact_requests"], 24)
            self.assertEqual(updater["deferred_requests"], 108)
            self.assertEqual(updater["planned_physical_updater_calls"], 132)
            judge = plan_judging(config, "count-check")
            self.assertEqual(judge["status"], "blocked_by_rubric_generation")
            self.assertEqual(judge["planned_calls"], 500)
            self.assertEqual(judge["logical_predictions"], 640)
            self.assertEqual(
                manifest["code"]["file_tree_sha256"],
                manifest["code"]["content_inventory_sha256"],
            )
            calls_database = Path(config["output_root"]) / "count-check" / "calls.sqlite3"
            self.assertFalse(calls_database.exists())

    @staticmethod
    def _request_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row["method"],
            row["block_id"],
            row["order"],
            row["replicate"],
            row["step"],
        )

    @staticmethod
    def _judge_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row["method"],
            row["block_id"],
            row["order"],
            row["replicate"],
            row["example_id"],
        )

    @staticmethod
    def _prepared(
        root: Path, run_id: str, *, safety_margin_tokens: int = 1024
    ) -> tuple[dict[str, Any], Path]:
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
        write_jsonl(data_path, rows)
        config = load_config(ROOT / "configs" / "pilot_v2.yaml")
        config["dataset"]["path"] = str(data_path)
        config["output_root"] = str(root / "runs")
        config["exclude_manifests"] = []
        config["sampling"].update(
            {"calibration_blocks": 1, "heldout_prompts": 2, "responses_per_prompt": 2}
        )
        config["repeat_control"]["enabled"] = False
        config["analysis"].update({"bootstrap_repetitions": 5, "make_plots": False})
        config["context"]["safety_margin_tokens"] = safety_margin_tokens
        prepare_run(config, run_id)
        return config, Path(config["output_root"]) / run_id


if __name__ == "__main__":
    unittest.main()
