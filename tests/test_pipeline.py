from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from rubric_order.api import plan_rubric_generation
from rubric_order.config import DEFAULTS
from rubric_order.estimate import Selection
from rubric_order.judging import judge_rubrics
from rubric_order.metrics import analyze_run
from rubric_order.readiness import readiness_report
from rubric_order.rubric_generation import build_rubrics
from rubric_order.run import prepare_run, write_jsonl
from tests.test_schemas_storage import RUBRIC


class MockOllama:
    def __init__(self) -> None:
        self.generate_calls = 0
        self.unloaded: list[str] = []

    def health_and_models(self, required_models: list[str]) -> dict[str, Any]:
        return {"ok": True, "models": required_models, "missing": []}

    def generate(self, **kwargs: Any) -> dict[str, Any]:
        self.generate_calls += 1
        schema = kwargs["response_schema"]
        if "score" in schema.get("properties", {}):
            raw = json.dumps({"score": self.generate_calls % 5})
        else:
            raw = json.dumps(RUBRIC)
        return {
            "response": raw,
            "latency_seconds": 0.001,
            "prompt_eval_count": 10,
            "eval_count": 2,
        }

    def unload(self, model: str) -> None:
        self.unloaded.append(model)


class PipelineTests(unittest.TestCase):
    def test_mocked_end_to_end_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "data.jsonl"
            rows = [
                {
                    "prompt": f"calibration-{score}-{index}",
                    "response": f"response-{score}-{index}",
                    "helpfulness": score,
                }
                for score in range(5)
                for index in range(6)
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
            config = copy.deepcopy(DEFAULTS)
            config["dataset"]["path"] = str(data_path)
            config["sampling"].update(
                {"calibration_blocks": 1, "heldout_prompts": 3, "responses_per_prompt": 2}
            )
            config["analysis"].update({"bootstrap_repetitions": 5, "make_plots": False})
            config["output_root"] = str(root / "runs")
            config["_config_path"] = "<mocked-test>"
            config["_project_root"] = str(Path(__file__).parents[1].resolve())
            prepare_run(config, "run")

            updater = MockOllama()
            build_result = build_rubrics(config, "run", Selection(), max_calls=100, client=updater)
            self.assertGreater(build_result["actual_calls"], 0)
            self.assertEqual(updater.unloaded, [config["models"]["updater"]])

            judge = MockOllama()
            judge_result = judge_rubrics(config, "run", Selection(), max_calls=100, client=judge)
            self.assertGreater(judge_result["actual_calls"], 0)
            self.assertEqual(judge.unloaded, [config["models"]["judge"]])

            summary = analyze_run(config, "run")
            self.assertEqual(set(summary["primary"]), {"batch", "sequential", "static"})
            self.assertTrue((root / "runs" / "run" / "metrics" / "summary.json").exists())

    def test_v2_synthetic_preparation_and_context_dry_run_make_no_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "data.jsonl"
            rows = [
                {
                    "prompt": f"calibration-{score}-{index}",
                    "response": f"response-{score}-{index}",
                    "helpfulness": score,
                }
                for score in range(5)
                for index in range(4)
            ]
            rows.extend(
                {
                    "prompt": f"paired-{prompt_index}",
                    "response": f"choice-{response_index}",
                    "helpfulness": (prompt_index + response_index) % 5,
                }
                for prompt_index in range(3)
                for response_index in range(2)
            )
            write_jsonl(data_path, rows)
            config = copy.deepcopy(DEFAULTS)
            config.update(
                {
                    "schema_version": "2.0",
                    "prompt_version": "2.0",
                    "exclude_manifests": [],
                    "approved_judge_artifact": "missing-approval.json",
                    "output_root": str(root / "runs"),
                    "_config_path": "<synthetic-v2>",
                    "_project_root": str(Path(__file__).parents[1].resolve()),
                }
            )
            config["dataset"]["path"] = str(data_path)
            config["sampling"].update({"calibration_blocks": 1, "heldout_prompts": 2})
            config["models"]["context_length"] = 16384
            config["rubric"]["initial"] = "rubrics/initial_helpfulness_v2.json"
            config["prompts"].update(
                {
                    "sequential": "prompts/update_sequential_v2.txt",
                    "batch": "prompts/induce_batch_v2.txt",
                    "judge": "prompts/judge_absolute_v2.txt",
                }
            )
            manifest = prepare_run(config, "dry-v2")
            plan = plan_rubric_generation(
                config,
                "dry-v2",
                methods=["batch"],
                blocks=[0],
                orders=["random"],
                replicates=[0],
            )
            run_dir = root / "runs" / "dry-v2"
            self.assertEqual(manifest["schema_version"], "2.0")
            self.assertEqual(plan["planned_calls"], 1)
            self.assertEqual(len(plan["context"]["calls"]), 1)
            self.assertFalse(plan["context"]["refused_calls"])
            self.assertFalse((run_dir / "calls.sqlite3").exists())
            readiness = readiness_report(config, "dry-v2", write_report=False)
            self.assertEqual(readiness["verdict"], "NOT_READY_FOR_MAIN")
            self.assertFalse((run_dir / "calls.sqlite3").exists())


if __name__ == "__main__":
    unittest.main()
