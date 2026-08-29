from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from rubric_order.config import load_config
from rubric_order.judge_validation import (
    DEV_STAGE,
    _reload_subset_rows,
    _run_subset,
    build_judge_request,
    plan_judge_requests,
    plan_judge_validation,
    run_judge_dev,
    run_judge_holdout,
)
from rubric_order.prompts import PromptTemplate, load_template
from rubric_order.run import read_json, read_jsonl, write_jsonl
from rubric_order.schemas import JUDGE_JSON_SCHEMA, validate_rubric_v2
from rubric_order.storage import CallContext, CallStore

PROJECT_ROOT = Path(__file__).parents[1].resolve()


class RecordingClient:
    def __init__(self) -> None:
        self.generated: list[dict[str, Any]] = []
        self.health_checks = 0
        self.unloads = 0

    def health_and_models(self, required: list[str]) -> dict[str, Any]:
        self.health_checks += 1
        return {"ok": True, "models": list(required), "missing": []}

    def generate(self, **kwargs: Any) -> dict[str, Any]:
        self.generated.append(kwargs)
        return {
            "response": '{"score":2}',
            "latency_seconds": 0.0,
            "prompt_eval_count": 10,
            "eval_count": 2,
        }

    def unload(self, _model: str) -> None:
        self.unloads += 1


class ExplodingClient:
    def __init__(self) -> None:
        self.contacts = 0

    def __getattr__(self, _name: str) -> Any:
        self.contacts += 1
        raise AssertionError("dry-run contacted a model client")


class JudgePreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(PROJECT_ROOT / "configs" / "judge_validation.yaml")
        initial = read_json(PROJECT_ROOT / "rubrics" / "initial_helpfulness_v2.json")
        self.rubric = validate_rubric_v2(initial, initial)
        self.example = {
            "example_id": "example-1",
            "prompt_id": "prompt-1",
            "prompt": "Explain why the sky appears blue.",
            "response": "Short wavelengths scatter more strongly in the atmosphere.",
            "human_score": 4,
        }

    def test_request_hashes_are_deterministic_and_semantic(self) -> None:
        template = load_template(self.config, "judge_absolute")
        first = self._build(self.config, template, run_id="one")
        second = self._build(self.config, template, run_id="two")
        timestamp_config = copy.deepcopy(self.config)
        timestamp_config["created_at"] = "2099-01-01T00:00:00Z"
        timestamped = self._build(timestamp_config, template, run_id="three")

        keys = (
            "request_hash",
            "prompt_template_hash",
            "rubric_hash",
            "schema_hash",
            "model_options_hash",
        )
        self.assertEqual({key: first[key] for key in keys}, {key: second[key] for key in keys})
        self.assertEqual(first["request_hash"], timestamped["request_hash"])

        changed_template = PromptTemplate(
            name=template.name,
            text=template.text + "\n",
            sha256=_sha256(template.text + "\n"),
        )
        changed_prompt = self._build(self.config, changed_template)
        self.assertNotEqual(first["prompt_template_hash"], changed_prompt["prompt_template_hash"])
        self.assertNotEqual(first["request_hash"], changed_prompt["request_hash"])

        changed_model = self._build(self.config, template, model_name="different:model")
        self.assertNotEqual(first["request_hash"], changed_model["request_hash"])

        for field, value in (
            ("temperature", 0.25),
            ("seed", 99),
            ("context_length", 8192),
            ("judge_max_tokens", 30),
            ("think", True),
        ):
            changed_config = copy.deepcopy(self.config)
            changed_config["models"][field] = value
            changed = self._build(changed_config, template)
            with self.subTest(model_option=field):
                self.assertNotEqual(first["model_options_hash"], changed["model_options_hash"])
                self.assertNotEqual(first["request_hash"], changed["request_hash"])

        changed_schema = copy.deepcopy(JUDGE_JSON_SCHEMA)
        changed_schema["description"] = "same validator, different declared schema"
        schema_request = self._build(self.config, template, response_schema=changed_schema)
        self.assertNotEqual(first["schema_hash"], schema_request["schema_hash"])
        self.assertNotEqual(first["request_hash"], schema_request["request_hash"])

        other_candidate = self._build(
            self.config,
            load_template(self.config, "judge_anchor_first"),
            candidate="judge_anchor_first",
        )
        self.assertNotEqual(first["request_hash"], other_candidate["request_hash"])

    def test_context_metadata_and_invalid_request_gate(self) -> None:
        config = copy.deepcopy(self.config)
        config["context"]["safety_margin_tokens"] = 0
        template = PromptTemplate("boundary", "{prompt_json}{response_json}", _sha256("x"))
        exact_example = {**self.example, "prompt": "x" * 49_072, "response": "r"}
        exact = self._build(config, template, example=exact_example, candidate="boundary")
        self.assertEqual(exact["estimated_context_headroom"], 0)
        self.assertTrue(exact["context_preflight_ok"])
        self.assertFalse(exact["token_estimate_exact"])
        self.assertEqual(exact["token_estimator"], "conservative_chars_div_3_v1")
        self.assertEqual(
            exact["estimated_context_headroom"],
            exact["context_length"]
            - exact["estimated_input_tokens"]
            - exact["reserved_output_tokens"]
            - exact["safety_margin_tokens"],
        )

        over = self._build(
            config,
            template,
            example={**exact_example, "prompt": exact_example["prompt"] + "x"},
            candidate="boundary",
        )
        self.assertFalse(over["context_preflight_ok"])
        self.assertEqual(over["estimated_context_headroom"], -1)
        self.assertIsNotNone(over["context_failure_reason"])

        rows = [self.example, {**self.example, "example_id": "too-large", "response": "x" * 60_000}]
        plan = plan_judge_requests(
            config,
            run_id="context-test",
            stage=DEV_STAGE,
            candidates=["judge_absolute"],
            model_name=config["models"]["judge"],
            rows=rows,
        )
        self.assertFalse(plan["ready"])
        self.assertEqual(plan["invalid_requests"], 1)

    def test_counts_and_read_only_cache_lookup(self) -> None:
        rows = [
            {
                **self.example,
                "example_id": f"example-{index}",
                "prompt_id": f"prompt-{index}",
                "prompt": f"prompt {index}",
            }
            for index in range(5)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "calls.sqlite3"
            fresh = plan_judge_requests(
                self.config,
                run_id="counts",
                stage=DEV_STAGE,
                candidates=["judge_absolute", "judge_anchor_first"],
                model_name=self.config["models"]["judge"],
                rows=rows,
                cache_database=database,
            )
            self.assertEqual(fresh["logical_calls"], 10)
            self.assertEqual(fresh["fresh_calls"], 10)
            self.assertEqual(fresh["cache_hits"], 0)
            self.assertFalse(database.exists())

            with CallStore(database) as store:
                for index, request in enumerate(fresh["requests"][:3]):
                    store.record(
                        context=CallContext("counts", DEV_STAGE, "judge", None, None, None, 0),
                        attempt=0,
                        model_name=self.config["models"]["judge"],
                        model_options={},
                        request_hash=request["request_hash"],
                        template_hash=request["prompt_template_hash"],
                        raw_response='{ "score": 2 }',
                        parsed_response={"score": 2},
                        status="success",
                        validation_error=None,
                        latency_seconds=float(index),
                        prompt_eval_count=10,
                        eval_count=2,
                    )
            cached = plan_judge_requests(
                self.config,
                run_id="counts",
                stage=DEV_STAGE,
                candidates=["judge_absolute", "judge_anchor_first"],
                model_name=self.config["models"]["judge"],
                rows=rows,
                cache_database=database,
            )
            self.assertEqual(cached["logical_calls"], 10)
            self.assertEqual(cached["fresh_calls"], 7)
            self.assertEqual(cached["cache_hits"], 3)

            invalid_database = Path(temporary) / "invalid.sqlite3"
            invalid_database.write_text("not a SQLite database", encoding="utf-8")
            unavailable = plan_judge_requests(
                self.config,
                run_id="counts",
                stage=DEV_STAGE,
                candidates=["judge_absolute", "judge_anchor_first"],
                model_name=self.config["models"]["judge"],
                rows=rows,
                cache_database=invalid_database,
            )
            self.assertEqual(unavailable["cache_unavailable"], 10)
            self.assertFalse(unavailable["ready"])

    def test_dry_run_and_mock_execution_use_the_same_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._synthetic_config(root)
            from rubric_order.judge_validation import prepare_judge_validation

            prepare_judge_validation(config, "parity")
            run_dir = Path(config["output_root"]) / "parity"
            subset_path = run_dir / "subsets" / "judge_dev.jsonl"
            subset = read_jsonl(subset_path)[:1]
            write_jsonl(subset_path, subset)
            row = _reload_subset_rows(config, subset)[0]
            template = load_template(config, "judge_absolute")
            expected = build_judge_request(
                config,
                run_id="parity",
                stage=DEV_STAGE,
                candidate="judge_absolute",
                model_name=config["models"]["judge"],
                rubric=self.rubric,
                template=template,
                example=row,
            )
            dry_plan = plan_judge_requests(
                config,
                run_id="parity",
                stage=DEV_STAGE,
                candidates=["judge_absolute"],
                model_name=config["models"]["judge"],
                rows=[row],
                cache_database=run_dir / "calls.sqlite3",
            )
            self.assertEqual(dry_plan["requests"][0]["request_hash"], expected["request_hash"])

            first_client = RecordingClient()
            first = _run_subset(
                config,
                "parity",
                subset_name="judge_dev",
                stage=DEV_STAGE,
                candidates=["judge_absolute"],
                model_name=config["models"]["judge"],
                max_calls=1,
                client=first_client,
            )
            self.assertEqual(first["actual_calls"], 1)
            self.assertEqual(first_client.generated[0]["prompt"], expected["rendered_prompt"])
            self.assertEqual(first_client.generated[0]["options"], expected["model_options"])
            self.assertEqual(
                first_client.generated[0]["response_schema"], expected["response_schema"]
            )
            prediction = read_jsonl(run_dir / "artifacts" / "judge_validation_predictions.jsonl")[0]
            self.assertEqual(prediction["request_hash"], expected["request_hash"])

            (run_dir / "artifacts" / "judge_validation_predictions.jsonl").unlink()
            second_client = RecordingClient()
            second = _run_subset(
                config,
                "parity",
                subset_name="judge_dev",
                stage=DEV_STAGE,
                candidates=["judge_absolute"],
                model_name=config["models"]["judge"],
                max_calls=1,
                client=second_client,
            )
            self.assertEqual(second["actual_calls"], 0)
            self.assertEqual(second["cache_hits"], 1)
            self.assertEqual(second_client.generated, [])

    def test_synthetic_dry_run_has_no_run_database_or_client_contact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._synthetic_config(root)
            client = ExplodingClient()
            result = run_judge_dev(
                config,
                "safe-dry-run",
                execute_model_calls=False,
                max_calls=0,
                client=client,
            )
            run_dir = Path(config["output_root"]) / "safe-dry-run"
            self.assertTrue(result["ready"])
            self.assertEqual(result["logical_calls"], 10)
            self.assertEqual(result["fresh_calls"], 10)
            self.assertEqual(result["cache_hits"], 0)
            self.assertEqual(result["invalid_requests"], 0)
            self.assertEqual(result["actual_calls"], 0)
            self.assertEqual(client.contacts, 0)
            self.assertFalse(run_dir.exists())
            self.assertFalse((run_dir / "calls.sqlite3").exists())
            with self.assertRaises(FileNotFoundError):
                run_judge_holdout(config, "safe-dry-run", execute_model_calls=False)
            self.assertFalse(run_dir.exists())
            manifest = read_json(result["dry_run_manifest"])
            self.assertEqual(manifest["model_responses"], 0)
            self.assertEqual(len(manifest["requests"]), 10)
            forbidden = {"rendered_prompt", "prompt", "response", "human_score"}
            self.assertTrue(
                all(not (forbidden & request.keys()) for request in manifest["requests"])
            )

    def test_real_configuration_plans_one_hundred_valid_requests(self) -> None:
        run_dir = PROJECT_ROOT / "runs" / "judge-validation-v2"
        before = {
            path.relative_to(run_dir).as_posix(): path.read_bytes()
            for path in run_dir.rglob("*")
            if path.is_file()
        }
        plan = plan_judge_validation(self.config, "judge-validation-v2")
        self.assertEqual(plan["logical_calls"], 100)
        self.assertEqual(plan["invalid_requests"], 0)
        self.assertTrue(plan["ready"])
        forbidden = {"rendered_prompt", "prompt", "response", "human_score"}
        self.assertTrue(all(not (forbidden & row.keys()) for row in plan["requests"]))
        after = {
            path.relative_to(run_dir).as_posix(): path.read_bytes()
            for path in run_dir.rglob("*")
            if path.is_file()
        }
        self.assertEqual(after, before)

    def _build(
        self,
        config: dict[str, Any],
        template: PromptTemplate,
        *,
        run_id: str = "run",
        candidate: str = "judge_absolute",
        model_name: str | None = None,
        example: dict[str, Any] | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return build_judge_request(
            config,
            run_id=run_id,
            stage=DEV_STAGE,
            candidate=candidate,
            model_name=model_name or config["models"]["judge"],
            rubric=self.rubric,
            template=template,
            example=example or self.example,
            response_schema=response_schema,
        )

    @staticmethod
    def _synthetic_config(root: Path) -> dict[str, Any]:
        data_dir = root / "data"
        data_dir.mkdir()
        train = [{"prompt": "train prompt", "response": "train response", "helpfulness": 2}]
        validation = [
            {
                "prompt": f"validation prompt {score} {index}",
                "response": f"validation response {score} {index}",
                "helpfulness": score,
            }
            for score in range(5)
            for index in range(2)
        ]
        (data_dir / "train.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in train), encoding="utf-8"
        )
        (data_dir / "validation.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in validation), encoding="utf-8"
        )
        config = load_config(PROJECT_ROOT / "configs" / "judge_validation.yaml")
        config["rubric"]["initial"] = str(PROJECT_ROOT / "rubrics" / "initial_helpfulness_v2.json")
        config["prompts"] = {
            key: str(PROJECT_ROOT / path) for key, path in config["prompts"].items()
        }
        config["dataset"]["path"] = str(data_dir)
        config["output_root"] = str(root / "runs")
        config["exclude_manifests"] = []
        config["judge_validation"]["dev_per_score"] = 1
        config["judge_validation"]["validation_per_score"] = 1
        config["_project_root"] = str(root)
        return config


def _sha256(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    unittest.main()
