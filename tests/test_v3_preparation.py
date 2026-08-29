from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from rubric_order.api import no_inference_v3_smoke_test
from rubric_order.config import load_config
from rubric_order.downstream import (
    content_inventory_hash,
    estimate_downstream_audit,
    load_manual_rubric_gate,
    plan_downstream_audit,
    run_downstream_audit,
    verify_approved_judge,
    write_downstream_dry_run,
)
from rubric_order.drift_annotations import (
    annotation_agreement,
    annotation_freeze_gate,
    cohens_kappa,
    export_blinded_drift,
    freeze_annotations,
    lexical_drift_flags,
    validate_annotation_import,
)
from rubric_order.estimate import RubricJob, Selection
from rubric_order.inventory_v3 import (
    active_inventory_ids,
    build_inventory_request,
    execute_inventory_requests,
    inventory_selection_schema,
    load_inventory,
    materialize_inventory_rubric,
    parse_inventory_selection,
    validate_inventory_selection,
    validate_materialized_inventory_rubric,
)
from rubric_order.prompts import load_template
from rubric_order.run import file_sha256, read_json, read_jsonl, write_json, write_jsonl
from rubric_order.sampling import ordered_examples
from rubric_order.v3_design import (
    estimate_main_v3,
    estimate_pilot_v3,
    plan_pilot_v3,
    plan_pilot_v3_judging,
    validate_paired_source,
)

PROJECT_ROOT = Path(__file__).parents[1].resolve()
FROZEN_RUNS = ("pilot-001", "pilot-002", "judge-validation-v2")
FUTURE_RUNS = ("pilot-002-downstream-audit", "pilot-003", "main-v3")


class SelectionClient:
    def __init__(self, selection: list[str] | None = None) -> None:
        self.selection = selection or []
        self.generate_calls = 0
        self.health_checks = 0
        self.unloads = 0

    def health_and_models(self, required: list[str]) -> dict[str, Any]:
        self.health_checks += 1
        return {"ok": True, "models": required, "missing": []}

    def generate(self, **_kwargs: Any) -> dict[str, Any]:
        self.generate_calls += 1
        return {
            "response": json.dumps({"adaptive_criterion_ids": self.selection}),
            "latency_seconds": 0.0,
            "prompt_eval_count": 20,
            "eval_count": 3,
        }

    def unload(self, _model: str) -> None:
        self.unloads += 1


class JudgeClient(SelectionClient):
    def generate(self, **_kwargs: Any) -> dict[str, Any]:
        self.generate_calls += 1
        return {
            "response": '{"score":2}',
            "latency_seconds": 0.0,
            "prompt_eval_count": 20,
            "eval_count": 2,
        }


def test_frozen_downstream_plan_is_exact_and_read_only() -> None:
    source = PROJECT_ROOT / "runs" / "pilot-002"
    before = content_inventory_hash(source)
    calls_hash = file_sha256(source / "calls.sqlite3")
    future_before = {name: (PROJECT_ROOT / "runs" / name).exists() for name in FUTURE_RUNS}
    config = load_config(PROJECT_ROOT / "configs" / "pilot_002_downstream_audit.yaml")

    approved = verify_approved_judge(config)
    plan = plan_downstream_audit(config)
    estimate = estimate_downstream_audit(config)

    assert approved == {
        "status": "JUDGE_APPROVED",
        "artifact": str(
            (PROJECT_ROOT / "runs/judge-validation-v2/judge_validation_summary.json").resolve()
        ),
        "model": "qwen3:14b",
        "candidate": "judge_anchor_first",
        "prompt_sha256": ("75dd61241b143deabe1c9e9435f897c17874ea4cc7ed0b513c43cd35687cda7c"),
        "model_options": {
            "num_ctx": 16384,
            "num_predict": 25,
            "seed": 42,
            "temperature": 0,
            "think": False,
        },
    }
    assert plan["updater_calls"] == 0
    assert plan["logical_predictions"] == 640
    assert plan["physical_requests"] == plan["fresh_requests"] == 500
    assert plan["invalid_requests"] == 0
    assert len(plan["requests"]) == 640
    assert plan["rubric_integrity"]["hashes_verified"]
    assert estimate["total_inference_calls"] == 500
    assert content_inventory_hash(source) == before
    assert file_sha256(source / "calls.sqlite3") == calls_hash
    assert {name: (PROJECT_ROOT / "runs" / name).exists() for name in FUTURE_RUNS} == future_before


def test_downstream_manual_gate_and_trajectory_replicates() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pilot_002_downstream_audit.yaml")
    plan = plan_downstream_audit(config, include_requests=False)
    repeated = next(
        row
        for row in plan["trajectory_groups"]["replicate_groups"]
        if row["trajectory_id"] == "sequential:0:random"
    )
    assert repeated["artifacts"] == 3
    assert repeated["replicates"] == [0, 1, 2]
    assert repeated["unique_canonical_rubrics"] == 1
    assert plan["trajectory_groups"]["statistical_rule"].startswith("replicates are nested")
    rows = read_jsonl(PROJECT_ROOT / "runs/pilot-002/artifacts/rubrics.jsonl")
    with pytest.raises(FileNotFoundError, match="no thematic classification was invented"):
        load_manual_rubric_gate(PROJECT_ROOT / "missing-manual-gate.json", rows)


def test_downstream_dry_run_stays_outside_frozen_and_creates_no_database() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pilot_002_downstream_audit.yaml")
    with tempfile.TemporaryDirectory() as temporary:
        config["downstream_audit"]["dry_run_root"] = temporary
        result = write_downstream_dry_run(config)
        assert result["ollama_contacted"] is False
        assert result["physical_requests"] == 500
        assert not list(Path(temporary).rglob("calls.sqlite3"))


def test_downstream_mock_canary_and_resume_use_no_updater() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pilot_002_downstream_audit.yaml")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        config["output_root"] = str(root / "runs")
        config["downstream_audit"]["dry_run_root"] = str(root / "dry-runs")
        dry_client = JudgeClient()
        dry = run_downstream_audit(
            config,
            "pilot-002-downstream-audit",
            execute_model_calls=False,
            max_calls=0,
            client=dry_client,
        )
        assert dry["actual_calls"] == dry_client.generate_calls == 0
        assert not (root / "runs/pilot-002-downstream-audit").exists()
        assert not list(root.rglob("calls.sqlite3"))

        canary_client = JudgeClient()
        canary = run_downstream_audit(
            config,
            "pilot-002-downstream-audit",
            execute_model_calls=True,
            max_calls=10,
            client=canary_client,
        )
        assert canary["status"] == "partial_budget_stop"
        assert canary["actual_calls"] == canary_client.generate_calls == 10
        resume_client = JudgeClient()
        resumed = run_downstream_audit(
            config,
            "pilot-002-downstream-audit",
            execute_model_calls=True,
            max_calls=500,
            client=resume_client,
        )
        assert resumed["status"] == "complete"
        assert resumed["logical_predictions_complete"] == 640
        assert resumed["actual_calls"] == resume_client.generate_calls == 490
        database = root / "runs/pilot-002-downstream-audit/calls.sqlite3"
        assert database.exists()
        assert not list((root / "runs").rglob("*updater*"))


def test_inventory_contract_and_server_side_materialization() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pilot_v3.yaml")
    inventory = load_inventory(config)
    initial = read_json(PROJECT_ROOT / "rubrics/initial_helpfulness_v3.json")
    assert active_inventory_ids(inventory) == [
        "actionability_and_practical_utility",
        "appropriate_level_of_detail",
        "uncertainty_and_limitation_handling",
        "supporting_justification_when_relevant",
    ]
    schema = inventory_selection_schema(inventory)
    assert schema["properties"]["adaptive_criterion_ids"]["maxItems"] == 2
    assert validate_inventory_selection({"adaptive_criterion_ids": []}, inventory) == {
        "adaptive_criterion_ids": []
    }
    canonical = validate_inventory_selection(
        {
            "adaptive_criterion_ids": [
                "supporting_justification_when_relevant",
                "actionability_and_practical_utility",
            ]
        },
        inventory,
    )
    assert canonical["adaptive_criterion_ids"] == [
        "actionability_and_practical_utility",
        "supporting_justification_when_relevant",
    ]
    for bad in (
        {"adaptive_criterion_ids": ["unknown"]},
        {"adaptive_criterion_ids": ["appropriate_level_of_detail"] * 2},
        {
            "adaptive_criterion_ids": [
                "actionability_and_practical_utility",
                "appropriate_level_of_detail",
                "uncertainty_and_limitation_handling",
            ]
        },
        {"adaptive_criterion_ids": [], "explanation": "not allowed"},
    ):
        with pytest.raises(ValueError):
            validate_inventory_selection(bad, inventory)
    with pytest.raises(ValueError):
        parse_inventory_selection(
            '{"adaptive_criterion_ids": ["completeness_and_coverage"]}', inventory
        )
    first = materialize_inventory_rubric(initial, inventory, canonical)
    second = materialize_inventory_rubric(initial, inventory, canonical)
    assert _canonical_bytes(first) == _canonical_bytes(second)
    assert first["title"] == initial["title"]
    assert first["criteria"][:3] == initial["criteria"][:3]
    assert first["score_anchors"] == initial["score_anchors"]
    tampered = copy.deepcopy(first)
    tampered["criteria"][3]["description"] = "Model-written replacement"
    with pytest.raises(ValueError, match="server-side"):
        validate_materialized_inventory_rubric(tampered, initial, inventory)


def test_inventory_batch_sequential_hash_chain_and_mock_execution_parity() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pilot_v3.yaml")
    source = validate_paired_source(config)
    inventory = load_inventory(config)
    initial = read_json(PROJECT_ROOT / "rubrics/initial_helpfulness_v3.json")
    batch_job = RubricJob("batch", 0, "ascending", 0, False, 10)
    sequential_job = RubricJob("sequential", 0, "ascending", 0, False, 10)
    batch = build_inventory_request(
        config,
        run_id="hash-parity",
        job=batch_job,
        step=1,
        state={"adaptive_criterion_ids": []},
        examples=ordered_examples(source["blocks"][0], "ascending"),
        template=load_template(config, "batch"),
        inventory=inventory,
        initial=initial,
    )
    first = build_inventory_request(
        config,
        run_id="hash-parity",
        job=sequential_job,
        step=1,
        state={"adaptive_criterion_ids": []},
        examples=[ordered_examples(source["blocks"][0], "ascending")[0]],
        template=load_template(config, "sequential"),
        inventory=inventory,
        initial=initial,
    )
    next_state = {"adaptive_criterion_ids": ["appropriate_level_of_detail"]}
    second = build_inventory_request(
        config,
        run_id="hash-parity",
        job=sequential_job,
        step=2,
        state=next_state,
        examples=[ordered_examples(source["blocks"][0], "ascending")[1]],
        template=load_template(config, "sequential"),
        inventory=inventory,
        initial=initial,
    )
    assert batch["base_method"] == "batch" and len(batch["example_ids"]) == 10
    assert first["base_method"] == "sequential" and len(first["example_ids"]) == 1
    assert second["previous_state_hash"] == hashlib.sha256(_canonical_bytes(next_state)).hexdigest()
    assert all(
        request[field]
        for request in (batch, first, second)
        for field in (
            "request_hash",
            "prompt_template_hash",
            "schema_hash",
            "model_options_hash",
            "context_preflight",
        )
    )

    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary) / "mock-pilot"
        write_json(
            run_dir / "manifest.json",
            {"run_id": "hash-parity", "model_calls_made": 0},
        )
        client = SelectionClient([])
        result = execute_inventory_requests(
            config,
            run_id="hash-parity",
            run_dir=run_dir,
            blocks=source["blocks"],
            max_calls=1,
            selection=Selection(
                blocks=frozenset({0}),
                orders=frozenset({"ascending"}),
                methods=frozenset({"batch"}),
                replicates=frozenset({0}),
            ),
            client=client,
        )
        stored = read_jsonl(run_dir / "artifacts/request_plans/updater.jsonl")
        assert result["actual_calls"] == client.generate_calls == 1
        assert stored[0]["request_hash"] == batch["request_hash"]
        assert result["request_hash_parity_enforced"]


def test_pilot_v3_reuses_exact_source_and_estimator() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pilot_v3.yaml")
    source = validate_paired_source(config)
    estimate = estimate_pilot_v3(config)
    plan = plan_pilot_v3(config)
    assert source["checks"]["hashes_match"]
    assert source["checks"]["orders_match"]
    assert estimate["updater_calls"] == 132
    assert estimate["judge_calls"] == 500
    assert estimate["total_inference_calls"] == 632
    assert plan["updater"]["exact_requests"] == 24
    assert plan["updater"]["deferred_requests"] == 108
    assert plan["ollama_contacted"] is False
    judge_plan = plan_pilot_v3_judging(config)
    assert judge_plan["status"] == "blocked_by_constrained_rubric_generation"
    assert judge_plan["physical_requests"] == 500
    assert judge_plan["ollama_contacted"] is False
    assert not (PROJECT_ROOT / "runs/pilot-003").exists()

    mismatch = copy.deepcopy(config)
    mismatch["paired_source"]["sha256"]["heldout"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_paired_source(mismatch)


def test_main_v3_design_is_frozen_to_exact_ablation() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "main_v3.yaml")
    estimate = estimate_main_v3(config)
    ablation = estimate["constrained_ablation"]
    assert estimate["original_main_v2_calls"] == 5156
    assert ablation["blocks"] == [0, 4, 9]
    assert ablation["orders"] == config["orders"]
    assert ablation["repeat_control"] is False
    assert ablation["updater_calls"] == 132
    assert ablation["judge_calls"] == 1200
    assert ablation["total_calls"] == 1332
    assert estimate["total_inference_calls"] == 6488
    assert config["models"]["judge"] == "qwen3:14b"
    excluded = {row["path"] for row in config["exclude_manifests"]}
    assert {
        "runs/pilot-001/manifest.json",
        "runs/judge-validation-v2/manifest.json",
        "runs/pilot-002/manifest.json",
        "runs/pilot-003/manifest.json",
    } <= excluded
    assert not (PROJECT_ROOT / "runs/main-v3").exists()


def test_blinded_annotations_validation_agreement_and_freeze() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        exported = export_blinded_drift(PROJECT_ROOT / "runs/pilot-002", root / "annotation-export")
        blinded = read_jsonl(exported["blinded_export"])
        assert blinded
        assert all(not ({"method", "block_id", "order", "replicate"} & set(row)) for row in blinded)
        first_ids = [row["blind_id"] for row in blinded]
        planned_again = export_blinded_drift(
            PROJECT_ROOT / "runs/pilot-002", root / "annotation-export-2"
        )
        assert first_ids == [row["blind_id"] for row in read_jsonl(planned_again["blinded_export"])]
        rows = [_annotation_row(blind_id) for blind_id in first_ids]
        left = root / "annotator-1.jsonl"
        right = root / "annotator-2.jsonl"
        write_jsonl(left, rows)
        write_jsonl(right, rows)
        validated = validate_annotation_import(
            left,
            exported["mapping"],
            expected_mapping_sha256=exported["mapping_sha256"],
        )
        assert validated["annotation_count"] == len(rows)
        agreement = annotation_agreement([left, right], exported["mapping"])
        assert agreement["disagreement_count"] == 0
        assert set(agreement["cohens_kappa"].values()) == {1.0}
        one = annotation_agreement([left], exported["mapping"])
        assert "Only one annotator" in one["methodological_warning"]
        freeze_path = root / "freeze.json"
        frozen = freeze_annotations([left, right], exported["mapping"], freeze_path)
        gate = annotation_freeze_gate(
            freeze_path, judging_started_at="2999-01-01T00:00:00+00:00", minimum_annotators=2
        )
        assert frozen["automated_flags_are_ground_truth"] is False
        assert gate["ready"]

        unknown = copy.deepcopy(rows)
        unknown[0]["blind_id"] = "DRIFT-unknown"
        unknown_path = root / "unknown.jsonl"
        write_jsonl(unknown_path, unknown)
        with pytest.raises(ValueError, match="unknown blind_id"):
            validate_annotation_import(unknown_path, exported["mapping"])
        duplicate_path = root / "duplicate.jsonl"
        write_jsonl(duplicate_path, rows + [rows[0]])
        with pytest.raises(ValueError, match="duplicate blind_id"):
            validate_annotation_import(duplicate_path, exported["mapping"])
        invalid = copy.deepcopy(rows)
        invalid[0]["task_specificity"] = 3
        invalid_path = root / "invalid.jsonl"
        write_jsonl(invalid_path, invalid)
        with pytest.raises(ValueError, match="invalid task_specificity"):
            validate_annotation_import(invalid_path, exported["mapping"])


def test_lexical_flags_are_only_heuristics_and_kappa_is_correct() -> None:
    rubric = read_json(
        PROJECT_ROOT
        / "runs/pilot-002/artifacts/rubrics/sequential/block_00/random/replicate_0/step_10.json"
    )
    flags = lexical_drift_flags(
        rubric,
        [{"prompt": "extract information", "response": "the extracted information"}],
    )
    assert flags["heuristic_only"]
    assert flags["semantic_validator"] is False
    assert flags["ground_truth"] is False
    assert flags["human_annotation_required"]
    assert cohens_kappa([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0


def test_v3_synthetic_smoke_has_zero_model_contact_and_no_database() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "smoke-v3"
        result = no_inference_v3_smoke_test(output)
        assert result["status"] == "ok"
        assert result["planned_updater_calls"] == 44
        assert result["invalid_requests"] == 0
        assert result["ollama_contacted"] is False
        assert result["sqlite_call_store_created"] is False
        assert not list(output.rglob("calls.sqlite3"))


def test_frozen_run_inventories_match_preparation_snapshot() -> None:
    expected = {
        "pilot-001": (112, "26774e56841209b456a1eae311c4e52516154cf69d0d07643fee8d17cf7a5d42"),
        "pilot-002": (282, "2075fb4d4ff3f843b2ab104515af2077cdae95176af87fb50b2a58538c23a587"),
        "judge-validation-v2": (
            14,
            "c6df5f17ec6958b3f25c1219805d09eb22cf8bf1f4f401fcf79094b57c2c6fad",
        ),
    }
    for run_id in FROZEN_RUNS:
        inventory = content_inventory_hash(PROJECT_ROOT / "runs" / run_id)
        assert (inventory["file_count"], inventory["sha256"]) == expected[run_id]


def _annotation_row(blind_id: str) -> dict[str, Any]:
    return {
        "blind_id": blind_id,
        "domain_general": "yes",
        "task_specificity": 0,
        "copies_example_terms": "no",
        "duplicate_or_near_duplicate": "no",
        "short_rationale": "The criterion is phrased for unrelated requests.",
    }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
