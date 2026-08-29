from __future__ import annotations

import ast
import copy
import json
import tempfile
from pathlib import Path

import nbformat

from rubric_order.api import plan_judging, prepare_experiment
from rubric_order.config import load_config
from rubric_order.downstream import APPROVED_JUDGE, audit_completed_downstream
from rubric_order.final_primary import (
    estimate_final_primary,
    pre_main_readiness,
)
from rubric_order.lineage import audit_extreme_pair_lineage, resolve_dataset_lineage

PROJECT_ROOT = Path(__file__).parents[1].resolve()
DOWNSTREAM_RUN = PROJECT_ROOT / "runs" / "pilot-002-downstream-audit"
PREDICTIONS = DOWNSTREAM_RUN / "artifacts" / "predictions.jsonl"
LINEAGE_REPORT = DOWNSTREAM_RUN / "audit" / "extreme_pair_lineage_audit.json"


def test_exact_extreme_ids_helpfulness_pairing_and_read_only_resolver() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pilot_002_downstream_audit.yaml")
    expected = {
        "652ebe3ffd79d4a39150": ("510c8153c2a8c48a8764", 716, 1, 4),
        "f0524971ff1119f63fad": ("510c8153c2a8c48a8764", 717, 0, 4),
        "075fed6ba83133a3c8e8": ("77f54e4522f88c3a1802", 887, 4, 0),
        "7290e79be42da9b49544": ("77f54e4522f88c3a1802", 886, 2, 0),
    }
    source_mtimes = {
        path: path.stat().st_mtime_ns
        for path in (
            PROJECT_ROOT / "HelpSteer2" / "validation.jsonl",
            PROJECT_ROOT / "runs" / "pilot-002" / "prepared" / "heldout.jsonl",
            PREDICTIONS,
        )
    }
    for example_id, (prompt_id, row_index, helpfulness, predicted) in expected.items():
        report = resolve_dataset_lineage(
            config,
            example_id=example_id,
            predictions_path=PREDICTIONS,
        )
        assert report["match_count"] == 1
        row = report["records"][0]
        assert row["recalculated_example_id"] == example_id
        assert row["prompt_id"] == row["recalculated_prompt_id"] == prompt_id
        assert row["source_split"] == "validation"
        assert row["source_row_index"] == row_index
        assert row["original_helpfulness"] == helpfulness
        assert row["prediction_human_score"] == [helpfulness]
        assert row["prediction_summary"]["predicted_scores"] == [predicted]
        assert row["checks"]["responses_not_swapped"]
        assert row["checks"]["request_hashes_reconstructed_match"]
        assert row["checks"]["successful_call_prompt_hashes_match"]
        assert not row["text_processing"]["prompt_truncated"]
        assert not row["text_processing"]["response_truncated"]
    assert source_mtimes == {path: path.stat().st_mtime_ns for path in source_mtimes}


def test_extreme_pair_gate_reproduces_all_four_ids() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pilot_002_downstream_audit.yaml")
    report = audit_extreme_pair_lineage(
        config,
        prompt_ids=["510c8153c2a8c48a8764", "77f54e4522f88c3a1802"],
        example_ids=[
            "652ebe3ffd79d4a39150",
            "f0524971ff1119f63fad",
            "075fed6ba83133a3c8e8",
            "7290e79be42da9b49544",
        ],
        predictions_path=PREDICTIONS,
    )
    assert report["verdict"] == "DATA_LINEAGE_APPROVED"
    assert not report["failures"]
    assert all(all(row["checks"].values()) for row in report["records"])


def test_downstream_independent_closure_and_no_pseudoreplication() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pilot_002_downstream_audit.yaml")
    report = audit_completed_downstream(
        config,
        DOWNSTREAM_RUN,
        lineage_report=LINEAGE_REPORT,
    )
    assert report["verdict"] == "DOWNSTREAM_AUDIT_APPROVED"
    assert report["counts"] == {
        "logical_predictions": 640,
        "physical_unique_request_hashes": 500,
        "examples": 20,
        "prompts": 10,
        "canonical_rubrics": 10,
        "new_calls_in_full_resume": 490,
        "canary_cache_hits": 10,
        "static_logical_reuses": 140,
        "logical_cache_hits": 150,
    }
    vectors = report["score_vectors"]
    assert vectors["unique_score_vector_count"] == 2
    assert vectors["changed_example_id"] == "b9b488c7992d2585ba45"
    assert vectors["score_change"] == [4, 3]
    assert sorted(row["rubric_count"] for row in vectors["vector_groups"]) == [4, 6]
    assert len(report["results_by_artifact"]) == 32
    assert len(report["results_by_canonical_rubric"]) == 10
    assert len(report["results_by_trajectory"]) == 24
    assert sum(row["prediction_count"] for row in report["results_by_trajectory"]) == 480
    assert all(row["pairwise_ranking_flip_rate"] == 0 for row in report["primary"])
    assert all(row["repeat_score_change_rate"] == 0 for row in report["repeat_variation"])


def test_final_primary_exact_estimator_methods_and_approved_judge() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "main_final_primary.yaml")
    estimate = estimate_final_primary(config)
    assert estimate["physical_calls"] == 5156
    assert estimate["breakdown"]["rubric_stage_calls"] == 506
    assert estimate["breakdown"]["judge_stage_calls"] == 4650
    assert estimate["constrained_calls"] == 0
    assert config["methods"] == ["static", "batch", "sequential"]
    assert "inventory_v3" not in config
    assert "constrained_ablation" not in config
    assert config["models"]["judge"] == APPROVED_JUDGE["model"]
    assert config["prompts"]["judge"] == "prompts/judge_anchor_first_v2.txt"
    assert config["paper_design"]["decision_timing"] == "before main execution"
    assert config["drift_annotations"]["required_before_judging"] is False
    assert config["rubric"]["max_validation_retries"] == 1
    assert config["execution_safety"]["hard_total_call_ceiling"] == 5200
    assert sum(config["execution_safety"]["stage_call_ceilings"].values()) == 5200


def test_final_primary_annotation_gate_is_disabled_before_judging() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        dataset = root / "synthetic.jsonl"
        rows = [
            {
                "prompt": f"prompt-{score}-{index}",
                "response": f"response-{score}-{index}",
                "helpfulness": score,
            }
            for score in range(5)
            for index in range(8)
        ]
        dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        config = copy.deepcopy(load_config(PROJECT_ROOT / "configs" / "main_final_primary.yaml"))
        config["dataset"]["path"] = str(dataset)
        config["output_root"] = str(root / "runs")
        config["sampling"].update(
            {"calibration_blocks": 1, "heldout_prompts": 2, "responses_per_prompt": 1}
        )
        config["repeat_control"] = {
            "enabled": False,
            "blocks": [],
            "orders": ["random"],
            "methods": ["batch", "sequential"],
            "replicates": [1, 2],
        }
        prepare_experiment(config, "annotation-gate-disabled")
        plan = plan_judging(config, "annotation-gate-disabled")
        assert plan["status"] == "blocked_by_rubric_generation"
        assert not plan["ready"]
        assert plan["requests"] == []
        assert plan["annotation_freeze"] == {"ready": True, "required": False, "failures": []}
        assert not (root / "runs" / "annotation-gate-disabled" / "calls.sqlite3").exists()


def test_final_notebook_safe_defaults_and_fifteen_stages() -> None:
    notebook = nbformat.read(
        PROJECT_ROOT / "notebooks" / "08_final_main_experiment.ipynb", as_version=4
    )
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assignments = {}
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    assignments[target.id] = ast.literal_eval(node.value)
                except (TypeError, ValueError):
                    pass
    assert assignments["EXECUTE_MODEL_CALLS"] is False
    assert assignments["MAX_CALLS"] == 0
    headings = [
        cell.source.splitlines()[0]
        for cell in notebook.cells
        if cell.cell_type == "markdown" and cell.source.startswith("## ")
    ]
    assert [heading.split(".")[0] for heading in headings[:15]] == [
        f"## {number}" for number in range(1, 16)
    ]
    assert assignments["UPDATER_CANARY_METHODS"] == ["batch"]
    assert assignments["UPDATER_CANARY_BLOCKS"] == [0]
    assert assignments["UPDATER_CANARY_ORDERS"] == ["ascending"]
    assert assignments["UPDATER_CANARY_REPLICATES"] == [0]
    assert assignments["UPDATER_CANARY_MAX_CALLS"] == 1
    assert assignments["RUN_UPDATER_CANARY"] is False
    assert assignments["RUN_UPDATER_FULL"] is False
    assert assignments["RUN_JUDGE_CANARY"] is False
    assert assignments["RUN_JUDGE_FULL"] is False
    assert assignments["UPDATER_FULL_MAX_CALLS"] == 519
    assert assignments["JUDGE_CANARY_MAX_CALLS"] == 1
    assert assignments["JUDGE_FULL_MAX_CALLS"] == 4679
    assert assignments["JUDGE_CANARY_ALLOW_PARTIAL"] is True
    assert "execute_model_calls=RUN_UPDATER_CANARY" in code
    assert "execute_model_calls=RUN_UPDATER_FULL" in code
    assert "execute_model_calls=RUN_JUDGE_CANARY" in code
    assert "execute_model_calls=RUN_JUDGE_FULL" in code


def test_saved_pre_main_freeze_is_ready_and_read_only() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "main_final_primary.yaml")
    report = pre_main_readiness(config)
    assert report["verdict"] == "READY_TO_START_MAIN"
    assert all(report["checks"].values())
    assert report["estimate"]["physical_calls"] == 5156
    assert report["ollama_contacted"] is False
    assert not (PROJECT_ROOT / "runs" / "main-final-primary").exists()
