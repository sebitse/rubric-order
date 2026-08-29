from __future__ import annotations

import hashlib
import json
import sqlite3
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import public_config, resolve_project_path
from .data import dataset_fingerprint, load_dataset
from .downstream import APPROVED_JUDGE, verify_approved_judge
from .estimate import estimate_calls
from .ollama import judge_options, updater_options
from .readiness import readiness_report
from .run import file_sha256, read_json, write_json
from .schemas import JUDGE_JSON_SCHEMA, RUBRIC_V2_JSON_SCHEMA

FINAL_PRIMARY_CALLS = 5_156
FINAL_PRIMARY_HARD_CEILING = 5_200
PROTECTED_RUNS = (
    "runs/pilot-001",
    "runs/pilot-002",
    "runs/pilot-002-downstream-audit",
    "runs/judge-validation-v2",
)
METRIC_DEFINITIONS = {
    "quadratic_weighted_kappa": "Five-category quadratic weighted agreement.",
    "spearman": "Rank correlation with average ranks for ties.",
    "mean_absolute_error": "Mean absolute difference between predicted and helpfulness scores.",
    "mean_signed_error": "Mean predicted score minus helpfulness score.",
    "accuracy": "Fraction of exact predicted/helpfulness score matches.",
    "calibration_order_instability": (
        "Mean per-response population SD across the four calibration orders."
    ),
    "score_change_rate": "Fraction of responses with more than one score across orders.",
    "mean_score_range": "Mean per-response maximum minus minimum across orders.",
    "strict_preference_flip": "Strict pairwise ranking reversal across two orders.",
    "tie_change": "Exactly one of two order-specific response comparisons is tied.",
    "repeat_noise": "Same-order variation across original and repeat-control replicates.",
    "analysis_unit": "Held-out response nested in prompt and calibration-block trajectory.",
}


def estimate_final_primary(config: dict[str, Any]) -> dict[str, Any]:
    """Return the exact original primary design or fail rather than altering methodology."""
    verify_approved_judge(config)
    forbidden = [key for key in ("inventory_v3", "constrained_ablation") if key in config]
    checks = {
        "schema_v2": config.get("schema_version") == "2.0",
        "only_primary_methods": config.get("methods") == ["static", "batch", "sequential"],
        "all_four_orders": config.get("orders")
        == ["ascending", "descending", "random", "reverse_random"],
        "ten_blocks": config["sampling"]["calibration_blocks"] == 10,
        "fifty_heldout_responses": config["sampling"]["heldout_prompts"]
        * config["sampling"]["responses_per_prompt"]
        == 50,
        "original_repeat_controls": config["repeat_control"]
        == {
            "enabled": True,
            "blocks": [0, 4, 9],
            "orders": ["random"],
            "methods": ["batch", "sequential"],
            "replicates": [1, 2],
        },
        "no_constrained_configuration": not forbidden,
    }
    if not all(checks.values()):
        raise ValueError(
            "FINAL_ESTIMATE_MISMATCH: final primary design contract failed: "
            + ", ".join(name for name, passed in checks.items() if not passed)
        )
    estimate = estimate_calls(config)
    if estimate["total_inference_calls"] != FINAL_PRIMARY_CALLS:
        raise ValueError(
            "FINAL_ESTIMATE_MISMATCH: expected 5156 physical calls, found "
            f"{estimate['total_inference_calls']}"
        )
    return {
        "verdict": "FINAL_ESTIMATE_APPROVED",
        "physical_calls": FINAL_PRIMARY_CALLS,
        "formatted_physical_calls": "5.156",
        "breakdown": estimate,
        "design_checks": checks,
        "constrained_calls": 0,
        "pilot_003_calls": 0,
        "sampling_materialized": False,
        "ollama_contacted": False,
    }


def create_pre_main_freeze(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Freeze the complete primary contract before a main run directory exists."""
    estimate = estimate_final_primary(config)
    settings = config.get("pre_main")
    if not isinstance(settings, dict):
        raise ValueError("final primary config requires pre_main settings")
    target = resolve_project_path(config, config["output_root"]) / settings["run_id"]
    if target.exists():
        raise FileExistsError("pre-main freeze must be created before the final run directory")
    lineage_path = resolve_project_path(config, settings["lineage_audit"])
    downstream_path = resolve_project_path(config, settings["downstream_audit"])
    lineage = read_json(lineage_path)
    downstream = read_json(downstream_path)
    if lineage.get("verdict") != "DATA_LINEAGE_APPROVED":
        raise ValueError("pre-main freeze requires DATA_LINEAGE_APPROVED")
    if downstream.get("verdict") != "DOWNSTREAM_AUDIT_APPROVED":
        raise ValueError("pre-main freeze requires DOWNSTREAM_AUDIT_APPROVED")
    approved = verify_approved_judge(config)
    train, validation = load_dataset(config)
    fingerprint = dataset_fingerprint(train, validation)
    file_entries = _frozen_file_entries(config, lineage_path, downstream_path)
    schema = {"rubric": RUBRIC_V2_JSON_SCHEMA, "judge": JUDGE_JSON_SCHEMA}
    model_options = {
        "updater": updater_options(config),
        "judge": judge_options(config),
    }
    exclusion_entries = [
        {
            "path": row["path"],
            "required": bool(row.get("required", False)),
            "sha256": file_sha256(resolve_project_path(config, row["path"])),
        }
        for row in config["exclude_manifests"]
    ]
    payload = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "PRE_MAIN_CONTRACT_FROZEN",
        "config": {
            "path": config["_config_path"],
            "sha256": file_sha256(config["_config_path"]),
            "resolved_sha256": _object_hash(public_config(config)),
        },
        "files": file_entries,
        "dataset_fingerprint": fingerprint,
        "exclusion_sources": exclusion_entries,
        "schema": {"sha256": _object_hash(schema), "content": schema},
        "model_options": {
            "sha256": _object_hash(model_options),
            "content": model_options,
        },
        "metric_definitions": {
            "sha256": _object_hash(METRIC_DEFINITIONS),
            "content": METRIC_DEFINITIONS,
        },
        "approved_judge": approved,
        "judge_validation": {
            "verdict": "JUDGE_APPROVED",
            "artifact": str(resolve_project_path(config, config["approved_judge_artifact"])),
            "sha256": file_sha256(resolve_project_path(config, config["approved_judge_artifact"])),
        },
        "data_lineage": {
            "verdict": lineage["verdict"],
            "artifact": str(lineage_path),
            "sha256": file_sha256(lineage_path),
        },
        "downstream_audit": {
            "verdict": downstream["verdict"],
            "artifact": str(downstream_path),
            "sha256": file_sha256(downstream_path),
        },
        "estimate": estimate,
        "annotation_gate": {
            **config["drift_annotations"],
            "status_at_freeze": (
                "disabled for main-final-primary before inference; optional post-hoc export only"
            ),
        },
        "design_decision": config["paper_design"],
        "execution_safety": config["execution_safety"],
        "protected_run_tree_hashes": {
            relative: _directory_tree_sha256(resolve_project_path(config, relative))
            for relative in PROTECTED_RUNS
        },
        "git_used": False,
        "ollama_contacted": False,
        "main_run_exists_at_freeze": False,
    }
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = resolve_project_path(config, settings["freeze_manifest"])
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite pre-main freeze: {manifest_path}")
    write_json(manifest_path, payload)
    readiness = pre_main_readiness(config, manifest_path)
    report_path = output / "pre_main_readiness_primary_no_annotation.md"
    report_path.write_text(_readiness_markdown(readiness), encoding="utf-8")
    return {
        **readiness,
        "freeze_manifest": str(manifest_path),
        "readiness_report": str(report_path),
    }


def pre_main_readiness(
    config: dict[str, Any], freeze_manifest: str | Path | None = None
) -> dict[str, Any]:
    """Verify a saved pre-main freeze without writing files or contacting models."""
    estimate = estimate_final_primary(config)
    settings = config["pre_main"]
    freeze_path = (
        Path(freeze_manifest).resolve()
        if freeze_manifest is not None
        else resolve_project_path(config, settings["freeze_manifest"])
    )
    freeze = read_json(freeze_path) if freeze_path.exists() else None
    target = resolve_project_path(config, config["output_root"]) / settings["run_id"]
    current_fingerprint = dataset_fingerprint(*load_dataset(config))
    frozen_files_ok = bool(freeze) and all(
        Path(row["path"]).is_file() and file_sha256(row["path"]) == row["sha256"]
        for row in freeze.get("files", [])
    )
    checks = {
        "freeze_manifest_exists": freeze is not None,
        "freeze_created_before_main": freeze is not None
        and freeze.get("main_run_exists_at_freeze") is False,
        "main_target_absent": not target.exists(),
        "final_estimator_exact_5156": estimate["physical_calls"] == FINAL_PRIMARY_CALLS,
        "frozen_files_unchanged": frozen_files_ok,
        "dataset_fingerprint_unchanged": bool(freeze)
        and freeze.get("dataset_fingerprint") == current_fingerprint,
        "data_lineage_approved": bool(freeze)
        and freeze.get("data_lineage", {}).get("verdict") == "DATA_LINEAGE_APPROVED",
        "downstream_audit_approved": bool(freeze)
        and freeze.get("downstream_audit", {}).get("verdict") == "DOWNSTREAM_AUDIT_APPROVED",
        "judge_validation_approved": bool(freeze)
        and freeze.get("judge_validation", {}).get("verdict") == "JUDGE_APPROVED",
        "approved_judge_invariant": config["models"]["judge"] == APPROVED_JUDGE["model"]
        and verify_approved_judge(config)["prompt_sha256"] == APPROVED_JUDGE["prompt_sha256"],
        "annotation_gate_disabled_and_documented": bool(freeze)
        and config["drift_annotations"].get("required_before_judging") is False
        and config["drift_annotations"].get("optional_posthoc_inspection_only") is True
        and config["drift_annotations"].get("manual_semantic_drift_rate_estimated_in_main") is False
        and freeze.get("annotation_gate", {}).get("required_before_judging") is False,
        "hard_ceiling_exact_5200": config["execution_safety"].get("hard_total_call_ceiling")
        == FINAL_PRIMARY_HARD_CEILING,
        "maximum_one_retry": config["rubric"].get("max_validation_retries") == 1
        and config["execution_safety"].get("max_retries_per_request") == 1,
        "no_constrained_methods": config["methods"] == ["static", "batch", "sequential"]
        and "constrained_ablation" not in config
        and "inventory_v3" not in config,
    }
    verdict = "READY_TO_START_MAIN" if all(checks.values()) else "BLOCKED_BEFORE_MAIN"
    return {
        "verdict": verdict,
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
        "estimate": estimate,
        "freeze_manifest": str(freeze_path),
        "main_run_id": settings["run_id"],
        "judging_status": "ready_after_rubric_generation; no manual annotation gate",
        "manual_annotation_policy": (
            "No manual semantic-drift rate is estimated in main. Optional post-hoc rubric "
            "inspection is non-blocking and lexical flags are not treated as ground truth."
        ),
        "ollama_contacted": False,
        "read_only": True,
    }


def final_main_readiness(
    config: dict[str, Any],
    run_id: str = "main-final-primary",
    *,
    freeze_manifest: str | Path | None = None,
    executed_notebook: str | Path | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    """Audit a completed final-primary run without contacting the model server."""
    estimate = estimate_final_primary(config)
    run_dir = resolve_project_path(config, config["output_root"]) / run_id
    freeze_path = (
        Path(freeze_manifest).resolve()
        if freeze_manifest is not None
        else resolve_project_path(config, config["pre_main"]["freeze_manifest"])
    )
    notebook_path = (
        Path(executed_notebook).resolve()
        if executed_notebook is not None
        else run_dir / "notebooks" / "08_final_main_experiment.executed.ipynb"
    )
    freeze = read_json(freeze_path) if freeze_path.exists() else {}
    call_audit = _final_call_audit(run_dir / "calls.sqlite3")
    rubric_rows = _read_jsonl(run_dir / "artifacts" / "rubrics.jsonl")
    predictions = _read_jsonl(run_dir / "artifacts" / "predictions.jsonl")
    metric_path = run_dir / "metrics" / "summary.json"
    metric_summary = read_json(metric_path) if metric_path.exists() else {}
    standard_readiness = (
        readiness_report(config, run_id, write_report=False)
        if (run_dir / "manifest.json").exists()
        else {"verdict": "NOT_READY_FOR_MAIN", "checks": {}, "failures": ["run missing"]}
    )
    protected_current = {
        relative: _directory_tree_sha256(resolve_project_path(config, relative))
        for relative in PROTECTED_RUNS
    }
    protected_frozen = freeze.get("protected_run_tree_hashes", {})
    frozen_files_unchanged = bool(freeze) and all(
        Path(row["path"]).is_file() and file_sha256(row["path"]) == row["sha256"]
        for row in freeze.get("files", [])
    )
    canonical_rubrics = {
        row.get("canonical_rubric_sha256")
        for row in rubric_rows
        if row.get("canonical_rubric_sha256")
    }
    rubric_counts = {
        "indexed": len(rubric_rows),
        "canonical_unique": len(canonical_rubrics),
        "by_method": dict(sorted(Counter(row.get("method") for row in rubric_rows).items())),
    }
    failure_cases = _failure_cases(predictions)
    checks = {
        "standard_readiness_passed": standard_readiness.get("verdict") == "READY_FOR_MAIN",
        "updater_requests_506": call_audit["unique_success_by_stage"].get("build-rubrics", 0)
        == estimate["breakdown"]["rubric_stage_calls"],
        "judge_requests_4650": call_audit["unique_success_by_stage"].get("judge", 0)
        == estimate["breakdown"]["judge_stage_calls"],
        "physical_call_ceiling_respected": call_audit["physical_calls"]
        <= FINAL_PRIMARY_HARD_CEILING,
        "maximum_one_retry_per_request": call_audit["maximum_attempt"] <= 1,
        "error_rate_at_most_one_percent": call_audit["error_rate"]
        <= float(config["execution_safety"]["max_error_rate"]),
        "no_three_consecutive_same_errors": call_audit["maximum_consecutive_same_error"]
        < int(config["execution_safety"]["max_consecutive_same_error"]),
        "zero_context_saturation": call_audit["context_saturation_count"] == 0,
        "approved_models_only": call_audit["models_by_stage"].get("build-rubrics", [])
        == [config["models"]["updater"]]
        and call_audit["models_by_stage"].get("judge", []) == [config["models"]["judge"]],
        "approved_judge_prompt_invariant": call_audit["prompt_hashes_by_stage"].get("judge", [])
        == [APPROVED_JUDGE["prompt_sha256"]],
        "metrics_complete": bool(metric_summary)
        and metric_path.is_file()
        and (run_dir / "metrics" / "rubric_similarity.csv").is_file()
        and (run_dir / "metrics" / "repeat_variation.csv").is_file(),
        "frozen_files_unchanged": frozen_files_unchanged,
        "protected_runs_unchanged": bool(protected_frozen)
        and protected_current == protected_frozen,
        "annotation_gate_disabled": config["drift_annotations"].get("required_before_judging")
        is False,
        "executed_notebook_exists": notebook_path.is_file(),
        "planned_total_5156": estimate["physical_calls"] == FINAL_PRIMARY_CALLS,
    }
    invalid_names = {
        "physical_call_ceiling_respected",
        "maximum_one_retry_per_request",
        "error_rate_at_most_one_percent",
        "no_three_consecutive_same_errors",
        "zero_context_saturation",
        "approved_models_only",
        "approved_judge_prompt_invariant",
        "frozen_files_unchanged",
        "protected_runs_unchanged",
        "annotation_gate_disabled",
        "planned_total_5156",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if any(name in invalid_names for name in failed):
        verdict = "MAIN_EXPERIMENT_INVALID"
    elif failed:
        verdict = "MAIN_EXPERIMENT_INCOMPLETE"
    else:
        verdict = "MAIN_EXPERIMENT_COMPLETE"
    manifest = read_json(run_dir / "manifest.json") if (run_dir / "manifest.json").exists() else {}
    report = {
        "verdict": verdict,
        "checks": checks,
        "failures": failed,
        "duration_seconds": call_audit["duration_seconds"],
        "planned_calls": {
            "updater": estimate["breakdown"]["rubric_stage_calls"],
            "judge": estimate["breakdown"]["judge_stage_calls"],
            "total": estimate["physical_calls"],
            "hard_ceiling": FINAL_PRIMARY_HARD_CEILING,
        },
        "realized_calls": call_audit,
        "logical_prediction_cache_hits": sum(
            bool(row.get("cache_hit", False)) for row in predictions
        ),
        "sampling": manifest.get("sampling"),
        "exclusions": manifest.get("exclusions"),
        "rubrics": rubric_counts,
        "prediction_count": len(predictions),
        "metrics": metric_summary,
        "failure_cases": failure_cases,
        "freeze_manifest": str(freeze_path),
        "executed_notebook": str(notebook_path),
        "protected_run_tree_hashes": protected_current,
        "artifact_hashes": _final_artifact_hashes(run_dir, notebook_path),
        "git_operations": 0,
        "approved_judge": verify_approved_judge(config),
    }
    if write_report and run_dir.exists():
        write_json(run_dir / "final_readiness_report.json", report)
        (run_dir / "final_readiness_report.md").write_text(
            _final_readiness_markdown(report), encoding="utf-8"
        )
    return report


def execution_safety_status(
    config: dict[str, Any], run_id: str = "main-final-primary"
) -> dict[str, Any]:
    """Return the persistent physical-call/error guard used between execution phases."""
    run_dir = resolve_project_path(config, config["output_root"]) / run_id
    calls = _final_call_audit(run_dir / "calls.sqlite3")
    safety = config["execution_safety"]
    checks = {
        "hard_total_call_ceiling": calls["physical_calls"]
        <= int(safety["hard_total_call_ceiling"]),
        "maximum_one_retry_per_request": calls["maximum_attempt"]
        <= int(safety["max_retries_per_request"]),
        "error_rate": calls["error_rate"] <= float(safety["max_error_rate"]),
        "consecutive_same_error": calls["maximum_consecutive_same_error"]
        < int(safety["max_consecutive_same_error"]),
        "zero_context_saturation": calls["context_saturation_count"] == 0,
    }
    return {
        "safe": all(checks.values()),
        "checks": checks,
        "calls": calls,
        "failures": [name for name, passed in checks.items() if not passed],
    }


def _frozen_file_entries(
    config: dict[str, Any], lineage_path: Path, downstream_path: Path
) -> list[dict[str, str]]:
    configured = {
        Path(config["_config_path"]),
        resolve_project_path(config, config["prompts"]["sequential"]),
        resolve_project_path(config, config["prompts"]["batch"]),
        resolve_project_path(config, config["prompts"]["judge"]),
        resolve_project_path(config, config["rubric"]["initial"]),
        resolve_project_path(config, config["approved_judge_artifact"]),
        lineage_path,
        downstream_path,
        resolve_project_path(config, config["pre_main"]["notebook"]),
        resolve_project_path(config, config["pre_main"]["protocol"]),
        *(resolve_project_path(config, row["path"]) for row in config["exclude_manifests"]),
        *(
            path
            for path in (Path(config["_project_root"]) / "src" / "rubric_order").glob("*.py")
            if path.is_file()
        ),
    }
    return [
        {"path": str(path.resolve()), "sha256": file_sha256(path)}
        for path in sorted(configured, key=lambda value: value.as_posix())
    ]


def _readiness_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Pre-main readiness",
            "",
            f"**Verdict:** `{report['verdict']}`",
            "",
            "The primary design contains only static, free-form batch, and free-form sequential "
            "conditions. Pilot-003, inventory-v3, and constrained ablation are deferred optional "
            "extensions and are not used in the current paper's primary results.",
            "",
            "This decision was made before main execution: pilot evidence shows semantic rubric "
            "drift but only a very small downstream effect; constrained mitigation adds a second "
            "research question that would dilute a 2-4 page paper. It has not been executed, so "
            "this report makes no claim that it does not work.",
            "",
            "The exact physical-call estimate is **5,156**, with a hard ceiling of **5,200**. "
            "Manual thematic drift is qualitative pilot-002 context only; main judging has no "
            "manual annotation gate.",
            "",
            "## Checks",
            "",
            *[f"- `{name}`: `{passed}`" for name, passed in report["checks"].items()],
            "",
        ]
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _final_call_audit(database: Path) -> dict[str, Any]:
    empty = {
        "physical_calls": 0,
        "successful_calls": 0,
        "errors": 0,
        "error_rate": 0.0,
        "retries": 0,
        "maximum_attempt": 0,
        "maximum_consecutive_same_error": 0,
        "unique_success_by_stage": {},
        "physical_calls_by_stage": {},
        "models_by_stage": {},
        "prompt_hashes_by_stage": {},
        "prompt_eval_tokens": 0,
        "output_tokens": 0,
        "latency_seconds": {"total": 0.0, "mean": None, "minimum": None, "maximum": None},
        "minimum_context_headroom_tokens": None,
        "context_saturation_count": 0,
        "duration_seconds": 0.0,
        "status_counts": {},
    }
    if not database.exists():
        return empty
    connection = sqlite3.connect(
        f"file:{database.resolve().as_posix()}?mode=ro&immutable=1", uri=True
    )
    connection.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in connection.execute("SELECT * FROM calls ORDER BY id")]
    finally:
        connection.close()
    statuses = Counter(str(row["status"]) for row in rows)
    errors = sum(count for status, count in statuses.items() if status != "success")
    latencies = [float(row["latency_seconds"] or 0.0) for row in rows]
    stage_calls = Counter(str(row["stage"]) for row in rows)
    stages = sorted(stage_calls)
    unique_success = {
        stage: len(
            {
                row["request_hash"]
                for row in rows
                if row["stage"] == stage and row["status"] == "success"
            }
        )
        for stage in stages
    }
    maximum_streak = 0
    current_streak = 0
    previous_error: tuple[str, str] | None = None
    for row in rows:
        if row["status"] == "success":
            current_streak = 0
            previous_error = None
            continue
        error = (str(row["status"]), str(row.get("validation_error") or ""))
        current_streak = current_streak + 1 if error == previous_error else 1
        previous_error = error
        maximum_streak = max(maximum_streak, current_streak)
    timestamps = [datetime.fromisoformat(str(row["timestamp"])) for row in rows]
    return {
        "physical_calls": len(rows),
        "successful_calls": statuses.get("success", 0),
        "errors": errors,
        "error_rate": errors / len(rows) if rows else 0.0,
        "retries": sum(int(row["attempt"]) > 0 for row in rows),
        "maximum_attempt": max((int(row["attempt"]) for row in rows), default=0),
        "maximum_consecutive_same_error": maximum_streak,
        "unique_success_by_stage": dict(sorted(unique_success.items())),
        "physical_calls_by_stage": dict(sorted(stage_calls.items())),
        "models_by_stage": {
            stage: sorted({str(row["model_name"]) for row in rows if row["stage"] == stage})
            for stage in stages
        },
        "prompt_hashes_by_stage": {
            stage: sorted(
                {str(row["prompt_template_hash"]) for row in rows if row["stage"] == stage}
            )
            for stage in stages
        },
        "prompt_eval_tokens": sum(int(row["prompt_eval_count"] or 0) for row in rows),
        "output_tokens": sum(int(row["eval_count"] or 0) for row in rows),
        "latency_seconds": {
            "total": sum(latencies),
            "mean": statistics.fmean(latencies) if latencies else None,
            "minimum": min(latencies) if latencies else None,
            "maximum": max(latencies) if latencies else None,
        },
        "minimum_context_headroom_tokens": min(
            (
                int(row["observed_headroom_tokens"])
                for row in rows
                if row.get("observed_headroom_tokens") is not None
            ),
            default=None,
        ),
        "context_saturation_count": sum(bool(row.get("context_saturation")) for row in rows),
        "duration_seconds": (
            (max(timestamps) - min(timestamps)).total_seconds() if timestamps else 0.0
        ),
        "status_counts": dict(sorted(statuses.items())),
    }


def _failure_cases(predictions: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    normal = [row for row in predictions if int(row.get("replicate", 0)) == 0]
    ordered = sorted(
        normal,
        key=lambda row: (
            -abs(int(row["predicted_score"]) - int(row["human_score"])),
            str(row["example_id"]),
            str(row["method"]),
            int(row["block_id"]),
            str(row["order"]),
        ),
    )
    return [
        {
            "example_id": row["example_id"],
            "prompt_id": row["prompt_id"],
            "method": row["method"],
            "block_id": row["block_id"],
            "order": row["order"],
            "human_score": row["human_score"],
            "predicted_score": row["predicted_score"],
            "absolute_error": abs(int(row["predicted_score"]) - int(row["human_score"])),
        }
        for row in ordered[:limit]
    ]


def _final_artifact_hashes(run_dir: Path, notebook_path: Path) -> dict[str, str]:
    candidates = {
        "manifest": run_dir / "manifest.json",
        "resolved_config": run_dir / "config.resolved.json",
        "call_database": run_dir / "calls.sqlite3",
        "rubric_index": run_dir / "artifacts" / "rubrics.jsonl",
        "predictions": run_dir / "artifacts" / "predictions.jsonl",
        "metrics_summary": run_dir / "metrics" / "summary.json",
        "executed_notebook": notebook_path,
    }
    return {name: file_sha256(path) for name, path in candidates.items() if path.is_file()}


def _final_readiness_markdown(report: dict[str, Any]) -> str:
    calls = report["realized_calls"]
    return "\n".join(
        [
            "# Final main experiment readiness",
            "",
            f"**Verdict:** `{report['verdict']}`",
            "",
            f"- Planned updater calls: `{report['planned_calls']['updater']}`",
            "- Realized updater requests: "
            f"`{calls['unique_success_by_stage'].get('build-rubrics', 0)}`",
            f"- Planned judge calls: `{report['planned_calls']['judge']}`",
            f"- Realized judge requests: `{calls['unique_success_by_stage'].get('judge', 0)}`",
            f"- Physical inference calls including retries: `{calls['physical_calls']}`",
            f"- Retries: `{calls['retries']}`; errors: `{calls['errors']}`",
            f"- Duration: `{report['duration_seconds']}` seconds",
            "",
            "## Checks",
            "",
            *[f"- `{name}`: `{passed}`" for name, passed in report["checks"].items()],
            "",
        ]
    )


def _object_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _directory_tree_sha256(directory: Path) -> str:
    if not directory.is_dir():
        raise FileNotFoundError(f"protected run is missing: {directory}")
    inventory = [
        {
            "path": path.relative_to(directory).as_posix(),
            "sha256": file_sha256(path),
        }
        for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file() and not path.name.endswith(("-wal", "-shm"))
    ]
    return _object_hash(inventory)
