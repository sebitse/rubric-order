from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import public_config, resolve_project_path
from .downstream import downstream_readiness, verify_approved_judge
from .estimate import Selection, estimate_calls
from .inventory_v3 import execute_inventory_requests, plan_inventory_requests
from .judging import judge_rubrics, plan_pilot_judge_requests
from .metrics import (
    calculate_agreement_table,
    calculate_primary_table,
    calculate_repeat_variation_table,
)
from .run import file_sha256, read_json, read_jsonl, write_json, write_jsonl


def validate_paired_source(config: dict[str, Any]) -> dict[str, Any]:
    paired = config.get("paired_source")
    if not isinstance(paired, dict):
        raise ValueError("pilot v3 configuration requires paired_source")
    source = resolve_project_path(config, paired["run"])
    manifest_path = source / "manifest.json"
    blocks_path = source / "prepared" / "calibration_blocks.json"
    heldout_path = source / "prepared" / "heldout.jsonl"
    for path in (manifest_path, blocks_path, heldout_path):
        if not path.exists():
            raise FileNotFoundError(f"paired pilot source artifact is missing: {path}")
    expected = paired.get("sha256", {})
    actual = {
        "manifest": file_sha256(manifest_path),
        "calibration_blocks": file_sha256(blocks_path),
        "heldout": file_sha256(heldout_path),
    }
    mismatches = [name for name, digest in actual.items() if expected.get(name) != digest]
    manifest = read_json(manifest_path)
    blocks = read_json(blocks_path)
    heldout = read_jsonl(heldout_path)
    checks = {
        "hashes_match": not mismatches,
        "block_count": len(blocks) == config["sampling"]["calibration_blocks"] == 2,
        "examples_per_block": all(
            len(block["examples"]) == config["sampling"]["examples_per_score"] * 5 == 10
            for block in blocks
        ),
        "orders_match": all(set(block["orders"]) == set(config["orders"]) for block in blocks),
        "heldout_match": len(heldout)
        == config["sampling"]["heldout_prompts"] * config["sampling"]["responses_per_prompt"]
        == 20,
        "manifest_run_id": manifest.get("run_id") == paired.get("run_id", "pilot-002"),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if mismatches:
        failures.append("hash mismatch: " + ", ".join(mismatches))
    if failures:
        raise ValueError("paired-source preflight failed: " + "; ".join(failures))
    identity_payload = {
        "block_orders": [block["orders"] for block in blocks],
        "block_example_ids": [[row["example_id"] for row in block["examples"]] for block in blocks],
        "heldout_example_ids": [row["example_id"] for row in heldout],
        "heldout_prompt_ids": [row["prompt_id"] for row in heldout],
    }
    return {
        "status": "paired_source_verified",
        "source_run": str(source),
        "checks": checks,
        "sha256": actual,
        "identity_sha256": _object_hash(identity_payload),
        "blocks": blocks,
        "heldout": heldout,
    }


def estimate_pilot_v3(config: dict[str, Any]) -> dict[str, Any]:
    source = validate_paired_source(config)
    verify_approved_judge(config)
    standard = estimate_calls(config, heldout_response_count=len(source["heldout"]))
    if standard["rubric_stage_calls"] != 132 or standard["judge_stage_calls"] != 500:
        raise ValueError("pilot v3 estimator must produce 132 updater and 500 judge calls")
    return {
        "workflow": "pilot-003",
        "paired_source_identity_sha256": source["identity_sha256"],
        "updater_calls": 132,
        "judge_calls": 500,
        "logical_predictions": 640,
        "total_inference_calls": 632,
        "resampling_performed": False,
    }


def plan_pilot_v3(config: dict[str, Any], run_id: str = "pilot-003") -> dict[str, Any]:
    source = validate_paired_source(config)
    estimate = estimate_pilot_v3(config)
    target = resolve_project_path(config, config["output_root"]) / run_id
    updater = plan_inventory_requests(
        config,
        run_id=run_id,
        blocks=source["blocks"],
        state_directory=(
            target / "artifacts" / "rubrics" / "constrained_sequential" if target.exists() else None
        ),
        cache_database=target / "calls.sqlite3" if target.exists() else None,
    )
    initial_phase_counts_ok = target.exists() or (
        updater["exact_requests"] == 24 and updater["deferred_requests"] == 108
    )
    if (
        not initial_phase_counts_ok
        or updater["exact_requests"] + updater["deferred_requests"] != 132
        or updater["planned_physical_updater_calls"] != 132
    ):
        raise ValueError("pilot v3 phase-aware updater plan has unexpected counts")
    return {
        "workflow": "pilot-003",
        "status": updater["status"],
        "ready": updater["ready"],
        "run_id": run_id,
        "paired_source": {
            key: source[key]
            for key in ("status", "source_run", "checks", "sha256", "identity_sha256")
        },
        "updater": updater,
        "judge": {
            "status": "blocked_by_constrained_rubric_generation",
            "logical_predictions": 640,
            "physical_requests": 500,
            "invalid_requests": 0,
        },
        "estimate": estimate,
        "ollama_contacted": False,
    }


def prepare_paired_pilot_v3(config: dict[str, Any], run_id: str = "pilot-003") -> dict[str, Any]:
    source = validate_paired_source(config)
    output_root = resolve_project_path(config, config["output_root"])
    target = output_root / run_id
    if target.exists():
        raise FileExistsError(f"paired pilot target already exists: {target}")
    target.mkdir(parents=True)
    prepared = target / "prepared"
    prepared.mkdir()
    source_path = Path(source["source_run"])
    shutil.copy2(source_path / "prepared" / "calibration_blocks.json", prepared)
    shutil.copy2(source_path / "prepared" / "heldout.jsonl", prepared)
    write_json(target / "config.resolved.json", public_config(config))
    payload = {
        "schema_version": "3.0",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": _object_hash(public_config(config)),
        "prepared": True,
        "paired_development_comparison": True,
        "resampling_performed": False,
        "paired_source": {
            key: source[key] for key in ("source_run", "sha256", "identity_sha256", "checks")
        },
        "model_calls_made": 0,
    }
    write_json(target / "manifest.json", payload)
    return payload


def write_pilot_v3_dry_run(config: dict[str, Any]) -> dict[str, Any]:
    plan = plan_pilot_v3(config)
    path = resolve_project_path(config, "dry_runs/pilot-v3/plan.json")
    write_json(
        path,
        {
            **plan,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inference_executed": False,
            "ollama_contacted": False,
        },
    )
    return {**plan, "dry_run_manifest": str(path), "ollama_contacted": False}


def run_pilot_v3_updater(
    config: dict[str, Any],
    *,
    run_id: str = "pilot-003",
    execute_model_calls: bool = False,
    max_calls: int = 0,
    resume: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    plan = plan_pilot_v3(config, run_id)
    if not execute_model_calls:
        if max_calls != 0:
            raise ValueError("disabled pilot v3 execution requires MAX_CALLS=0")
        return {**write_pilot_v3_dry_run(config), "executed": False, "actual_calls": 0}
    if run_id != "pilot-003":
        raise ValueError("paired constrained pilot run_id must be exactly 'pilot-003'")
    if not plan["ready"]:
        raise ValueError("pilot v3 preflight is not ready")
    required = plan["updater"]["execution_fresh_calls"]
    if max_calls < required:
        raise ValueError(f"pilot v3 updater requires {required} calls; MAX_CALLS={max_calls}")
    target = resolve_project_path(config, config["output_root"]) / run_id
    if target.exists():
        if not resume:
            raise FileExistsError(
                "pilot v3 target exists; pass resume=True only after inspecting its manifest"
            )
        manifest = read_json(target / "manifest.json")
        if manifest.get("config_hash") != _object_hash(public_config(config)):
            raise ValueError("existing pilot v3 target uses a different configuration")
    else:
        prepare_paired_pilot_v3(config, run_id)
    source = validate_paired_source(config)
    result = execute_inventory_requests(
        config,
        run_id=run_id,
        run_dir=target,
        blocks=source["blocks"],
        max_calls=max_calls,
        client=client,
    )
    return {**plan, **result, "executed": True}


def plan_pilot_v3_judging(config: dict[str, Any], run_id: str = "pilot-003") -> dict[str, Any]:
    validate_paired_source(config)
    verify_approved_judge(config)
    target = resolve_project_path(config, config["output_root"]) / run_id
    manifest_path = target / "manifest.json"
    index_path = target / "artifacts" / "rubrics.jsonl"
    if not manifest_path.exists() or not index_path.exists():
        return {
            "workflow": "pilot-003",
            "stage": "judge",
            "status": "blocked_by_constrained_rubric_generation",
            "ready": False,
            "logical_predictions": 640,
            "physical_requests": 500,
            "fresh_requests": 500,
            "updater_calls": 0,
            "ollama_contacted": False,
        }
    manifest = read_json(manifest_path)
    if manifest.get("config_hash") != _object_hash(public_config(config)):
        raise ValueError("pilot v3 judging target uses a different configuration")
    rows = read_jsonl(index_path)
    counts = {
        method: sum(row["method"] == method for row in rows)
        for method in ("static", "constrained_batch", "constrained_sequential")
    }
    if counts != {"static": 8, "constrained_batch": 12, "constrained_sequential": 12}:
        raise ValueError(f"pilot v3 rubric inventory is incomplete: {counts}")
    heldout = read_jsonl(target / "prepared" / "heldout.jsonl")
    plan = plan_pilot_judge_requests(
        config,
        run_id=run_id,
        run_dir=target,
        rubric_rows=rows,
        heldout=heldout,
    )
    if plan["logical_predictions"] != 640 or plan["physical_requests"] != 500:
        raise ValueError("pilot v3 judging must be exactly 640 logical / 500 physical")
    return {
        **plan,
        "workflow": "pilot-003",
        "stage": "judge",
        "updater_calls": 0,
        "ollama_contacted": False,
    }


def run_pilot_v3_judge(
    config: dict[str, Any],
    *,
    run_id: str = "pilot-003",
    execute_model_calls: bool = False,
    max_calls: int = 0,
    client: Any | None = None,
) -> dict[str, Any]:
    plan = plan_pilot_v3_judging(config, run_id)
    if not execute_model_calls:
        if max_calls != 0:
            raise ValueError("disabled pilot v3 judging requires MAX_CALLS=0")
        return {**plan, "executed": False, "actual_calls": 0}
    if not plan["ready"]:
        raise ValueError("pilot v3 judging is blocked by incomplete constrained rubrics")
    if max_calls < plan["fresh_requests"]:
        raise ValueError(
            f"pilot v3 judging requires {plan['fresh_requests']} fresh calls; MAX_CALLS={max_calls}"
        )
    result = judge_rubrics(config, run_id, Selection(), max_calls, client=client)
    return {**plan_pilot_v3_judging(config, run_id), **result, "executed": True}


def estimate_main_v3(config: dict[str, Any]) -> dict[str, Any]:
    verify_approved_judge(config)
    base = estimate_calls(config)
    constrained = config.get("constrained_ablation")
    if not isinstance(constrained, dict):
        raise ValueError("main v3 requires constrained_ablation settings")
    blocks = constrained.get("blocks")
    orders = constrained.get("orders")
    if blocks != [0, 4, 9]:
        raise ValueError("constrained ablation must use exactly blocks 0, 4, and 9")
    if orders != config["orders"] or len(orders) != 4:
        raise ValueError("constrained ablation must use all four configured orders")
    if constrained.get("repeat_control") is not False:
        raise ValueError("constrained ablation may not add repeat controls")
    heldout = config["sampling"]["heldout_prompts"] * config["sampling"]["responses_per_prompt"]
    batch_updater = len(blocks) * len(orders)
    sequential_updater = batch_updater * config["sampling"]["examples_per_score"] * 5
    batch_judge = batch_updater * heldout
    sequential_judge = batch_updater * heldout
    ablation = batch_updater + sequential_updater + batch_judge + sequential_judge
    if base["total_inference_calls"] != 5156 or ablation != 1332:
        raise ValueError("main v3 estimator does not match frozen 5156 + 1332 design")
    required_sources = {
        "runs/pilot-001/manifest.json",
        "runs/judge-validation-v2/manifest.json",
        "runs/pilot-002/manifest.json",
        "runs/pilot-003/manifest.json",
    }
    configured_sources = {
        str(row["path"]).replace("\\", "/") for row in config["exclude_manifests"]
    }
    if not required_sources <= configured_sources:
        raise ValueError("main v3 exclusions do not separate all development and final sets")
    return {
        "workflow": "main-v3",
        "original_main_v2_calls": base["total_inference_calls"],
        "constrained_ablation": {
            "blocks": blocks,
            "orders": orders,
            "repeat_control": False,
            "batch_updater_calls": batch_updater,
            "sequential_updater_calls": sequential_updater,
            "updater_calls": batch_updater + sequential_updater,
            "batch_judge_calls": batch_judge,
            "sequential_judge_calls": sequential_judge,
            "judge_calls": batch_judge + sequential_judge,
            "total_calls": ablation,
        },
        "total_inference_calls": base["total_inference_calls"] + ablation,
        "sampling_materialized": False,
        "ollama_contacted": False,
    }


def analyze_paired_pilot_v3(
    free_form_predictions: str | Path,
    constrained_predictions: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Compare saved free-form and constrained predictions on exact matched units."""
    free_rows = read_jsonl(free_form_predictions)
    constrained_rows = read_jsonl(constrained_predictions)
    method_pairs = {
        "batch": "constrained_batch",
        "sequential": "constrained_sequential",
    }
    paired_rows: list[dict[str, Any]] = []
    free_selected: list[dict[str, Any]] = []
    constrained_selected: list[dict[str, Any]] = []
    for free_method, constrained_method in method_pairs.items():
        left = {_prediction_key(row): row for row in free_rows if row["method"] == free_method}
        right = {
            _prediction_key(row): row
            for row in constrained_rows
            if row["method"] == constrained_method
        }
        if set(left) != set(right):
            missing_left = len(set(right) - set(left))
            missing_right = len(set(left) - set(right))
            raise ValueError(
                "paired pilot predictions do not share exact units: "
                f"missing_free={missing_left}, missing_constrained={missing_right}"
            )
        for key in sorted(left):
            free = left[key]
            constrained = right[key]
            if free["human_score"] != constrained["human_score"]:
                raise ValueError(f"paired human-score mismatch for {key}")
            free_selected.append(free)
            constrained_selected.append({**constrained, "method": free_method})
            free_error = abs(int(free["predicted_score"]) - int(free["human_score"]))
            constrained_error = abs(
                int(constrained["predicted_score"]) - int(constrained["human_score"])
            )
            paired_rows.append(
                {
                    "method": free_method,
                    "block_id": free["block_id"],
                    "order": free["order"],
                    "replicate": free["replicate"],
                    "example_id": free["example_id"],
                    "prompt_id": free["prompt_id"],
                    "human_score": free["human_score"],
                    "free_form_score": free["predicted_score"],
                    "constrained_score": constrained["predicted_score"],
                    "score_difference": int(constrained["predicted_score"])
                    - int(free["predicted_score"]),
                    "absolute_error_difference": constrained_error - free_error,
                    "score_changed": free["predicted_score"] != constrained["predicted_score"],
                }
            )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "paired_predictions.jsonl", paired_rows)
    summary = {
        "paired": True,
        "causal_interpretation_permitted": False,
        "paired_prediction_count": len(paired_rows),
        "free_form": {
            "primary": calculate_primary_table(
                [row for row in free_selected if int(row["replicate"]) == 0]
            ),
            "agreement": calculate_agreement_table(
                [row for row in free_selected if int(row["replicate"]) == 0]
            ),
            "repeat_noise": calculate_repeat_variation_table(free_selected),
        },
        "constrained": {
            "primary": calculate_primary_table(
                [row for row in constrained_selected if int(row["replicate"]) == 0]
            ),
            "agreement": calculate_agreement_table(
                [row for row in constrained_selected if int(row["replicate"]) == 0]
            ),
            "repeat_noise": calculate_repeat_variation_table(constrained_selected),
        },
        "paired_change": _paired_change_summary(paired_rows),
        "drift_annotations": "join separately by canonical rubric hash after blinded freeze",
    }
    write_json(output / "summary.json", summary)
    return summary


def write_main_v3_dry_run(config: dict[str, Any]) -> dict[str, Any]:
    estimate = estimate_main_v3(config)
    path = resolve_project_path(config, "dry_runs/main-v3/estimate.json")
    write_json(
        path,
        {
            **estimate,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inference_executed": False,
            "ollama_contacted": False,
        },
    )
    return {**estimate, "dry_run_manifest": str(path)}


def pilot_v3_readiness(
    config: dict[str, Any],
    downstream_run: str | Path,
    *,
    downstream_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = validate_paired_source(config)
    if downstream_config is None:
        raise ValueError("pilot v3 readiness requires the downstream audit configuration")
    downstream = downstream_readiness(downstream_config, downstream_run)
    checks = {
        "paired_source_verified": source["status"] == "paired_source_verified",
        "downstream_ready": downstream["verdict"] == "READY_FOR_PILOT_003",
        "target_absent": not (
            resolve_project_path(config, config["output_root"]) / "pilot-003"
        ).exists(),
    }
    return {
        "verdict": "READY_TO_PREPARE_PILOT_003"
        if all(checks.values())
        else "NOT_READY_FOR_PILOT_003",
        "checks": checks,
        "downstream": downstream,
    }


def main_v3_readiness(
    config: dict[str, Any], *, annotation_freeze: dict[str, Any] | None = None
) -> dict[str, Any]:
    estimate = estimate_main_v3(config)
    target = resolve_project_path(config, config["output_root"]) / "main-v3"
    checks = {
        "approved_judge": True,
        "estimator_exact": estimate["total_inference_calls"] == 6488,
        "target_absent": not target.exists(),
        "sampling_not_materialized": estimate["sampling_materialized"] is False,
        "annotation_freeze_ready": bool(annotation_freeze and annotation_freeze.get("ready")),
    }
    return {
        "verdict": "READY_TO_PREPARE_MAIN_V3" if all(checks.values()) else "NOT_READY_FOR_MAIN_V3",
        "checks": checks,
        "prerequisite_note": (
            "Drift annotations must be frozen before judging; absence is an expected "
            "pre-execution prerequisite during preparation."
        ),
        "estimate": estimate,
    }


def _object_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _prediction_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["block_id"]),
        str(row["order"]),
        int(row["replicate"]),
        str(row["example_id"]),
    )


def _paired_change_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for method in sorted({row["method"] for row in rows}):
        group = [row for row in rows if row["method"] == method]
        result.append(
            {
                "method": method,
                "paired_predictions": len(group),
                "score_change_rate": sum(row["score_changed"] for row in group) / len(group),
                "mean_score_difference": sum(row["score_difference"] for row in group) / len(group),
                "mean_absolute_error_difference": sum(
                    row["absolute_error_difference"] for row in group
                )
                / len(group),
                "strict_direction_flips_against_human": sum(
                    (row["free_form_score"] - row["human_score"])
                    * (row["constrained_score"] - row["human_score"])
                    < 0
                    for row in group
                ),
            }
        )
    return result
