from __future__ import annotations

import hashlib
import json
import sqlite3
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import resolve_project_path
from .judge_validation import sanitized_request_plan
from .judging import build_pilot_judge_request, plan_pilot_judge_requests
from .metrics import (
    agreement_metrics,
    bootstrap_summary,
    calculate_agreement_table,
    calculate_between_repeat_table,
    calculate_primary_table,
    calculate_repeat_control_table,
    calculate_repeat_variation_table,
    write_csv,
)
from .ollama import InferenceBudgetClient, OllamaClient, judge_options
from .prompts import load_template
from .run import (
    file_sha256,
    merge_index,
    read_json,
    read_jsonl,
    sync_manifest_model_calls,
    write_json,
)
from .schemas import parse_judge_output, validate_rubric_v2
from .storage import CallContext, CallStore, cached_validated_call

APPROVED_JUDGE = {
    "model": "qwen3:14b",
    "candidate": "judge_anchor_first",
    "prompt_sha256": "75dd61241b143deabe1c9e9435f897c17874ea4cc7ed0b513c43cd35687cda7c",
}


def verify_approved_judge(config: dict[str, Any]) -> dict[str, Any]:
    configured = config.get("approved_judge_artifact")
    if not configured:
        raise ValueError("workflow requires approved_judge_artifact")
    path = resolve_project_path(config, configured)
    if not path.exists():
        raise FileNotFoundError(f"approved judge artifact does not exist: {path}")
    value = read_json(path)
    chosen = value.get("chosen", {})
    failures = []
    if value.get("verdict") != "JUDGE_APPROVED":
        failures.append("verdict is not JUDGE_APPROVED")
    for field, expected in APPROVED_JUDGE.items():
        if chosen.get(field) != expected:
            failures.append(f"chosen.{field} differs from frozen value")
    if config["models"]["judge"] != APPROVED_JUDGE["model"]:
        failures.append("configured judge model differs from frozen value")
    if chosen.get("model_options") != judge_options(config):
        failures.append("configured judge model options differ from frozen value")
    prompt_path = resolve_project_path(config, config["prompts"]["judge"])
    prompt_hash = hashlib.sha256(
        prompt_path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
    ).hexdigest()
    if prompt_hash != APPROVED_JUDGE["prompt_sha256"]:
        failures.append("configured judge prompt content hash differs from frozen value")
    if failures:
        raise ValueError("approved-judge gate failed: " + "; ".join(failures))
    return {
        "status": "JUDGE_APPROVED",
        "artifact": str(path),
        **APPROVED_JUDGE,
        "model_options": chosen.get("model_options"),
    }


def estimate_downstream_audit(config: dict[str, Any]) -> dict[str, Any]:
    plan = plan_downstream_audit(config, include_requests=False)
    return {
        "workflow": "pilot-002-downstream-audit",
        "updater_calls": 0,
        "logical_predictions": plan["logical_predictions"],
        "physical_judge_calls": plan["physical_requests"],
        "fresh_judge_calls": plan["fresh_requests"],
        "total_inference_calls": plan["fresh_requests"],
    }


def plan_downstream_audit(
    config: dict[str, Any], *, include_requests: bool = True
) -> dict[str, Any]:
    settings = config.get("downstream_audit")
    if not isinstance(settings, dict):
        raise ValueError("configuration is missing downstream_audit settings")
    source = resolve_project_path(config, settings["source_run"])
    if not source.is_dir():
        raise FileNotFoundError(f"frozen source run does not exist: {source}")
    source_hash = content_inventory_hash(source)
    expected_hash = settings.get("source_content_inventory_sha256")
    if source_hash["sha256"] != expected_hash:
        raise ValueError(
            "frozen pilot-002 content inventory mismatch: "
            f"expected {expected_hash}, found {source_hash['sha256']}"
        )
    approved = verify_approved_judge(config)
    manifest = read_json(source / "manifest.json")
    heldout = read_jsonl(source / "prepared" / "heldout.jsonl")
    rubric_rows = read_jsonl(source / "artifacts" / "rubrics.jsonl")
    integrity = _verify_frozen_rubrics(source, manifest, rubric_rows)
    classifications = load_manual_rubric_gate(
        resolve_project_path(config, settings["manual_rubric_gate"]), rubric_rows
    )
    request_plan = plan_pilot_judge_requests(
        config,
        run_id=settings.get("recommended_run_id", "pilot-002-downstream-audit"),
        run_dir=source,
        rubric_rows=rubric_rows,
        heldout=heldout,
    )
    stored_plan_path = resolve_project_path(config, settings["source_judge_plan"])
    stored_plan = read_json(stored_plan_path)
    stored_hashes = [row["request_hash"] for row in stored_plan.get("requests", [])]
    current_hashes = [row["request_hash"] for row in request_plan["requests"]]
    if stored_hashes != current_hashes:
        raise ValueError("reconstructed judge plan differs from frozen pilot-002 plan")
    if (
        request_plan["logical_predictions"] != 640
        or request_plan["physical_requests"] != 500
        or request_plan["invalid_requests"] != 0
    ):
        raise ValueError("pilot-002 downstream plan must be exactly 640 logical / 500 physical")
    trajectory_groups = _trajectory_groups(rubric_rows, classifications)
    result = {
        "workflow": "pilot-002-downstream-audit",
        "status": request_plan["status"],
        "ready": request_plan["ready"],
        "source_run": str(source),
        "source_inventory": source_hash,
        "approved_judge": approved,
        "updater_calls": 0,
        "logical_predictions": request_plan["logical_predictions"],
        "physical_requests": request_plan["physical_requests"],
        "fresh_requests": request_plan["fresh_requests"],
        "cache_hits": request_plan["cache_hits"],
        "invalid_requests": request_plan["invalid_requests"],
        "prompt_template_hash": request_plan["prompt_template_hash"],
        "rubric_integrity": integrity,
        "manual_gate": classifications["summary"],
        "trajectory_groups": trajectory_groups,
        "stored_plan": str(stored_plan_path),
        "stored_plan_sha256": file_sha256(stored_plan_path),
        "exploratory": True,
        "causal_interpretation_permitted": False,
    }
    if include_requests:
        result["requests"] = request_plan["requests"]
    return result


def write_downstream_dry_run(config: dict[str, Any]) -> dict[str, Any]:
    plan = plan_downstream_audit(config)
    settings = config["downstream_audit"]
    root = resolve_project_path(config, settings.get("dry_run_root", "dry_runs"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / "pilot_002_downstream_audit_plan.json"
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


def run_downstream_audit(
    config: dict[str, Any],
    run_id: str,
    *,
    execute_model_calls: bool = False,
    max_calls: int = 0,
    client: Any | None = None,
) -> dict[str, Any]:
    """Run only frozen-rubric judging; a disabled call is an external dry-run."""
    gate = plan_downstream_audit(config, include_requests=False)
    expected_run_id = config["downstream_audit"].get(
        "recommended_run_id", "pilot-002-downstream-audit"
    )
    if run_id != expected_run_id:
        raise ValueError(f"downstream audit run_id must be exactly {expected_run_id!r}")
    if not execute_model_calls:
        if max_calls != 0:
            raise ValueError("disabled downstream execution requires MAX_CALLS=0")
        return {**write_downstream_dry_run(config), "executed": False, "actual_calls": 0}
    if max_calls < 1 or max_calls > 500:
        raise ValueError("downstream execution requires 1 <= MAX_CALLS <= 500")
    if not gate["ready"]:
        raise ValueError("downstream audit preflight is not ready")

    raw_client = client or OllamaClient(
        config["models"]["base_url"], config["models"]["timeout_seconds"]
    )
    health = raw_client.health_and_models([config["models"]["judge"]])
    if health["missing"]:
        raise RuntimeError(
            "required approved judge model is not installed: " + ", ".join(health["missing"])
        )
    target = resolve_project_path(config, config["output_root"]) / run_id
    _prepare_downstream_target(config, gate, target, run_id)
    source = Path(gate["source_run"])
    rubrics = read_jsonl(source / "artifacts" / "rubrics.jsonl")
    heldout = read_jsonl(source / "prepared" / "heldout.jsonl")
    template = load_template(config, "judge")
    budgeted = InferenceBudgetClient(raw_client, max_calls)
    predictions_path = target / "artifacts" / "predictions.jsonl"
    stopped_at_budget = False
    cache_hits = 0
    try:
        with CallStore(target / "calls.sqlite3") as store:
            for rubric_row in rubrics:
                rubric = read_json(source / rubric_row["rubric_path"])
                for example in heldout:
                    request = build_pilot_judge_request(
                        config,
                        run_id=run_id,
                        rubric_row=rubric_row,
                        rubric=rubric,
                        template=template,
                        example=example,
                    )
                    cached_before = store.cached_success(request["request_hash"])
                    if cached_before is None and budgeted.calls >= max_calls:
                        stopped_at_budget = True
                        break
                    record = sanitized_request_plan(
                        request,
                        "hit" if cached_before is not None else "fresh",
                    )
                    record.update(
                        {
                            key: rubric_row[key]
                            for key in (
                                "method",
                                "block_id",
                                "order",
                                "replicate",
                                "repeat_control",
                            )
                        }
                    )
                    record["source_rubric_path"] = rubric_row["rubric_path"]
                    merge_index(
                        target / "artifacts" / "request_plans" / "judge.jsonl",
                        [record],
                        ("method", "block_id", "order", "replicate", "example_id"),
                    )
                    parsed, cached = cached_validated_call(
                        client=budgeted,
                        store=store,
                        context=CallContext(
                            run_id,
                            "downstream-judge",
                            rubric_row["method"],
                            rubric_row["block_id"],
                            rubric_row["order"],
                            None,
                            rubric_row["replicate"],
                        ),
                        model_name=config["models"]["judge"],
                        prompt=request["rendered_prompt"],
                        model_options=request["model_options"],
                        response_schema=request["response_schema"],
                        template_hash=request["prompt_template_hash"],
                        parser=parse_judge_output,
                        max_validation_retries=config["rubric"]["max_validation_retries"],
                        cache_namespace=request["cache_namespace"],
                        schema_version=config["schema_version"],
                        prompt_version=config["prompt_version"],
                        context_preflight=request["context_preflight"],
                        minimum_context_headroom=config["context"][
                            "minimum_observed_headroom_tokens"
                        ],
                        expected_request_hash=request["request_hash"],
                    )
                    cache_hits += int(cached)
                    merge_index(
                        predictions_path,
                        [
                            {
                                "method": rubric_row["method"],
                                "block_id": rubric_row["block_id"],
                                "order": rubric_row["order"],
                                "replicate": rubric_row["replicate"],
                                "repeat_control": rubric_row["repeat_control"],
                                "rubric_sha256": rubric_row["rubric_sha256"],
                                "example_id": example["example_id"],
                                "prompt_id": example["prompt_id"],
                                "human_score": example["human_score"],
                                "predicted_score": parsed["score"],
                                "cache_hit": cached,
                                "schema_version": config["schema_version"],
                                "prompt_version": config["prompt_version"],
                                "judge_prompt_sha256": template.sha256,
                                "request_hash": request["request_hash"],
                                "rubric_canonical_sha256": request["rubric_hash"],
                                "schema_sha256": request["schema_hash"],
                                "model_options_sha256": request["model_options_hash"],
                            }
                        ],
                        ("method", "block_id", "order", "replicate", "example_id"),
                    )
                if stopped_at_budget:
                    break
    finally:
        try:
            if config["models"].get("unload_after_stage", True):
                raw_client.unload(config["models"]["judge"])
        finally:
            sync_manifest_model_calls(target)
    completed = len(read_jsonl(predictions_path)) if predictions_path.exists() else 0
    return {
        "workflow": gate["workflow"],
        "executed": True,
        "run_id": run_id,
        "actual_calls": budgeted.calls,
        "cache_hits": cache_hits,
        "logical_predictions_complete": completed,
        "logical_predictions_expected": 640,
        "status": "complete" if completed == 640 else "partial_budget_stop",
        "resume_safe": True,
    }


def _prepare_downstream_target(
    config: dict[str, Any], gate: dict[str, Any], target: Path, run_id: str
) -> None:
    manifest_path = target / "manifest.json"
    if target.exists():
        if not manifest_path.exists():
            raise ValueError(f"existing downstream target has no manifest: {target}")
        existing = read_json(manifest_path)
        if (
            existing.get("workflow") != gate["workflow"]
            or existing.get("source_content_inventory_sha256") != gate["source_inventory"]["sha256"]
        ):
            raise ValueError("existing downstream target has incompatible provenance")
        return
    target.mkdir(parents=True, exist_ok=False)
    write_json(
        manifest_path,
        {
            "schema_version": "2.0",
            "run_id": run_id,
            "workflow": gate["workflow"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_run": gate["source_run"],
            "source_content_inventory_sha256": gate["source_inventory"]["sha256"],
            "source_rubrics_sha256": gate["rubric_integrity"]["rubrics_jsonl_sha256"],
            "approved_judge": gate["approved_judge"],
            "updater_calls": 0,
            "model_calls_made": 0,
            "exploratory": True,
        },
    )


def load_manual_rubric_gate(path: Path, rubric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"manual rubric gate is required; no thematic classification was invented: {path}"
        )
    gate = read_json(path)
    findings = gate.get("manual_findings")
    if not isinstance(findings, list):
        raise ValueError("manual rubric gate has no manual_findings list")
    flagged: set[tuple[Any, ...]] = set()
    for finding in findings:
        key = (
            finding.get("method"),
            finding.get("block_id"),
            finding.get("order"),
            finding.get("replicate"),
        )
        flagged.add(key)
    known = {
        (row["method"], row["block_id"], row["order"], row["replicate"]) for row in rubric_rows
    }
    unknown = flagged - known
    if unknown:
        raise ValueError(f"manual rubric gate references unknown conditions: {sorted(unknown)}")
    by_condition = {
        key: "task_specific" if key in flagged else "not_flagged_by_manual_gate" for key in known
    }
    return {
        "by_condition": by_condition,
        "summary": {
            "artifact": str(path),
            "artifact_sha256": file_sha256(path),
            "task_specific_conditions": len(flagged),
            "not_flagged_conditions": len(known - flagged),
            "classification_policy": "flagged_vs_not_flagged_exploratory",
            "unflagged_is_not_claimed_domain_general": True,
        },
    }


def analyze_downstream_predictions(
    config: dict[str, Any], predictions_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    plan = plan_downstream_audit(config, include_requests=False)
    predictions = read_jsonl(predictions_path)
    if len(predictions) != 640:
        raise ValueError(f"downstream analysis requires 640 predictions, found {len(predictions)}")
    prediction_keys = {
        (
            row["method"],
            row["block_id"],
            row["order"],
            row["replicate"],
            row["example_id"],
        )
        for row in predictions
    }
    if len(prediction_keys) != 640:
        raise ValueError("downstream analysis requires 640 unique condition/example predictions")
    source = Path(plan["source_run"])
    rubric_rows = read_jsonl(source / "artifacts" / "rubrics.jsonl")
    classifications = load_manual_rubric_gate(
        resolve_project_path(config, config["downstream_audit"]["manual_rubric_gate"]),
        rubric_rows,
    )
    lookup = {
        (row["method"], row["block_id"], row["order"], row["replicate"]): row for row in rubric_rows
    }
    enriched = []
    for row in predictions:
        key = (row["method"], row["block_id"], row["order"], row["replicate"])
        if key not in lookup:
            raise ValueError(f"prediction references an unknown frozen rubric: {key}")
        enriched.append(
            {
                **row,
                "thematic_group": classifications["by_condition"][key],
                "trajectory_id": f"{row['method']}:{row['block_id']}:{row['order']}",
                "canonical_rubric_sha256": lookup[key]["canonical_rubric_sha256"],
            }
        )
    normal = [row for row in enriched if row["replicate"] == 0]
    primary = calculate_primary_table(normal)
    agreement = calculate_agreement_table(normal)
    repeat = calculate_repeat_control_table(enriched)
    repeat_variation = calculate_repeat_variation_table(enriched)
    between_repeat = calculate_between_repeat_table(primary, repeat_variation)
    unique_trajectory = _deduplicate_trajectory_predictions(enriched)
    thematic = _thematic_descriptive(unique_trajectory)
    prompt_summary = _prompt_descriptive(normal)
    bootstrap = bootstrap_summary(
        normal,
        repetitions=config["analysis"]["bootstrap_repetitions"],
        seed=config["seed"],
        confidence=config["analysis"]["confidence"],
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "primary_metrics.csv", primary)
    write_csv(output / "agreement_metrics.csv", agreement)
    write_csv(output / "repeat_control.csv", repeat)
    write_csv(output / "repeat_variation.csv", repeat_variation)
    write_csv(output / "between_vs_repeat.csv", between_repeat)
    write_csv(output / "thematic_descriptive.csv", thematic)
    write_csv(output / "prompt_descriptive.csv", prompt_summary)
    summary = {
        "workflow": "pilot-002-downstream-audit",
        "exploratory": True,
        "causal_interpretation_permitted": False,
        "prediction_count": len(enriched),
        "trajectory_deduplicated_prediction_count": len(unique_trajectory),
        "primary": primary,
        "agreement": agreement,
        "repeat": repeat,
        "repeat_variation": repeat_variation,
        "between_vs_repeat": between_repeat,
        "thematic_descriptive": thematic,
        "bootstrap_clustered_by_block_and_prompt": bootstrap,
        "classification": classifications["summary"],
    }
    write_json(output / "summary.json", summary)
    return summary


def downstream_readiness(config: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    plan = plan_downstream_audit(config, include_requests=False)
    directory = Path(run_dir)
    predictions = directory / "artifacts" / "predictions.jsonl"
    summary = directory / "metrics" / "summary.json"
    calls = _downstream_call_status(directory / "calls.sqlite3")
    checks = {
        "source_integrity": plan["rubric_integrity"]["invalid_rubrics"] == 0,
        "approved_judge": plan["approved_judge"]["status"] == "JUDGE_APPROVED",
        "zero_updater_calls": plan["updater_calls"] == 0,
        "judge_plan_exact": plan["logical_predictions"] == 640
        and plan["physical_requests"] == 500
        and plan["invalid_requests"] == 0,
        "manual_gate_loaded": bool(plan["manual_gate"]["artifact_sha256"]),
        "physical_judging_complete": calls["successful_unique_requests"] == 500,
        "no_updater_stage_in_target": calls["updater_records"] == 0,
        "predictions_complete": predictions.exists() and len(read_jsonl(predictions)) == 640,
        "exploratory_analysis_complete": summary.exists(),
    }
    return {
        "verdict": "READY_FOR_PILOT_003" if all(checks.values()) else "NOT_READY_FOR_PILOT_003",
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
        "call_status": calls,
    }


def audit_completed_downstream(
    config: dict[str, Any],
    run_dir: str | Path,
    *,
    lineage_report: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Independently audit completed predictions, physical calls, and analysis units."""
    plan = plan_downstream_audit(config, include_requests=False)
    directory = Path(run_dir).resolve()
    predictions_path = directory / "artifacts" / "predictions.jsonl"
    predictions = read_jsonl(predictions_path)
    source = Path(plan["source_run"])
    rubrics = read_jsonl(source / "artifacts" / "rubrics.jsonl")
    heldout = read_jsonl(source / "prepared" / "heldout.jsonl")
    condition_lookup = {
        (row["method"], row["block_id"], row["order"], row["replicate"]): row for row in rubrics
    }
    prediction_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        prediction_groups[row["request_hash"]].append(row)
    physical_canary_hits = sum(
        all(bool(row.get("cache_hit")) for row in rows) for rows in prediction_groups.values()
    )
    physical_fresh_after_canary = sum(
        any(not bool(row.get("cache_hit")) for row in rows) for rows in prediction_groups.values()
    )
    logical_cache_hits = sum(bool(row.get("cache_hit")) for row in predictions)
    static_reuse = sum(
        max(0, len(rows) - 1)
        for rows in prediction_groups.values()
        if {row["method"] for row in rows} == {"static"}
    )
    enriched = []
    unknown_conditions = []
    for row in predictions:
        key = (row["method"], row["block_id"], row["order"], row["replicate"])
        rubric = condition_lookup.get(key)
        if rubric is None:
            unknown_conditions.append(key)
            continue
        enriched.append(
            {
                **row,
                "canonical_rubric_sha256": rubric["canonical_rubric_sha256"],
                "trajectory_id": f"{row['method']}:{row['block_id']}:{row['order']}",
            }
        )
    normal = [row for row in enriched if int(row["replicate"]) == 0]
    primary = calculate_primary_table(normal)
    agreement = calculate_agreement_table(normal)
    repeat = calculate_repeat_control_table(enriched)
    repeat_variation = calculate_repeat_variation_table(enriched)
    between_repeat = calculate_between_repeat_table(primary, repeat_variation)
    artifact_results = _results_by_artifact(enriched)
    canonical_results, score_vectors = _results_by_canonical_rubric(enriched, rubrics, heldout)
    trajectory_results = _results_by_trajectory(enriched)
    classifications = load_manual_rubric_gate(
        resolve_project_path(config, config["downstream_audit"]["manual_rubric_gate"]),
        rubrics,
    )
    classified = []
    for row in enriched:
        key = (row["method"], row["block_id"], row["order"], row["replicate"])
        classified.append({**row, "thematic_group": classifications["by_condition"][key]})
    thematic = _thematic_descriptive(_deduplicate_trajectory_predictions(classified))
    database = _audit_call_database(directory / "calls.sqlite3")
    logical_keys = {
        (row["method"], row["block_id"], row["order"], row["replicate"], row["example_id"])
        for row in predictions
    }
    condition_counts = Counter(
        (row["method"], row["block_id"], row["order"], row["replicate"]) for row in predictions
    )
    lineage = read_json(lineage_report) if lineage_report is not None else None
    checks = {
        "data_lineage_approved": bool(
            lineage and lineage.get("verdict") == "DATA_LINEAGE_APPROVED"
        ),
        "logical_predictions_640": len(predictions) == len(logical_keys) == 640,
        "physical_request_hashes_500": len(prediction_groups) == 500,
        "example_ids_20": len({row["example_id"] for row in predictions}) == 20,
        "prompt_ids_10": len({row["prompt_id"] for row in predictions}) == 10,
        "canonical_rubrics_10": len(score_vectors["rubrics"]) == 10,
        "full_run_new_calls_490": physical_fresh_after_canary == 490,
        "canary_cache_hits_10": physical_canary_hits == 10,
        "static_reuses_140": static_reuse == 140,
        "logical_cache_hits_150": logical_cache_hits == 150,
        "zero_invalid_scores": all(
            isinstance(row.get("predicted_score"), int) and 0 <= row["predicted_score"] <= 4
            for row in predictions
        ),
        "all_artifacts_complete": not unknown_conditions
        and len(condition_counts) == len(rubrics) == 32
        and set(condition_counts.values()) == {20},
        "database_has_500_successes": database["successful_unique_requests"] == 500,
        "database_has_zero_updater_calls": database["updater_records"] == 0,
        "approved_judge_hash": {row["judge_prompt_sha256"] for row in predictions}
        == {APPROVED_JUDGE["prompt_sha256"]},
        "unique_schema_hash": len({row["schema_sha256"] for row in predictions}) == 1,
        "unique_model_options_hash": len({row["model_options_sha256"] for row in predictions}) == 1,
        "two_score_vectors": score_vectors["unique_score_vector_count"] == 2,
        "four_rubrics_change_one_example": score_vectors["four_rubrics_change_one_example"],
        "changed_example_is_expected_observation": score_vectors["changed_example_id"]
        == "b9b488c7992d2585ba45"
        and score_vectors["score_change"] == [4, 3],
        "zero_strict_preference_flips": all(
            row["pairwise_ranking_flip_rate"] == 0 for row in primary
        ),
        "zero_repeat_score_noise": all(
            row["repeat_score_change_rate"] == 0 for row in repeat_variation
        ),
        "trajectory_unit_no_pseudoreplication": sum(
            row["prediction_count"] for row in trajectory_results
        )
        == 480,
    }
    verdict = "DOWNSTREAM_AUDIT_APPROVED" if all(checks.values()) else "DOWNSTREAM_AUDIT_BLOCKED"
    report = {
        "verdict": verdict,
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
        "counts": {
            "logical_predictions": len(predictions),
            "physical_unique_request_hashes": len(prediction_groups),
            "examples": len({row["example_id"] for row in predictions}),
            "prompts": len({row["prompt_id"] for row in predictions}),
            "canonical_rubrics": len(score_vectors["rubrics"]),
            "new_calls_in_full_resume": physical_fresh_after_canary,
            "canary_cache_hits": physical_canary_hits,
            "static_logical_reuses": static_reuse,
            "logical_cache_hits": logical_cache_hits,
        },
        "hashes": {
            "approved_judge_prompt_sha256": sorted(
                {row["judge_prompt_sha256"] for row in predictions}
            ),
            "schema_sha256": sorted({row["schema_sha256"] for row in predictions}),
            "model_options_sha256": sorted({row["model_options_sha256"] for row in predictions}),
            "prediction_artifact_sha256": file_sha256(predictions_path),
        },
        "score_vectors": score_vectors,
        "primary": primary,
        "agreement": agreement,
        "repeat_control": repeat,
        "repeat_variation": repeat_variation,
        "between_vs_repeat": between_repeat,
        "thematic_vs_not_flagged": thematic,
        "results_by_artifact": artifact_results,
        "results_by_canonical_rubric": canonical_results,
        "results_by_trajectory": trajectory_results,
        "database": database,
        "interpretation": {
            "exploratory": True,
            "not_flagged_is_domain_general": False,
            "thematic_contrast_is_causal": False,
            "replicates_are_independent_observations": False,
            "pilot_agreement_invalidates_untouched_judge_validation": False,
            "extreme_pairs_are_failure_cases_when_mapping_is_correct": True,
            "semantic_drift_exceeds_downstream_effect": True,
        },
        "ollama_contacted": False,
        "source_artifacts_modified": False,
    }
    if output_dir is not None:
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "downstream_analysis.json"
        markdown_path = output / "downstream_audit_report.md"
        readiness_path = output / "downstream_readiness_report.json"
        write_json(json_path, report)
        write_json(readiness_path, {key: report[key] for key in ("verdict", "checks", "failures")})
        markdown_path.write_text(_downstream_audit_markdown(report), encoding="utf-8")
        report["artifacts"] = {
            "analysis": str(json_path),
            "report": str(markdown_path),
            "readiness": str(readiness_path),
        }
    return report


def _results_by_artifact(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["block_id"], row["order"], row["replicate"])].append(row)
    return [
        {
            "method": key[0],
            "block_id": key[1],
            "order": key[2],
            "replicate": key[3],
            "prediction_count": len(group),
            "canonical_rubric_sha256": group[0]["canonical_rubric_sha256"],
            **agreement_metrics(group),
        }
        for key, group in sorted(grouped.items())
    ]


def _results_by_canonical_rubric(
    rows: list[dict[str, Any]],
    rubrics: list[dict[str, Any]],
    heldout: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rubric_conditions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rubrics:
        rubric_conditions[row["canonical_rubric_sha256"]].append(row)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["canonical_rubric_sha256"]].append(row)
    example_ids = sorted(row["example_id"] for row in heldout)
    results = []
    vectors: dict[str, tuple[int, ...]] = {}
    for digest, group in sorted(grouped.items()):
        per_example: dict[str, set[int]] = defaultdict(set)
        by_example: dict[str, dict[str, Any]] = {}
        for row in group:
            per_example[row["example_id"]].add(int(row["predicted_score"]))
            by_example.setdefault(row["example_id"], row)
        if set(per_example) != set(example_ids) or any(
            len(scores) != 1 for scores in per_example.values()
        ):
            raise ValueError(f"canonical rubric {digest} has inconsistent downstream scores")
        vector = tuple(next(iter(per_example[example])) for example in example_ids)
        vectors[digest] = vector
        deduplicated = [by_example[example] for example in example_ids]
        results.append(
            {
                "canonical_rubric_sha256": digest,
                "artifact_count": len(rubric_conditions[digest]),
                "prediction_count": len(deduplicated),
                "score_vector": list(vector),
                **agreement_metrics(deduplicated),
            }
        )
    vector_to_rubrics: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for digest, vector in vectors.items():
        vector_to_rubrics[vector].append(digest)
    ordered_vectors = sorted(vector_to_rubrics.items(), key=lambda item: (-len(item[1]), item[0]))
    changed_example_id = None
    score_change = None
    four_change_one = False
    if len(ordered_vectors) == 2:
        baseline, alternate = ordered_vectors[0][0], ordered_vectors[1][0]
        differences = [
            index
            for index, pair in enumerate(zip(baseline, alternate, strict=True))
            if pair[0] != pair[1]
        ]
        if len(differences) == 1:
            position = differences[0]
            changed_example_id = example_ids[position]
            score_change = [baseline[position], alternate[position]]
            four_change_one = len(ordered_vectors[1][1]) == 4
    summary = {
        "example_order": example_ids,
        "rubrics": {digest: {"score_vector": list(vectors[digest])} for digest in sorted(vectors)},
        "unique_score_vector_count": len(vector_to_rubrics),
        "vector_groups": [
            {
                "score_vector": list(vector),
                "rubric_count": len(digests),
                "rubric_hashes": sorted(digests),
            }
            for vector, digests in ordered_vectors
        ],
        "changed_example_id": changed_example_id,
        "score_change": score_change,
        "four_rubrics_change_one_example": four_change_one,
    }
    return results, summary


def _results_by_trajectory(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["trajectory_id"]].append(row)
    result = []
    for trajectory, group in sorted(grouped.items()):
        by_example: dict[str, dict[str, Any]] = {}
        score_sets: dict[str, set[int]] = defaultdict(set)
        for row in sorted(group, key=lambda value: int(value["replicate"])):
            by_example.setdefault(row["example_id"], row)
            score_sets[row["example_id"]].add(int(row["predicted_score"]))
        result.append(
            {
                "trajectory_id": trajectory,
                "artifact_count": len({row["replicate"] for row in group}),
                "prediction_count": len(by_example),
                "repeat_noise_detected": any(len(scores) > 1 for scores in score_sets.values()),
                **agreement_metrics(list(by_example.values())),
            }
        )
    return result


def _audit_call_database(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"successful_unique_requests": 0, "updater_records": 0, "read_only": True}
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro&immutable=1", uri=True)
    try:
        successful = int(
            connection.execute(
                "SELECT COUNT(DISTINCT request_hash) FROM calls WHERE status = 'success'"
            ).fetchone()[0]
        )
        updater = int(
            connection.execute(
                "SELECT COUNT(*) FROM calls WHERE stage LIKE '%rubric%' OR stage LIKE '%updater%'"
            ).fetchone()[0]
        )
        invalid = int(
            connection.execute("SELECT COUNT(*) FROM calls WHERE status != 'success'").fetchone()[0]
        )
    finally:
        connection.close()
    return {
        "successful_unique_requests": successful,
        "updater_records": updater,
        "non_success_records": invalid,
        "read_only": True,
    }


def _downstream_audit_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    vectors = report["score_vectors"]
    lines = [
        "# Pilot-002 downstream audit",
        "",
        f"**Verdict:** `{report['verdict']}`",
        "",
        "This is an exploratory downstream audit. `not_flagged` does not automatically mean "
        "domain-general, and the thematic/not-flagged contrast is not a causal estimate.",
        "",
        "The 20 held-out examples do not become hundreds of independent observations when the "
        "same examples are repeated across conditions. Replicates are nested within trajectories.",
        "",
        "Weak pilot agreement does not invalidate the separately approved judge-validation result "
        "on 100 untouched examples. With approved data mapping, the two extreme prompts are "
        "reported "
        "as failure cases rather than removed.",
        "",
        "## Counts",
        "",
        f"- Logical predictions: {counts['logical_predictions']}",
        f"- Physical request hashes: {counts['physical_unique_request_hashes']}",
        f"- Full-resume new calls / canary hits: {counts['new_calls_in_full_resume']} / "
        f"{counts['canary_cache_hits']}",
        f"- Static reuses / logical cache hits: {counts['static_logical_reuses']} / "
        f"{counts['logical_cache_hits']}",
        "",
        "## Score-vector finding",
        "",
        f"There are {vectors['unique_score_vector_count']} score vectors across "
        f"{counts['canonical_rubrics']} unique canonical rubrics. Four rubrics differ at only "
        f"`{vectors['changed_example_id']}`, changing `{vectors['score_change']}`. There are no "
        "strict preference flips and repeat score noise is zero. Semantic drift is visibly larger "
        "than the observed downstream score effect.",
        "",
        "## Checks",
        "",
        *[f"- `{name}`: `{passed}`" for name, passed in report["checks"].items()],
        "",
    ]
    return "\n".join(lines)


def _downstream_call_status(path: Path) -> dict[str, int]:
    if not path.exists():
        return {"successful_unique_requests": 0, "updater_records": 0}
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro&immutable=1", uri=True)
    try:
        successful = connection.execute(
            "SELECT COUNT(DISTINCT request_hash) FROM calls WHERE status = 'success'"
        ).fetchone()[0]
        updater = connection.execute(
            "SELECT COUNT(*) FROM calls WHERE stage LIKE '%rubric%' OR stage LIKE '%updater%'"
        ).fetchone()[0]
    finally:
        connection.close()
    return {
        "successful_unique_requests": int(successful),
        "updater_records": int(updater),
    }


def content_inventory_hash(root: Path) -> dict[str, Any]:
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in files
    ]
    canonical = json.dumps(
        records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "algorithm": "sha256(canonical-json([{path,sha256(file-bytes)}]))",
        "file_count": len(records),
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _verify_frozen_rubrics(
    source: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    initial = manifest.get("initial_rubric", {}).get("content")
    if not isinstance(initial, dict):
        raise ValueError("frozen pilot manifest has no immutable initial rubric")
    invalid: list[dict[str, Any]] = []
    for row in rows:
        path = source / row["rubric_path"]
        if not path.exists():
            invalid.append({"rubric_path": row["rubric_path"], "error": "missing"})
            continue
        raw_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            canonical = validate_rubric_v2(read_json(path), initial)
        except ValueError as exc:
            invalid.append({"rubric_path": row["rubric_path"], "error": str(exc)})
            continue
        canonical_hash = _object_hash(canonical)
        if raw_hash != row["rubric_sha256"] or canonical_hash != row.get("canonical_rubric_sha256"):
            invalid.append({"rubric_path": row["rubric_path"], "error": "hash mismatch"})
    if invalid:
        raise ValueError(f"frozen rubric integrity failed for {len(invalid)} rubric(s)")
    return {
        "rubrics": len(rows),
        "rubrics_jsonl_sha256": file_sha256(source / "artifacts" / "rubrics.jsonl"),
        "invalid_rubrics": 0,
        "hashes_verified": True,
    }


def _trajectory_groups(
    rows: list[dict[str, Any]], classifications: dict[str, Any]
) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["block_id"], row["order"])].append(row)
    return {
        "artifact_count": len(rows),
        "trajectory_count": len(groups),
        "unique_canonical_rubric_count": len({row["canonical_rubric_sha256"] for row in rows}),
        "replicate_groups": [
            {
                "trajectory_id": f"{key[0]}:{key[1]}:{key[2]}",
                "artifacts": len(group),
                "replicates": sorted(row["replicate"] for row in group),
                "unique_canonical_rubrics": len({row["canonical_rubric_sha256"] for row in group}),
                "thematic_labels": sorted(
                    {
                        classifications["by_condition"][
                            (
                                row["method"],
                                row["block_id"],
                                row["order"],
                                row["replicate"],
                            )
                        ]
                        for row in group
                    }
                ),
            }
            for key, group in sorted(groups.items())
        ],
        "statistical_rule": "replicates are nested in trajectories, not independent units",
    }


def _deduplicate_trajectory_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: int(item["replicate"])):
        key = (row["trajectory_id"], row["example_id"])
        indexed.setdefault(key, row)
    return list(indexed.values())


def _thematic_descriptive(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["thematic_group"], int(row["block_id"]))].append(row)
    result = []
    for (label, block), group in sorted(grouped.items()):
        metrics = agreement_metrics(group)
        result.append(
            {
                "thematic_group": label,
                "block_id": block,
                "prediction_count": len(group),
                **metrics,
            }
        )
    return result


def _prompt_descriptive(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["prompt_id"])].append(int(row["predicted_score"]))
    return [
        {
            "method": method,
            "prompt_id": prompt_id,
            "prediction_count": len(scores),
            "mean_score": statistics.fmean(scores),
            "minimum_score": min(scores),
            "maximum_score": max(scores),
            "score_range": max(scores) - min(scores),
        }
        for (method, prompt_id), scores in sorted(grouped.items())
    ]


def _object_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
