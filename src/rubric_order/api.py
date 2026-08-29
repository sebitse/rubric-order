from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import (
    DEFAULTS,
    apply_config_overrides,
    load_config,
    public_config,
    resolve_project_path,
)
from .data import load_dataset
from .downstream import (
    analyze_downstream_predictions,
    audit_completed_downstream,
    downstream_readiness,
    estimate_downstream_audit,
    plan_downstream_audit,
    run_downstream_audit,
    verify_approved_judge,
    write_downstream_dry_run,
)
from .drift_annotations import (
    annotation_agreement,
    annotation_freeze_gate,
    export_blinded_drift,
    freeze_annotations,
    lexical_drift_flags,
    plan_blinded_drift_export,
    validate_annotation_import,
)
from .estimate import Selection, estimate_calls, plan_rubric_jobs
from .final_primary import (
    create_pre_main_freeze,
    estimate_final_primary,
    pre_main_readiness,
)
from .final_primary import (
    execution_safety_status as _execution_safety_status,
)
from .inventory_v3 import (
    load_inventory,
    materialize_inventory_rubric,
    plan_inventory_requests,
    validate_inventory_selection,
)
from .judge_validation import (
    compact_judge_plan,
    estimate_judge_validation,
    finalize_judge_validation,
    plan_judge_validation,
    prepare_judge_validation,
    run_judge_dev,
    run_judge_holdout,
    summarize_judge_dev,
)
from .judging import judge_rubrics as _judge_rubrics
from .judging import plan_pilot_judge_requests
from .lineage import audit_extreme_pair_lineage, resolve_dataset_lineage
from .metrics import analyze_run
from .ollama import OllamaClient
from .readiness import readiness_report
from .rubric_generation import build_rubrics as _build_rubrics
from .rubric_generation import plan_rubric_context
from .run import (
    load_prepared,
    prepare_run,
    read_json,
    read_jsonl,
    run_directory,
    write_dry_run,
    write_json,
    write_jsonl,
)
from .schemas import validate_rubric, validate_rubric_v2
from .v3_design import (
    analyze_paired_pilot_v3,
    estimate_main_v3,
    estimate_pilot_v3,
    main_v3_readiness,
    pilot_v3_readiness,
    plan_pilot_v3,
    plan_pilot_v3_judging,
    run_pilot_v3_judge,
    run_pilot_v3_updater,
    validate_paired_source,
    write_main_v3_dry_run,
    write_pilot_v3_dry_run,
)

_V2_API_EXPORTS = (
    compact_judge_plan,
    estimate_judge_validation,
    finalize_judge_validation,
    plan_judge_validation,
    prepare_judge_validation,
    readiness_report,
    run_judge_dev,
    run_judge_holdout,
    summarize_judge_dev,
)

_V3_API_EXPORTS = (
    analyze_downstream_predictions,
    audit_completed_downstream,
    analyze_paired_pilot_v3,
    annotation_agreement,
    annotation_freeze_gate,
    downstream_readiness,
    estimate_downstream_audit,
    estimate_main_v3,
    estimate_pilot_v3,
    export_blinded_drift,
    freeze_annotations,
    lexical_drift_flags,
    main_v3_readiness,
    pilot_v3_readiness,
    plan_blinded_drift_export,
    plan_downstream_audit,
    plan_pilot_v3,
    plan_pilot_v3_judging,
    run_downstream_audit,
    run_pilot_v3_judge,
    run_pilot_v3_updater,
    validate_annotation_import,
    validate_paired_source,
    verify_approved_judge,
    write_downstream_dry_run,
    write_main_v3_dry_run,
    write_pilot_v3_dry_run,
    audit_extreme_pair_lineage,
    resolve_dataset_lineage,
    create_pre_main_freeze,
    estimate_final_primary,
    pre_main_readiness,
)


def find_project_root(start: str | Path | None = None) -> Path:
    """Locate the project from its root or notebooks directory."""
    origin = Path(start or Path.cwd()).resolve()
    candidates = (origin, *origin.parents)
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "rubric_order"
        ).is_dir():
            return candidate
    raise FileNotFoundError(
        f"could not locate the rubric-order project above {origin}; start Jupyter from the "
        "project root or notebooks directory"
    )


def load_experiment_config(
    config_path: str | Path,
    data_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load configuration and apply recorded dotted-key runtime overrides."""
    config = load_config(config_path, str(data_path) if data_path is not None else None)
    return apply_config_overrides(config, dict(overrides or {}))


def validate_dataset(config: dict[str, Any], preview_rows: int = 5) -> dict[str, Any]:
    train, validation = load_dataset(config)
    preview = [
        {
            "example_id": row["example_id"],
            "prompt_id": row["prompt_id"],
            "prompt": _truncate(row["prompt"]),
            "response": _truncate(row["response"]),
            "human_score": row["human_score"],
        }
        for row in train[:preview_rows]
    ]
    return {
        "dataset_path": str(Path(config["dataset"]["path"]).resolve()),
        "format": config["dataset"]["format"],
        "column_mapping": dict(config["dataset"]["columns"]),
        "splits": {
            config["dataset"]["train_split"]: len(train),
            **(
                {config["dataset"]["validation_split"]: len(validation)}
                if validation is not None
                else {}
            ),
        },
        "official_validation_available": validation is not None,
        "train_prompt_count": len({row["prompt_id"] for row in train}),
        "validation_prompt_count": (
            len({row["prompt_id"] for row in validation}) if validation is not None else None
        ),
        "train_label_counts": {
            score: sum(row["human_score"] == score for row in train) for score in range(5)
        },
        "validation_label_counts": (
            {score: sum(row["human_score"] == score for row in validation) for score in range(5)}
            if validation is not None
            else None
        ),
        "labels_are_integers_0_to_4": True,
        "relevant_fields": ["prompt", "response", "human_score"],
        "preview": preview,
    }


def prepare_experiment(
    config: dict[str, Any], run_id: str, *, resume: bool = True
) -> dict[str, Any]:
    """Prepare or safely reopen a prompt-grouped experimental run."""
    return prepare_run(config, run_id, resume=resume and run_directory(config, run_id).exists())


def inspect_protocol(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    _run_dir, manifest, blocks, heldout = load_prepared(config, run_id)
    block_summaries: list[dict[str, Any]] = []
    for block in blocks:
        labels = {row["example_id"]: row["human_score"] for row in block["examples"]}
        order_values = list(block["orders"].values())
        same_multiset = all(Counter(order) == Counter(order_values[0]) for order in order_values)
        reverse_ok = block["orders"]["reverse_random"] == list(reversed(block["orders"]["random"]))
        block_summaries.append(
            {
                "block_id": block["block_id"],
                "distinct_prompt_count": block["distinct_prompt_count"],
                "examples": [
                    {
                        "example_id": row["example_id"],
                        "human_score": row["human_score"],
                        "prompt": _truncate(row["prompt"], 80),
                        "response": _truncate(row["response"], 80),
                    }
                    for row in block["examples"]
                ],
                "orders": {
                    name: [
                        {
                            "position": position,
                            "example_id": example_id,
                            "human_score": labels[example_id],
                        }
                        for position, example_id in enumerate(order, start=1)
                    ]
                    for name, order in block["orders"].items()
                },
                "all_orders_same_multiset": same_multiset,
                "reverse_random_is_exact_reverse": reverse_ok,
            }
        )
    return {
        "manifest": manifest,
        "partition": manifest["partition"],
        "calibration": manifest["calibration"],
        "blocks": block_summaries,
        "heldout_prompt_count": len({row["prompt_id"] for row in heldout}),
        "heldout_response_count": len(heldout),
        "heldout_label_counts": dict(
            sorted(Counter(row["human_score"] for row in heldout).items())
        ),
        "zero_prompt_leakage": manifest["partition"]["prompt_leakage_count"] == 0,
        "all_blocks_disjoint": manifest["calibration"]["all_blocks_disjoint"],
    }


def estimate_experiment(
    config: dict[str, Any],
    run_id: str | None = None,
    *,
    methods: Sequence[str] | None = None,
    blocks: Sequence[int] | None = None,
    orders: Sequence[str] | None = None,
    replicates: Sequence[int] | None = None,
) -> dict[str, int]:
    selection = make_selection(methods, blocks, orders, replicates)
    heldout_count = None
    if run_id is not None:
        _run_dir, manifest, _blocks, _heldout = load_prepared(config, run_id)
        heldout_count = manifest["heldout_response_count"]
    return estimate_calls(config, selection, heldout_count)


def check_ollama(config: dict[str, Any], stage: str = "both") -> dict[str, Any]:
    if stage not in {"updater", "judge", "both"}:
        raise ValueError("stage must be updater, judge, or both")
    models = config["models"]
    required: list[str] = []
    if stage in {"updater", "both"}:
        required.append(models["updater"])
    if stage in {"judge", "both"}:
        required.append(models["judge"])
    result = OllamaClient(models["base_url"], models["timeout_seconds"]).health_and_models(required)
    result["required"] = required
    result["inference_performed"] = False
    return result


def execution_safety_status(
    config: dict[str, Any], run_id: str = "main-final-primary"
) -> dict[str, Any]:
    """Expose the persistent final-run safety guard through the public API."""
    return _execution_safety_status(config, run_id)


def installed_model_provenance(config: dict[str, Any]) -> dict[str, Any]:
    """Read installed Ollama metadata; this never pulls a missing model or runs inference."""
    models = config["models"]
    client = OllamaClient(models["base_url"], models["timeout_seconds"])
    result = client.model_inventory([models["updater"], models["judge"]])
    result["inference_performed"] = False
    result["models_downloaded"] = False
    return result


def plan_rubric_generation(
    config: dict[str, Any],
    run_id: str,
    *,
    methods: Sequence[str] | None = None,
    blocks: Sequence[int] | None = None,
    orders: Sequence[str] | None = None,
    replicates: Sequence[int] | None = None,
) -> dict[str, Any]:
    run_dir, _manifest, prepared_blocks, _heldout = load_prepared(config, run_id)
    selection = make_selection(methods, blocks, orders, replicates)
    jobs = plan_rubric_jobs(config, selection)
    block_sizes = {block["block_id"]: len(block["examples"]) for block in prepared_blocks}
    rows = [
        {
            "method": job.method,
            "block_id": job.block_id,
            "order": job.order,
            "replicate": job.replicate,
            "repeat_control": job.repeat_control,
            "updater_calls": job.updater_calls,
            "examples": block_sizes[job.block_id],
        }
        for job in jobs
    ]
    planned_calls = sum(row["updater_calls"] for row in rows)
    context_plan = plan_rubric_context(
        config,
        prepared_blocks,
        jobs,
        run_id=run_id,
        run_dir=run_dir,
    )
    return {
        "stage": "build-rubrics",
        "status": "ready" if context_plan["ready"] else "invalid",
        "model": config["models"]["updater"],
        "jobs": rows,
        "planned_calls": planned_calls,
        "estimated_max_generated_tokens": planned_calls * config["models"]["updater_max_tokens"],
        "selection": selection_dict(selection),
        "context": context_plan,
        **{
            key: context_plan[key]
            for key in (
                "ready",
                "planned_physical_updater_calls",
                "exact_requests",
                "deferred_requests",
                "cache_hits",
                "fresh_requests",
                "execution_fresh_calls",
                "cache_unavailable",
                "invalid_requests",
                "materialized_requests",
                "materialization_mismatches",
                "batch_exact_requests",
                "sequential_exact_requests",
                "sequential_deferred_requests",
            )
        },
    }


def build_rubrics(
    config: dict[str, Any],
    run_id: str,
    *,
    execute_model_calls: bool = False,
    max_calls: int = 0,
    methods: Sequence[str] | None = None,
    blocks: Sequence[int] | None = None,
    orders: Sequence[str] | None = None,
    replicates: Sequence[int] | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    plan = plan_rubric_generation(
        config,
        run_id,
        methods=methods,
        blocks=blocks,
        orders=orders,
        replicates=replicates,
    )
    if not execute_model_calls:
        run_dir = run_directory(config, run_id)
        dry_path = write_dry_run(
            run_dir,
            "build-rubrics",
            {**plan, "execution_requested": False, "max_calls": max_calls},
        )
        return {
            **plan,
            "executed": False,
            "actual_calls": 0,
            "dry_run_manifest": str(dry_path),
        }
    if not plan["ready"]:
        raise ValueError("rubric-generation preflight is not ready")
    execution_budget = _execution_budget(config, run_id, "build-rubrics", max_calls)
    _enforce_execution_budget(
        plan["execution_fresh_calls"],
        execution_budget["effective_max_calls"],
        "rubric generation fresh requests",
    )
    selection = make_selection(methods, blocks, orders, replicates)
    result = _build_rubrics(
        config,
        run_id,
        selection,
        execution_budget["effective_max_calls"],
        client=client,
    )
    final_plan = plan_rubric_generation(
        config,
        run_id,
        methods=methods,
        blocks=blocks,
        orders=orders,
        replicates=replicates,
    )
    write_json(
        run_directory(config, run_id)
        / "artifacts"
        / "request_plans"
        / "updater_final_manifest.json",
        final_plan,
    )
    return {
        **result,
        **final_plan,
        "executed": True,
        "execution_budget": execution_budget,
        "execution_cache_hits": result["cache_hits"],
        "preflight_before_execution": plan,
        "final_phase_plan": final_plan,
        "call_summary": summarize_calls(
            config,
            run_id,
            stage="build-rubrics",
            methods=methods,
            blocks=blocks,
            orders=orders,
            replicates=replicates,
        ),
    }


def plan_judging(
    config: dict[str, Any],
    run_id: str,
    *,
    methods: Sequence[str] | None = None,
    blocks: Sequence[int] | None = None,
    orders: Sequence[str] | None = None,
    replicates: Sequence[int] | None = None,
) -> dict[str, Any]:
    run_dir, _manifest, _blocks, heldout = load_prepared(config, run_id)
    selection = make_selection(methods, blocks, orders, replicates)
    expected = _expected_rubric_keys(config, selection)
    existing_rows = _selected_existing_rubrics(run_dir, selection)
    valid_rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for row in existing_rows:
        problem = _rubric_problem(run_dir, row)
        if problem is None:
            valid_rows.append(row)
        else:
            invalid.append({"key": _rubric_key(row), "error": problem})
    existing_keys = {_rubric_key(row) for row in valid_rows}
    missing = [
        dict(zip(_RUBRIC_KEY_FIELDS, key, strict=True)) for key in sorted(expected - existing_keys)
    ]
    expected_static = any(key[0] == "static" for key in expected)
    expected_non_static = sum(key[0] != "static" for key in expected)
    expected_physical = len(heldout) * (expected_non_static + int(expected_static))
    common = {
        "stage": "judge",
        "model": config["models"]["judge"],
        "selection": selection_dict(selection),
        "heldout_responses": len(heldout),
        "expected_rubrics": len(expected),
        "valid_rubrics": len(valid_rows),
        "missing_rubrics": missing,
        "invalid_rubrics": invalid,
        "expected_logical_predictions": len(expected) * len(heldout),
        "expected_physical_requests": expected_physical,
        "annotation_freeze": _annotation_freeze_for_judging(config, run_dir),
    }
    if not common["annotation_freeze"]["ready"]:
        return {
            **common,
            "status": "blocked_by_annotation_freeze",
            "ready": False,
            "requests": [],
            "exact_requests": 0,
            "logical_predictions": len(expected) * len(heldout),
            "physical_requests": 0,
            "planned_calls": expected_physical,
            "cache_hits": 0,
            "fresh_requests": 0,
            "cache_unavailable": 0,
            "invalid_requests": 0,
            "estimated_max_generated_tokens": (
                expected_physical * config["models"]["judge_max_tokens"]
            ),
        }
    if missing or invalid or not valid_rows:
        return {
            **common,
            "status": "blocked_by_rubric_generation",
            "ready": False,
            "requests": [],
            "exact_requests": 0,
            "logical_predictions": len(expected) * len(heldout),
            "physical_requests": 0,
            "planned_calls": expected_physical,
            "cache_hits": 0,
            "fresh_requests": 0,
            "cache_unavailable": 0,
            "invalid_requests": len(invalid),
            "estimated_max_generated_tokens": (
                expected_physical * config["models"]["judge_max_tokens"]
            ),
        }
    exact = plan_pilot_judge_requests(
        config,
        run_id=run_id,
        run_dir=run_dir,
        rubric_rows=valid_rows,
        heldout=heldout,
    )
    return {
        **common,
        **exact,
        "estimated_max_generated_tokens": (
            exact["planned_calls"] * config["models"]["judge_max_tokens"]
        ),
    }


def judge_responses(
    config: dict[str, Any],
    run_id: str,
    *,
    execute_model_calls: bool = False,
    max_calls: int = 0,
    allow_partial_execution: bool = False,
    methods: Sequence[str] | None = None,
    blocks: Sequence[int] | None = None,
    orders: Sequence[str] | None = None,
    replicates: Sequence[int] | None = None,
    client: Any | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    plan = plan_judging(
        config,
        run_id,
        methods=methods,
        blocks=blocks,
        orders=orders,
        replicates=replicates,
    )
    if not execute_model_calls:
        dry_path = write_dry_run(
            run_directory(config, run_id),
            "judge",
            {**plan, "execution_requested": False, "max_calls": max_calls},
        )
        return {
            **plan,
            "executed": False,
            "actual_calls": 0,
            "dry_run_manifest": str(dry_path),
        }
    if not plan["ready"]:
        raise ValueError(
            f"judging selection is not ready ({plan['status']}): "
            f"{len(plan['missing_rubrics'])} missing rubrics, "
            f"{len(plan['invalid_rubrics'])} invalid rubrics, annotation failures="
            f"{plan.get('annotation_freeze', {}).get('failures', [])}"
        )
    execution_budget = _execution_budget(config, run_id, "judge", max_calls)
    if allow_partial_execution:
        if execution_budget["effective_max_calls"] < 1:
            raise ValueError("partial judging execution requires MAX_CALLS >= 1")
    else:
        _enforce_execution_budget(
            plan["fresh_requests"],
            execution_budget["effective_max_calls"],
            "judging fresh requests",
        )
    selection = make_selection(methods, blocks, orders, replicates)
    result = _judge_rubrics(
        config,
        run_id,
        selection,
        execution_budget["effective_max_calls"],
        client=client,
        progress_callback=progress_callback,
    )
    final_plan = plan_judging(
        config,
        run_id,
        methods=methods,
        blocks=blocks,
        orders=orders,
        replicates=replicates,
    )
    write_json(
        run_directory(config, run_id) / "artifacts" / "request_plans" / "judge_final_manifest.json",
        final_plan,
    )
    return {
        **result,
        **final_plan,
        "executed": True,
        "status": result["status"],
        "execution_budget": execution_budget,
        "partial_execution_allowed": allow_partial_execution,
        "execution_cache_hits": result["cache_hits"],
        "preflight_before_execution": plan,
        "final_phase_plan": final_plan,
        "judgment_summary": summarize_judgments(
            config,
            run_id,
            methods=methods,
            blocks=blocks,
            orders=orders,
            replicates=replicates,
        ),
    }


def initial_rubric(config: dict[str, Any]) -> dict[str, Any]:
    value = read_json(resolve_project_path(config, config["rubric"]["initial"]))
    if config.get("schema_version") == "2.0":
        return validate_rubric_v2(value, value)
    return validate_rubric(value)


def inspect_rubric_trajectory(
    config: dict[str, Any],
    run_id: str,
    *,
    method: str,
    block: int,
    order: str,
    replicate: int = 0,
) -> dict[str, Any]:
    run_dir, _manifest, _blocks, _heldout = load_prepared(config, run_id)
    trajectory_dir = (
        run_dir
        / "artifacts"
        / "rubrics"
        / method
        / f"block_{block:02d}"
        / order
        / f"replicate_{replicate}"
    )
    calls = get_call_records(
        config,
        run_id,
        stage="build-rubrics",
        methods=[method],
        blocks=[block],
        orders=[order],
        replicates=[replicate],
    )
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in calls:
        if row["step"] is not None:
            by_step[int(row["step"])].append(row)
    steps: list[dict[str, Any]] = []
    paths = (
        [path for path in sorted(trajectory_dir.glob("step_*.json")) if ".meta." not in path.name]
        if trajectory_dir.exists()
        else []
    )
    for path in paths:
        step = int(path.stem.split("_")[1])
        rubric = _validate_configured_rubric(config, read_json(path))
        records = by_step.get(step, [])
        successful = next((row for row in reversed(records) if row["status"] == "success"), None)
        steps.append(
            {
                "step": step,
                "rubric_path": str(path),
                "characters": len(json.dumps(rubric, ensure_ascii=False, sort_keys=True)),
                "criteria_count": len(rubric["criteria"]),
                "retry_count": sum(int(row["attempt"]) > 0 for row in records),
                "prompt_eval_count": successful["prompt_eval_count"] if successful else None,
                "eval_count": successful["eval_count"] if successful else None,
                "latency_seconds": successful["latency_seconds"] if successful else None,
                "thinking_detected": any(row["thinking_detected"] for row in records),
                "unexpected_output_detected": any(
                    row["unexpected_output_detected"] for row in records
                ),
            }
        )
    final = read_json(steps[-1]["rubric_path"]) if steps else None
    return {
        "method": method,
        "block_id": block,
        "order": order,
        "replicate": replicate,
        "steps": steps,
        "final_rubric": final,
        "complete": bool(steps),
    }


def get_call_records(
    config: dict[str, Any],
    run_id: str,
    *,
    stage: str | None = None,
    methods: Sequence[str] | None = None,
    blocks: Sequence[int] | None = None,
    orders: Sequence[str] | None = None,
    replicates: Sequence[int] | None = None,
    include_raw: bool = False,
) -> list[dict[str, Any]]:
    database = run_directory(config, run_id) / "calls.sqlite3"
    if not database.exists():
        return []
    selection = make_selection(methods, blocks, orders, replicates)
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro&immutable=1", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM calls ORDER BY id").fetchall()
    finally:
        connection.close()
    records: list[dict[str, Any]] = []
    for sqlite_row in rows:
        row = dict(sqlite_row)
        if stage is not None and row["stage"] != stage:
            continue
        if not _call_selected(row, selection):
            continue
        raw = row.pop("raw_response")
        parsed = row.pop("parsed_response")
        stripped = raw.strip() if isinstance(raw, str) else ""
        row["thinking_detected"] = "<think" in stripped.casefold()
        row["unexpected_output_detected"] = bool(
            row["status"] != "success"
            or (stripped and not (stripped.startswith("{") and stripped.endswith("}")))
        )
        if include_raw:
            row["raw_response"] = raw
            row["parsed_response"] = json.loads(parsed) if parsed else None
        records.append(row)
    return records


def summarize_calls(
    config: dict[str, Any],
    run_id: str,
    *,
    stage: str | None = None,
    methods: Sequence[str] | None = None,
    blocks: Sequence[int] | None = None,
    orders: Sequence[str] | None = None,
    replicates: Sequence[int] | None = None,
) -> dict[str, Any]:
    records = get_call_records(
        config,
        run_id,
        stage=stage,
        methods=methods,
        blocks=blocks,
        orders=orders,
        replicates=replicates,
    )
    latencies = [float(row["latency_seconds"]) for row in records]
    return {
        "call_records": len(records),
        "successful": sum(row["status"] == "success" for row in records),
        "failed": sum(row["status"] != "success" for row in records),
        "schema_failures": sum(row["status"] == "validation_error" for row in records),
        "request_failures": sum(row["status"] == "request_error" for row in records),
        "retries": sum(int(row["attempt"]) > 0 for row in records),
        "prompt_eval_tokens": sum(int(row["prompt_eval_count"] or 0) for row in records),
        "eval_tokens": sum(int(row["eval_count"] or 0) for row in records),
        "latency_seconds": _latency_summary(latencies),
        "thinking_detected": any(row["thinking_detected"] for row in records),
        "unexpected_output_detected": any(row["unexpected_output_detected"] for row in records),
    }


def summarize_judgments(
    config: dict[str, Any],
    run_id: str,
    *,
    methods: Sequence[str] | None = None,
    blocks: Sequence[int] | None = None,
    orders: Sequence[str] | None = None,
    replicates: Sequence[int] | None = None,
    sample_size: int = 8,
) -> dict[str, Any]:
    run_dir, _manifest, _blocks, heldout = load_prepared(config, run_id)
    selection = make_selection(methods, blocks, orders, replicates)
    rubrics = _selected_existing_rubrics(run_dir, selection)
    expected_predictions = len(rubrics) * len(heldout)
    prediction_path = run_dir / "artifacts" / "predictions.jsonl"
    predictions = (
        [row for row in read_jsonl(prediction_path) if _prediction_selected(row, selection)]
        if prediction_path.exists()
        else []
    )
    call_summary = summarize_calls(
        config,
        run_id,
        stage="judge",
        methods=methods,
        blocks=blocks,
        orders=orders,
        replicates=replicates,
    )
    return {
        "expected_predictions": expected_predictions,
        "completed": len(predictions),
        "cached": sum(bool(row.get("cache_hit", False)) for row in predictions),
        "failed": call_summary["failed"],
        "remaining": max(0, expected_predictions - len(predictions)),
        "score_distribution": dict(
            sorted(Counter(int(row["predicted_score"]) for row in predictions).items())
        ),
        **{
            key: call_summary[key]
            for key in (
                "schema_failures",
                "request_failures",
                "retries",
                "prompt_eval_tokens",
                "eval_tokens",
                "latency_seconds",
            )
        },
        "sample": [
            {
                "prompt_id": row["prompt_id"],
                "human_score": row["human_score"],
                "predicted_score": row["predicted_score"],
                "method": row["method"],
                "block_id": row["block_id"],
                "order": row["order"],
                "replicate": row["replicate"],
            }
            for row in predictions[:sample_size]
        ],
    }


def experiment_completeness(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    run_dir, _manifest, _blocks, heldout = load_prepared(config, run_id)
    selection = Selection()
    expected_rubrics = _expected_rubric_keys(config, selection)
    existing_rows = _selected_existing_rubrics(run_dir, selection)
    existing_rubric_keys = [_rubric_key(row) for row in existing_rows]
    duplicate_rubrics = [key for key, count in Counter(existing_rubric_keys).items() if count > 1]
    valid_keys: set[tuple[Any, ...]] = set()
    invalid: list[dict[str, Any]] = []
    for row in existing_rows:
        problem = _rubric_problem(run_dir, row)
        if problem is None:
            valid_keys.add(_rubric_key(row))
        else:
            invalid.append({"key": _rubric_key(row), "error": problem})
    missing_rubrics = [
        dict(zip(_RUBRIC_KEY_FIELDS, key, strict=True))
        for key in sorted(expected_rubrics - valid_keys)
    ]
    prediction_path = run_dir / "artifacts" / "predictions.jsonl"
    predictions = read_jsonl(prediction_path) if prediction_path.exists() else []
    prediction_key_rows = [
        (*_rubric_key(row), row["example_id"])
        for row in predictions
        if _rubric_key(row) in expected_rubrics
    ]
    predicted_keys = set(prediction_key_rows)
    duplicate_predictions = [
        key for key, count in Counter(prediction_key_rows).items() if count > 1
    ]
    missing_prediction_groups: list[dict[str, Any]] = []
    for rubric_key in sorted(expected_rubrics):
        missing_count = sum(
            (*rubric_key, example["example_id"]) not in predicted_keys for example in heldout
        )
        if missing_count:
            missing_prediction_groups.append(
                {
                    **dict(zip(_RUBRIC_KEY_FIELDS, rubric_key, strict=True)),
                    "missing_responses": missing_count,
                }
            )
    expected_prediction_count = len(expected_rubrics) * len(heldout)
    missing_prediction_count = sum(
        group["missing_responses"] for group in missing_prediction_groups
    )
    return {
        "complete": (
            not missing_rubrics
            and not invalid
            and not duplicate_rubrics
            and not duplicate_predictions
            and missing_prediction_count == 0
        ),
        "expected_rubrics": len(expected_rubrics),
        "valid_rubrics": len(valid_keys),
        "missing_rubrics": missing_rubrics,
        "invalid_rubrics": invalid,
        "duplicate_rubrics": duplicate_rubrics,
        "expected_predictions": expected_prediction_count,
        "completed_predictions": expected_prediction_count - missing_prediction_count,
        "missing_prediction_count": missing_prediction_count,
        "missing_prediction_groups": missing_prediction_groups,
        "duplicate_predictions": duplicate_predictions,
    }


def analyze_experiment(
    config: dict[str, Any],
    run_id: str,
    *,
    allow_partial_results: bool = False,
) -> dict[str, Any]:
    if config.get("schema_version") in {"2.0", "3.0"} and allow_partial_results:
        raise ValueError("v2/v3 analysis cannot approve or publish partial results")
    completeness = experiment_completeness(config, run_id)
    if not completeness["complete"] and not allow_partial_results:
        raise ValueError(
            "experiment is incomplete; inspect experiment_completeness or set "
            "allow_partial_results=True explicitly. "
            f"Missing rubrics: {len(completeness['missing_rubrics'])}; "
            f"invalid rubrics: {len(completeness['invalid_rubrics'])}; "
            f"missing predictions: {completeness['missing_prediction_count']}"
        )
    summary = analyze_run(config, run_id)
    summary["completeness"] = completeness
    summary["partial_results_allowed"] = allow_partial_results
    write_json(run_directory(config, run_id) / "metrics" / "summary.json", summary)
    return summary


def resolved_configuration(config: dict[str, Any]) -> dict[str, Any]:
    return public_config(config)


def experiment_directory(config: dict[str, Any], run_id: str) -> Path:
    return run_directory(config, run_id)


def metric_tables(config: dict[str, Any], run_id: str) -> dict[str, list[dict[str, str]]]:
    metrics_dir = run_directory(config, run_id) / "metrics"
    names = (
        "primary_metrics",
        "agreement_metrics",
        "paired_differences",
        "repeat_control",
        "rubric_text",
        "rubric_similarity",
    )
    return {
        name: _read_csv_records(metrics_dir / f"{name}.csv")
        for name in names
        if (metrics_dir / f"{name}.csv").exists()
    }


def no_inference_smoke_test(output: str | Path) -> dict[str, Any]:
    """Prepare synthetic data and estimates without constructing an Ollama client."""
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"smoke output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_path = output_path.parent / f".{output_path.name}.synthetic.jsonl"
    rows = [
        {
            "prompt": f"calibration prompt {score}-{index}",
            "response": f"synthetic response at score {score}",
            "helpfulness": score,
        }
        for score in range(5)
        for index in range(8)
    ]
    rows.extend(
        {
            "prompt": f"paired prompt {prompt_index}",
            "response": f"paired response {response_index}",
            "helpfulness": (prompt_index + response_index) % 5,
        }
        for prompt_index in range(5)
        for response_index in range(2)
    )
    write_jsonl(data_path, rows)
    config = copy.deepcopy(DEFAULTS)
    config["dataset"]["path"] = str(data_path)
    config["sampling"].update(
        {"calibration_blocks": 1, "heldout_prompts": 3, "responses_per_prompt": 2}
    )
    config["analysis"].update({"bootstrap_repetitions": 5, "make_plots": False})
    config["repeat_control"]["enabled"] = False
    config["output_root"] = str(output_path.parent)
    config["_config_path"] = "<synthetic-smoke>"
    config["_project_root"] = str(Path.cwd().resolve())
    config["_runtime_overrides"] = {}
    try:
        manifest = prepare_experiment(config, output_path.name, resume=False)
        estimate = estimate_experiment(config, output_path.name)
        dry_path = write_dry_run(
            output_path,
            "complete-smoke",
            {
                "estimate": estimate,
                "synthetic": True,
                "ollama_contacted": False,
                "sqlite_call_store_created": False,
            },
        )
    finally:
        if data_path.exists():
            data_path.unlink()
    calls_database = output_path / "calls.sqlite3"
    if calls_database.exists():
        raise AssertionError("smoke test unexpectedly created a model-call database")
    return {
        "status": "ok",
        "run_dir": str(output_path),
        "manifest": str(output_path / "manifest.json"),
        "dry_run_manifest": str(dry_path),
        "estimated_calls": estimate,
        "model_calls_made": manifest["model_calls_made"],
        "ollama_contacted": False,
    }


def no_inference_v3_smoke_test(output: str | Path) -> dict[str, Any]:
    """Exercise constrained planning, materialization, and blinding without a model client."""
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"v3 smoke output already exists: {output_path}")
    root = find_project_root()
    config = load_experiment_config(root / "configs" / "pilot_v3.yaml")
    config = copy.deepcopy(config)
    config["sampling"]["calibration_blocks"] = 1
    config["repeat_control"]["enabled"] = False
    examples = [
        {
            "example_id": f"synthetic-{score}-{replicate}",
            "prompt_id": f"prompt-{score}-{replicate}",
            "prompt": f"Synthetic request {score}-{replicate}",
            "response": f"Synthetic response {score}-{replicate}",
            "human_score": score,
        }
        for score in range(5)
        for replicate in range(2)
    ]
    identifiers = [row["example_id"] for row in examples]
    block = {
        "block_id": 0,
        "examples": examples,
        "orders": {
            "ascending": identifiers,
            "descending": list(reversed(identifiers)),
            "random": identifiers[::2] + identifiers[1::2],
            "reverse_random": list(reversed(identifiers[::2] + identifiers[1::2])),
        },
    }
    plan = plan_inventory_requests(config, run_id="synthetic-v3", blocks=[block])
    inventory = load_inventory(config)
    initial = read_json(root / "rubrics" / "initial_helpfulness_v3.json")
    selection = validate_inventory_selection(
        {"adaptive_criterion_ids": ["appropriate_level_of_detail"]}, inventory
    )
    rubric = materialize_inventory_rubric(initial, inventory, selection)
    source = output_path / "annotation_source"
    rubric_path = source / "artifacts" / "rubrics" / "static" / "initial.json"
    write_json(rubric_path, rubric)
    raw_hash = hashlib.sha256(rubric_path.read_bytes()).hexdigest()
    canonical_hash = hashlib.sha256(
        json.dumps(rubric, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    write_jsonl(
        source / "artifacts" / "rubrics.jsonl",
        [
            {
                "method": "synthetic",
                "block_id": 0,
                "order": "blinded",
                "replicate": 0,
                "repeat_control": False,
                "rubric_path": rubric_path.relative_to(source).as_posix(),
                "rubric_sha256": raw_hash,
                "canonical_rubric_sha256": canonical_hash,
                "schema_version": "3.0",
            }
        ],
    )
    blind = plan_blinded_drift_export(source)
    result = {
        "status": "ok",
        "synthetic": True,
        "planned_updater_calls": plan["planned_physical_updater_calls"],
        "invalid_requests": plan["invalid_requests"],
        "materialized_adaptive_ids": rubric["adaptive_criterion_ids"],
        "blinded_items": blind["blinded_item_count"],
        "ollama_contacted": False,
        "sqlite_call_store_created": False,
    }
    write_json(output_path / "smoke_result.json", result)
    if list(output_path.rglob("calls.sqlite3")):
        raise AssertionError("v3 smoke unexpectedly created a call database")
    return result


def make_selection(
    methods: Sequence[str] | None = None,
    blocks: Sequence[int] | None = None,
    orders: Sequence[str] | None = None,
    replicates: Sequence[int] | None = None,
) -> Selection:
    method_set = frozenset(methods) if methods is not None else None
    block_set = frozenset(int(value) for value in blocks) if blocks is not None else None
    order_set = frozenset(orders) if orders is not None else None
    replicate_set = (
        frozenset(int(value) for value in replicates) if replicates is not None else None
    )
    if method_set is not None and not method_set <= {"static", "batch", "sequential"}:
        raise ValueError(f"unknown methods: {sorted(method_set)}")
    configured_orders = {"ascending", "descending", "random", "reverse_random"}
    if order_set is not None and not order_set <= configured_orders:
        raise ValueError(f"unknown orders: {sorted(order_set)}")
    if block_set is not None and (not block_set or min(block_set) < 0):
        raise ValueError("blocks must contain non-negative integers")
    if replicate_set is not None and (not replicate_set or min(replicate_set) < 0):
        raise ValueError("replicates must contain non-negative integers")
    return Selection(block_set, order_set, method_set, replicate_set)


def selection_dict(selection: Selection) -> dict[str, list[Any] | None]:
    return {
        "methods": sorted(selection.methods) if selection.methods is not None else None,
        "blocks": sorted(selection.blocks) if selection.blocks is not None else None,
        "orders": sorted(selection.orders) if selection.orders is not None else None,
        "replicates": sorted(selection.replicates) if selection.replicates is not None else None,
    }


_RUBRIC_KEY_FIELDS = ("method", "block_id", "order", "replicate")


def _expected_rubric_keys(config: dict[str, Any], selection: Selection) -> set[tuple[Any, ...]]:
    keys = {
        (job.method, job.block_id, job.order, job.replicate)
        for job in plan_rubric_jobs(config, selection)
    }
    if selection.allows_method("static") and selection.allows_replicate(0):
        for block_id in range(config["sampling"]["calibration_blocks"]):
            if not selection.allows_block(block_id):
                continue
            for order in config["orders"]:
                if selection.allows_order(order):
                    keys.add(("static", block_id, order, 0))
    return keys


def _selected_existing_rubrics(run_dir: Path, selection: Selection) -> list[dict[str, Any]]:
    index_path = run_dir / "artifacts" / "rubrics.jsonl"
    if not index_path.exists():
        return []
    return [
        row
        for row in read_jsonl(index_path)
        if selection.allows_method(row["method"])
        and selection.allows_block(row["block_id"])
        and selection.allows_order(row["order"])
        and selection.allows_replicate(row["replicate"])
    ]


def _rubric_problem(run_dir: Path, row: dict[str, Any]) -> str | None:
    path = run_dir / row["rubric_path"]
    if not path.exists():
        return f"missing file: {path}"
    try:
        value = read_json(path)
        if value.get("schema_version") == "2.0":
            initial_path = run_dir / "manifest.json"
            manifest = read_json(initial_path)
            initial = manifest.get("initial_rubric", {}).get("content")
            if not isinstance(initial, dict):
                return "v2 manifest is missing the immutable initial rubric"
            validate_rubric_v2(value, initial)
        else:
            validate_rubric(value)
    except (ValueError, json.JSONDecodeError) as exc:
        return f"invalid rubric: {exc}"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != row["rubric_sha256"]:
        return "rubric hash does not match the stored index"
    canonical_digest = hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if canonical_digest != row.get("canonical_rubric_sha256"):
        return "canonical rubric hash does not match the stored index"
    return None


def _validate_configured_rubric(config: dict[str, Any], value: Any) -> dict[str, Any]:
    if config.get("schema_version") == "2.0":
        initial = read_json(resolve_project_path(config, config["rubric"]["initial"]))
        return validate_rubric_v2(value, initial)
    return validate_rubric(value)


def _rubric_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in _RUBRIC_KEY_FIELDS)


def _call_selected(row: Mapping[str, Any], selection: Selection) -> bool:
    block_id = row["block_id"]
    order = row["order_name"]
    return (
        selection.allows_method(row["method"])
        and (block_id is None or selection.allows_block(int(block_id)))
        and (order is None or selection.allows_order(str(order)))
        and selection.allows_replicate(int(row["replicate"]))
    )


def _prediction_selected(row: Mapping[str, Any], selection: Selection) -> bool:
    return (
        selection.allows_method(row["method"])
        and selection.allows_block(int(row["block_id"]))
        and selection.allows_order(row["order"])
        and selection.allows_replicate(int(row["replicate"]))
    )


def _enforce_execution_budget(planned_calls: int, max_calls: int, stage: str) -> None:
    if max_calls < planned_calls:
        raise ValueError(
            f"{stage} plans {planned_calls} calls but MAX_CALLS is {max_calls}; execution refused"
        )


def _execution_budget(
    config: dict[str, Any], run_id: str, stage: str, requested_max_calls: int
) -> dict[str, Any]:
    """Cap a resumable execution against persistent stage and run-wide call ceilings."""
    safety = config.get("execution_safety")
    if not isinstance(safety, dict):
        return {
            "requested_max_calls": requested_max_calls,
            "effective_max_calls": requested_max_calls,
            "prior_total_calls": 0,
            "prior_stage_calls": 0,
            "remaining_total_calls": None,
            "remaining_stage_calls": None,
            "hard_total_call_ceiling": None,
            "stage_call_ceiling": None,
        }
    if stage not in safety["stage_call_ceilings"]:
        raise ValueError(f"execution safety has no ceiling for stage {stage!r}")
    database = run_directory(config, run_id) / "calls.sqlite3"
    prior_total = 0
    prior_stage = 0
    if database.exists():
        connection = sqlite3.connect(
            f"file:{database.resolve().as_posix()}?mode=ro&immutable=1", uri=True
        )
        try:
            prior_total = int(connection.execute("SELECT COUNT(*) FROM calls").fetchone()[0])
            prior_stage = int(
                connection.execute(
                    "SELECT COUNT(*) FROM calls WHERE stage = ?", (stage,)
                ).fetchone()[0]
            )
        finally:
            connection.close()
    total_ceiling = int(safety["hard_total_call_ceiling"])
    stage_ceiling = int(safety["stage_call_ceilings"][stage])
    remaining_total = max(0, total_ceiling - prior_total)
    remaining_stage = max(0, stage_ceiling - prior_stage)
    effective = min(requested_max_calls, remaining_total, remaining_stage)
    return {
        "requested_max_calls": requested_max_calls,
        "effective_max_calls": effective,
        "prior_total_calls": prior_total,
        "prior_stage_calls": prior_stage,
        "remaining_total_calls": remaining_total,
        "remaining_stage_calls": remaining_stage,
        "hard_total_call_ceiling": total_ceiling,
        "stage_call_ceiling": stage_ceiling,
    }


def _annotation_freeze_for_judging(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    settings = config.get("drift_annotations")
    if not isinstance(settings, dict) or not settings.get("required_before_judging", False):
        return {"ready": True, "required": False, "failures": []}
    relative = settings.get(
        "freeze_relative_path", "artifacts/drift_annotations/annotation_freeze.json"
    )
    freeze_path = run_dir / relative
    judging_started_at = None
    database = run_dir / "calls.sqlite3"
    if database.exists():
        connection = sqlite3.connect(
            f"file:{database.resolve().as_posix()}?mode=ro&immutable=1", uri=True
        )
        try:
            judging_started_at = connection.execute(
                "SELECT MIN(timestamp) FROM calls WHERE stage = 'judge'"
            ).fetchone()[0]
        finally:
            connection.close()
    gate = annotation_freeze_gate(
        freeze_path,
        judging_started_at=judging_started_at,
        minimum_annotators=int(settings.get("minimum_annotators", 1)),
    )
    return {
        **gate,
        "required": True,
        "path": str(freeze_path),
        "judging_started_at": judging_started_at,
    }


def _latency_summary(values: Iterable[float]) -> dict[str, float | None]:
    items = list(values)
    return {
        "total": sum(items),
        "mean": statistics.fmean(items) if items else None,
        "minimum": min(items) if items else None,
        "maximum": max(items) if items else None,
    }


def _truncate(value: str, length: int = 120) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= length else compact[: length - 1] + "…"


def _read_csv_records(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
