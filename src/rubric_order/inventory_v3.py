from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from .config import resolve_project_path
from .context import preflight_context, sha256_text
from .estimate import RubricJob, Selection, plan_rubric_jobs
from .ollama import InferenceBudgetClient, OllamaClient, updater_options
from .prompts import PromptTemplate, load_template
from .run import merge_index, read_json, sync_manifest_model_calls, write_json
from .sampling import ordered_examples
from .storage import (
    CallContext,
    CallStore,
    cached_validated_call,
    make_request_hash,
    read_only_cache_status,
)


def load_inventory(config: dict[str, Any]) -> dict[str, Any]:
    configured = config.get("inventory_v3", {}).get("path")
    if not configured:
        raise ValueError("inventory-constrained workflow requires inventory_v3.path")
    value = read_json(resolve_project_path(config, configured))
    if value.get("schema_version") != "3.0" or value.get("inventory_version") != "1.0":
        raise ValueError("unsupported adaptive criterion inventory version")
    criteria = value.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("adaptive criterion inventory must contain criteria")
    required = {
        "id",
        "active",
        "name",
        "description",
        "positive_signals",
        "negative_signals",
        "overlap_review",
    }
    identifiers: list[str] = []
    for index, criterion in enumerate(criteria):
        if not isinstance(criterion, dict) or set(criterion) != required:
            raise ValueError(f"inventory criterion {index} has an invalid contract")
        identifier = criterion["id"]
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"inventory criterion {index} has an invalid ID")
        if not isinstance(criterion["active"], bool):
            raise ValueError(f"inventory criterion {identifier} has invalid active state")
        identifiers.append(identifier)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("adaptive criterion inventory IDs must be unique")
    return deepcopy(value)


def load_initial_inventory_rubric(config: dict[str, Any]) -> dict[str, Any]:
    initial = read_json(_initial_path(config))
    _assert_initial_v3(initial)
    reference_path = config.get("inventory_v3", {}).get("core_reference")
    if reference_path:
        reference = read_json(resolve_project_path(config, reference_path))
        if (
            initial["title"] != reference.get("title")
            or initial["criteria"][:3] != reference.get("criteria", [])[:3]
            or initial["score_anchors"] != reference.get("score_anchors")
        ):
            raise ValueError("v3 title, core criteria, or anchors differ from frozen v2 reference")
    return initial


def active_inventory(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    return [deepcopy(row) for row in inventory["criteria"] if row["active"]]


def inventory_selection_schema(inventory: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["adaptive_criterion_ids"],
        "additionalProperties": False,
        "properties": {
            "adaptive_criterion_ids": {
                "type": "array",
                "minItems": 0,
                "maxItems": 2,
                "uniqueItems": True,
                "items": {"type": "string", "enum": active_inventory_ids(inventory)},
            }
        },
    }


def active_inventory_ids(inventory: dict[str, Any]) -> list[str]:
    return [row["id"] for row in inventory["criteria"] if row["active"]]


def validate_inventory_selection(value: Any, inventory: dict[str, Any]) -> dict[str, list[str]]:
    if not isinstance(value, dict) or set(value) != {"adaptive_criterion_ids"}:
        raise ValueError("selection must contain exactly adaptive_criterion_ids")
    identifiers = value["adaptive_criterion_ids"]
    if not isinstance(identifiers, list) or any(not isinstance(item, str) for item in identifiers):
        raise ValueError("adaptive_criterion_ids must be a list of strings")
    if len(identifiers) > 2:
        raise ValueError("at most two adaptive criterion IDs may be selected")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate adaptive criterion IDs are not permitted")
    order = {identifier: index for index, identifier in enumerate(active_inventory_ids(inventory))}
    unknown = [identifier for identifier in identifiers if identifier not in order]
    if unknown:
        raise ValueError(f"unknown or inactive adaptive criterion IDs: {sorted(unknown)}")
    return {"adaptive_criterion_ids": sorted(identifiers, key=order.__getitem__)}


def parse_inventory_selection(raw: str, inventory: dict[str, Any]) -> dict[str, list[str]]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"response is not valid JSON: {exc.msg}") from exc
    return validate_inventory_selection(value, inventory)


def materialize_inventory_rubric(
    initial: dict[str, Any],
    inventory: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    clean = validate_inventory_selection(selection, inventory)
    _assert_initial_v3(initial)
    definitions = {row["id"]: row for row in active_inventory(inventory)}
    adaptive = [
        {
            "name": definitions[identifier]["name"],
            "description": definitions[identifier]["description"],
            "positive_signals": deepcopy(definitions[identifier]["positive_signals"]),
            "negative_signals": deepcopy(definitions[identifier]["negative_signals"]),
        }
        for identifier in clean["adaptive_criterion_ids"]
    ]
    return {
        "schema_version": "3.0",
        "title": initial["title"],
        "criteria": deepcopy(initial["criteria"][:3]) + adaptive,
        "adaptive_criterion_ids": clean["adaptive_criterion_ids"],
        "score_anchors": deepcopy(initial["score_anchors"]),
    }


def validate_materialized_inventory_rubric(
    value: Any, initial: dict[str, Any], inventory: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("materialized v3 rubric must be an object")
    expected = materialize_inventory_rubric(
        initial,
        inventory,
        {"adaptive_criterion_ids": value.get("adaptive_criterion_ids")},
    )
    if value != expected:
        raise ValueError("materialized v3 rubric differs from server-side inventory definitions")
    return expected


def build_inventory_request(
    config: dict[str, Any],
    *,
    run_id: str,
    job: RubricJob,
    step: int,
    state: dict[str, Any],
    examples: list[dict[str, Any]],
    template: PromptTemplate,
    inventory: dict[str, Any],
    initial: dict[str, Any],
) -> dict[str, Any]:
    clean_state = validate_inventory_selection(state, inventory)
    schema = inventory_selection_schema(inventory)
    public_inventory = [
        {
            key: row[key]
            for key in (
                "id",
                "name",
                "description",
                "positive_signals",
                "negative_signals",
            )
        }
        for row in active_inventory(inventory)
    ]
    if job.method == "batch":
        if step != 1:
            raise ValueError("constrained batch has exactly one step")
        prompt = template.render(
            state_json=_compact(clean_state),
            inventory_json=_compact(public_inventory),
            examples_json=_compact([_model_example(row) for row in examples]),
            schema_json=_compact(schema),
        )
    elif job.method == "sequential":
        if len(examples) != 1:
            raise ValueError("constrained sequential requires exactly one example")
        prompt = template.render(
            state_json=_compact(clean_state),
            inventory_json=_compact(public_inventory),
            example_json=_compact(_model_example(examples[0])),
            schema_json=_compact(schema),
        )
    else:
        raise ValueError(f"unsupported constrained updater method: {job.method}")
    options = updater_options(config)
    namespace = f"inventory-v3:{job.method}:{job.block_id}:{job.order}:{job.replicate}"
    context = preflight_context(
        prompt,
        context_length=config["models"]["context_length"],
        max_output_tokens=config["models"]["updater_max_tokens"],
        safety_margin_tokens=config["context"]["safety_margin_tokens"],
        method=f"constrained_{job.method}",
        block_id=job.block_id,
        order=job.order,
        reject=False,
    )
    previous_rubric = materialize_inventory_rubric(initial, inventory, clean_state)
    return {
        "run_id": run_id,
        "stage": "build-constrained-rubrics",
        "method": f"constrained_{job.method}",
        "base_method": job.method,
        "block_id": job.block_id,
        "order": job.order,
        "replicate": job.replicate,
        "repeat_control": job.repeat_control,
        "step": step,
        "example_id": examples[0]["example_id"] if len(examples) == 1 else None,
        "prompt_id": examples[0]["prompt_id"] if len(examples) == 1 else None,
        "example_ids": [row["example_id"] for row in examples],
        "prompt_ids": [row["prompt_id"] for row in examples],
        "rendered_prompt": prompt,
        "rendered_prompt_hash": sha256_text(prompt),
        "model": config["models"]["updater"],
        "model_options": options,
        "response_schema": schema,
        "cache_namespace": namespace,
        "request_status": "exact",
        "prompt_template_hash": template.sha256,
        "previous_state_hash": _object_hash(clean_state),
        "input_rubric_hash": _object_hash(previous_rubric),
        "rendered_rubric_hash": _object_hash(previous_rubric),
        "schema_hash": _object_hash(schema),
        "model_options_hash": _object_hash(options),
        "estimated_input_tokens": context["estimated_input_tokens"],
        "context_length": context["context_length"],
        "reserved_output_tokens": context["max_output_tokens"],
        "safety_margin_tokens": context["safety_margin_tokens"],
        "estimated_context_headroom": context["estimated_headroom_tokens"],
        "context_preflight_ok": context["allowed"],
        "context_preflight": context,
        "request_hash": make_request_hash(
            config["models"]["updater"],
            prompt,
            options,
            schema,
            template.sha256,
            namespace,
        ),
    }


def plan_inventory_requests(
    config: dict[str, Any],
    *,
    run_id: str,
    blocks: list[dict[str, Any]],
    selection: Selection | None = None,
    state_directory: Path | None = None,
    cache_database: Path | None = None,
) -> dict[str, Any]:
    selected = selection or Selection()
    jobs = plan_rubric_jobs(config, selected)
    inventory = load_inventory(config)
    initial = load_initial_inventory_rubric(config)
    batch_template = load_template(config, "batch")
    sequential_template = load_template(config, "sequential")
    empty = {"adaptive_criterion_ids": []}
    rows: list[dict[str, Any]] = []
    for job in jobs:
        examples = ordered_examples(blocks[job.block_id], job.order)
        if job.method == "batch":
            requests = [
                build_inventory_request(
                    config,
                    run_id=run_id,
                    job=job,
                    step=1,
                    state=empty,
                    examples=examples,
                    template=batch_template,
                    inventory=inventory,
                    initial=initial,
                )
            ]
        else:
            requests = []
            state: dict[str, Any] | None = empty
            for step, example in enumerate(examples, start=1):
                if step > 1:
                    path = _state_path(state_directory, job, step - 1)
                    state = read_json(path) if path is not None and path.exists() else None
                if state is None:
                    rows.append(
                        _deferred_inventory_request(
                            config,
                            run_id,
                            job,
                            step,
                            example,
                            sequential_template,
                            inventory,
                            initial,
                        )
                    )
                    continue
                requests.append(
                    build_inventory_request(
                        config,
                        run_id=run_id,
                        job=job,
                        step=step,
                        state=state,
                        examples=[example],
                        template=sequential_template,
                        inventory=inventory,
                        initial=initial,
                    )
                )
        for request in requests:
            status = (
                read_only_cache_status(cache_database, request["request_hash"])
                if cache_database is not None
                else "fresh"
            )
            rows.append(sanitize_inventory_request(request, status))
    exact = [row for row in rows if row["request_status"] == "exact"]
    deferred = [row for row in rows if row["request_status"] == "deferred"]
    invalid = [row for row in rows if not row["context_preflight_ok"]]
    return {
        "stage": "build-constrained-rubrics",
        "status": "ready" if not invalid else "invalid",
        "ready": not invalid,
        "requests": rows,
        "exact_requests": len(exact),
        "deferred_requests": len(deferred),
        "planned_physical_updater_calls": len(rows),
        "cache_hits": sum(row["cache_status"] == "hit" for row in exact),
        "fresh_requests": sum(row["cache_status"] == "fresh" for row in exact),
        "execution_fresh_calls": sum(row["cache_status"] == "fresh" for row in exact)
        + len(deferred),
        "invalid_requests": len(invalid),
        "batch_exact_requests": sum(row["method"] == "constrained_batch" for row in exact),
        "sequential_exact_requests": sum(
            row["method"] == "constrained_sequential" for row in exact
        ),
        "sequential_deferred_requests": sum(
            row["method"] == "constrained_sequential" for row in deferred
        ),
        "minimum_estimated_headroom_tokens": min(
            (row["estimated_context_headroom"] for row in rows), default=None
        ),
    }


def execute_inventory_requests(
    config: dict[str, Any],
    *,
    run_id: str,
    run_dir: Path,
    blocks: list[dict[str, Any]],
    max_calls: int,
    selection: Selection | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Execute the exact constrained updater contract in an already prepared run."""
    selected = selection or Selection()
    jobs = plan_rubric_jobs(config, selected)
    planned = sum(job.updater_calls for job in jobs)
    resume_plan = plan_inventory_requests(
        config,
        run_id=run_id,
        blocks=blocks,
        selection=selected,
        state_directory=run_dir / "artifacts" / "rubrics" / "constrained_sequential",
        cache_database=run_dir / "calls.sqlite3",
    )
    required = resume_plan["execution_fresh_calls"]
    if max_calls < required:
        raise ValueError(
            f"constrained rubric generation requires {required} fresh calls; MAX_CALLS={max_calls}"
        )
    inventory = load_inventory(config)
    initial = load_initial_inventory_rubric(config)
    batch_template = load_template(config, "batch")
    sequential_template = load_template(config, "sequential")
    raw_client = client or OllamaClient(
        config["models"]["base_url"], config["models"]["timeout_seconds"]
    )
    health = raw_client.health_and_models([config["models"]["updater"]])
    if health["missing"]:
        raise RuntimeError(
            "required constrained updater model is not installed: " + ", ".join(health["missing"])
        )
    budgeted = InferenceBudgetClient(raw_client, max_calls)
    index_path = run_dir / "artifacts" / "rubrics.jsonl"
    if selected.allows_method("static") and selected.allows_replicate(0):
        static_path = run_dir / "artifacts" / "rubrics" / "static" / "initial.json"
        write_json(static_path, initial)
        for block in blocks:
            if not selected.allows_block(block["block_id"]):
                continue
            for order in config["orders"]:
                if selected.allows_order(order):
                    merge_index(
                        index_path,
                        [
                            _inventory_index(
                                "static", block["block_id"], order, 0, False, static_path, run_dir
                            )
                        ],
                        ("method", "block_id", "order", "replicate"),
                    )
    cache_hits = 0
    try:
        with CallStore(run_dir / "calls.sqlite3") as store:
            for job in jobs:
                examples = ordered_examples(blocks[job.block_id], job.order)
                state = {"adaptive_criterion_ids": []}
                steps = (
                    [(1, examples)]
                    if job.method == "batch"
                    else [(step, [example]) for step, example in enumerate(examples, start=1)]
                )
                final_rubric: dict[str, Any] | None = None
                for step, step_examples in steps:
                    template = batch_template if job.method == "batch" else sequential_template
                    request = build_inventory_request(
                        config,
                        run_id=run_id,
                        job=job,
                        step=step,
                        state=state,
                        examples=step_examples,
                        template=template,
                        inventory=inventory,
                        initial=initial,
                    )
                    completed = _load_inventory_step(
                        run_dir, job, step, request, initial, inventory
                    )
                    if completed is not None:
                        state, final_rubric = completed
                        continue
                    if not request["context_preflight_ok"]:
                        raise ValueError(
                            "constrained updater request failed context preflight before model "
                            f"contact: method={job.method}, block={job.block_id}, "
                            f"order={job.order}, replicate={job.replicate}, step={step}"
                        )
                    materialized = {
                        **sanitize_inventory_request(
                            request, store.cache_status(request["request_hash"])
                        ),
                        "materialized_at": datetime.now(timezone.utc).isoformat(),
                        "materialized_before_client_call": True,
                    }
                    merge_index(
                        run_dir / "artifacts" / "request_plans" / "updater.jsonl",
                        [materialized],
                        ("method", "block_id", "order", "replicate", "step"),
                    )
                    parsed, cached = cached_validated_call(
                        client=budgeted,
                        store=store,
                        context=CallContext(
                            run_id,
                            "build-constrained-rubrics",
                            f"constrained_{job.method}",
                            job.block_id,
                            job.order,
                            step,
                            job.replicate,
                        ),
                        model_name=config["models"]["updater"],
                        prompt=request["rendered_prompt"],
                        model_options=request["model_options"],
                        response_schema=request["response_schema"],
                        template_hash=request["prompt_template_hash"],
                        parser=partial(parse_inventory_selection, inventory=inventory),
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
                    state = parsed
                    final_rubric = materialize_inventory_rubric(initial, inventory, state)
                    _write_inventory_step(
                        run_dir,
                        job,
                        step,
                        state,
                        final_rubric,
                        request,
                        store.metadata_for_request(request["request_hash"]),
                        cached,
                    )
                if final_rubric is None:
                    raise AssertionError("constrained updater job produced no rubric")
                final_path = _inventory_rubric_path(run_dir, job, steps[-1][0])
                merge_index(
                    index_path,
                    [
                        _inventory_index(
                            f"constrained_{job.method}",
                            job.block_id,
                            job.order,
                            job.replicate,
                            job.repeat_control,
                            final_path,
                            run_dir,
                        )
                    ],
                    ("method", "block_id", "order", "replicate"),
                )
    finally:
        try:
            if config["models"].get("unload_after_stage", True):
                raw_client.unload(config["models"]["updater"])
        finally:
            sync_manifest_model_calls(run_dir)
    return {
        "planned_calls": planned,
        "fresh_calls_before_execution": required,
        "actual_calls": budgeted.calls,
        "cache_hits": cache_hits,
        "request_hash_parity_enforced": True,
    }


def sanitize_inventory_request(request: dict[str, Any], cache_status: str) -> dict[str, Any]:
    excluded = {"rendered_prompt", "model_options", "response_schema", "context_preflight"}
    return {
        **{key: value for key, value in request.items() if key not in excluded},
        "cache_status": cache_status,
    }


def _deferred_inventory_request(
    config: dict[str, Any],
    run_id: str,
    job: RubricJob,
    step: int,
    example: dict[str, Any],
    template: PromptTemplate,
    inventory: dict[str, Any],
    initial: dict[str, Any],
) -> dict[str, Any]:
    active = active_inventory_ids(inventory)
    longest = sorted(active, key=len, reverse=True)[:2]
    bound = build_inventory_request(
        config,
        run_id=run_id,
        job=job,
        step=step,
        state={"adaptive_criterion_ids": longest},
        examples=[example],
        template=template,
        inventory=inventory,
        initial=initial,
    )
    record = sanitize_inventory_request(bound, "not_applicable")
    record.update(
        {
            "request_status": "deferred",
            "deferred_reason": "depends_on_previous_inventory_state",
            "request_hash": None,
            "rendered_prompt_hash": None,
            "previous_state_hash": None,
            "input_rubric_hash": None,
            "rendered_rubric_hash": None,
            "context_bound_is_conservative": True,
        }
    )
    return record


def _state_path(directory: Path | None, job: RubricJob, step: int) -> Path | None:
    if directory is None:
        return None
    return (
        directory
        / f"block_{job.block_id:02d}"
        / job.order
        / f"replicate_{job.replicate}"
        / f"step_{step:02d}.state.json"
    )


def _initial_path(config: dict[str, Any]) -> Path:
    configured = config.get("inventory_v3", {}).get("initial", config["rubric"]["initial"])
    return resolve_project_path(config, configured)


def _inventory_rubric_path(run_dir: Path, job: RubricJob, step: int) -> Path:
    return (
        run_dir
        / "artifacts"
        / "rubrics"
        / f"constrained_{job.method}"
        / f"block_{job.block_id:02d}"
        / job.order
        / f"replicate_{job.replicate}"
        / f"step_{step:02d}.json"
    )


def _inventory_state_path(run_dir: Path, job: RubricJob, step: int) -> Path:
    return _inventory_rubric_path(run_dir, job, step).with_suffix(".state.json")


def _load_inventory_step(
    run_dir: Path,
    job: RubricJob,
    step: int,
    request: dict[str, Any],
    initial: dict[str, Any],
    inventory: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    rubric_path = _inventory_rubric_path(run_dir, job, step)
    state_path = _inventory_state_path(run_dir, job, step)
    meta_path = rubric_path.with_suffix(".meta.json")
    present = [path.exists() for path in (rubric_path, state_path, meta_path)]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(f"incomplete constrained rubric step cannot be resumed: {rubric_path}")
    state = validate_inventory_selection(read_json(state_path), inventory)
    rubric = validate_materialized_inventory_rubric(read_json(rubric_path), initial, inventory)
    metadata = read_json(meta_path)
    expected = {
        "request_hash": request["request_hash"],
        "previous_state_hash": request["previous_state_hash"],
        "new_state_hash": _object_hash(state),
        "rendered_rubric_hash": _object_hash(rubric),
    }
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        raise ValueError(
            "completed constrained step breaks request/state chain: " + ", ".join(mismatches)
        )
    return state, rubric


def _write_inventory_step(
    run_dir: Path,
    job: RubricJob,
    step: int,
    state: dict[str, Any],
    rubric: dict[str, Any],
    request: dict[str, Any],
    call_metadata: dict[str, Any] | None,
    cache_hit: bool,
) -> None:
    rubric_path = _inventory_rubric_path(run_dir, job, step)
    write_json(rubric_path, rubric)
    write_json(_inventory_state_path(run_dir, job, step), state)
    write_json(
        rubric_path.with_suffix(".meta.json"),
        {
            "method": f"constrained_{job.method}",
            "block_id": job.block_id,
            "order": job.order,
            "replicate": job.replicate,
            "step": step,
            "request_hash": request["request_hash"],
            "previous_state_hash": request["previous_state_hash"],
            "new_state_hash": _object_hash(state),
            "input_rubric_hash": request["input_rubric_hash"],
            "rendered_rubric_hash": _object_hash(rubric),
            "prompt_template_hash": request["prompt_template_hash"],
            "schema_hash": request["schema_hash"],
            "model_options_hash": request["model_options_hash"],
            "context_preflight": request["context_preflight"],
            "cache_status": "hit" if cache_hit else "fresh",
            "call_metadata": call_metadata,
            "materialized_before_client_call": True,
        },
    )


def _inventory_index(
    method: str,
    block_id: int,
    order: str,
    replicate: int,
    repeat_control: bool,
    rubric_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    raw = rubric_path.read_bytes()
    value = json.loads(raw)
    return {
        "method": method,
        "block_id": block_id,
        "order": order,
        "replicate": replicate,
        "repeat_control": repeat_control,
        "rubric_path": rubric_path.relative_to(run_dir).as_posix(),
        "rubric_sha256": hashlib.sha256(raw).hexdigest(),
        "canonical_rubric_sha256": _object_hash(value),
        "schema_version": "3.0",
    }


def _model_example(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "example_id": row["example_id"],
        "prompt": row["prompt"],
        "response": row["response"],
        "human_helpfulness": row["human_score"],
    }


def _assert_initial_v3(initial: dict[str, Any]) -> None:
    if set(initial) != {
        "schema_version",
        "title",
        "criteria",
        "adaptive_criterion_ids",
        "score_anchors",
    }:
        raise ValueError("initial constrained rubric has an invalid top-level contract")
    if initial.get("schema_version") != "3.0":
        raise ValueError("initial constrained rubric must use schema_version 3.0")
    if initial.get("adaptive_criterion_ids") != []:
        raise ValueError("initial constrained rubric must have an empty adaptive state")
    if len(initial.get("criteria", [])) != 3:
        raise ValueError("initial constrained rubric must contain exactly three core criteria")
    fields = {"name", "description", "positive_signals", "negative_signals"}
    if any(not isinstance(row, dict) or set(row) != fields for row in initial["criteria"]):
        raise ValueError("initial constrained rubric has an invalid core criterion contract")
    anchors = initial.get("score_anchors")
    if not isinstance(anchors, dict) or set(anchors) != {str(score) for score in range(5)}:
        raise ValueError("initial constrained rubric must retain all five anchors")


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _object_hash(value: Any) -> str:
    return hashlib.sha256(_compact(value).encode("utf-8")).hexdigest()
