from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from .config import resolve_project_path
from .context import preflight_context, sha256_text, truncated_example_ids
from .estimate import RubricJob, Selection, plan_rubric_jobs
from .ollama import InferenceBudgetClient, OllamaClient, updater_options
from .prompts import load_template
from .run import (
    load_prepared,
    merge_index,
    read_json,
    read_jsonl,
    sync_manifest_model_calls,
    write_json,
)
from .sampling import ordered_examples
from .schemas import (
    RUBRIC_JSON_SCHEMA,
    RUBRIC_V2_JSON_SCHEMA,
    drift_audit,
    parse_rubric,
    parse_rubric_v2,
    validate_rubric,
    validate_rubric_v2,
)
from .storage import (
    CallContext,
    CallStore,
    cached_validated_call,
    make_request_hash,
    read_only_cache_status,
)


def build_updater_request(
    config: dict[str, Any],
    *,
    run_id: str,
    job: RubricJob,
    step: int,
    input_rubric: dict[str, Any],
    examples: list[dict[str, Any]],
    template: Any,
    response_schema: dict[str, Any],
) -> dict[str, Any]:
    """Build the exact updater request shared by planning and execution."""
    if job.method == "batch":
        if step != 1:
            raise ValueError("batch updater has exactly one step")
        prompt = template.render(
            rubric_json=_compact(input_rubric),
            schema_json=_compact(response_schema),
            examples_json=_compact(
                [
                    {
                        "example_id": row["example_id"],
                        "prompt": row["prompt"],
                        "response": row["response"],
                        "human_helpfulness": row["human_score"],
                    }
                    for row in examples
                ]
            ),
        )
    elif job.method == "sequential":
        if len(examples) != 1:
            raise ValueError("sequential updater requires exactly one example per request")
        example = examples[0]
        prompt = template.render(
            rubric_json=_compact(input_rubric),
            schema_json=_compact(response_schema),
            prompt_json=json.dumps(example["prompt"], ensure_ascii=False),
            response_json=json.dumps(example["response"], ensure_ascii=False),
            label=example["human_score"],
        )
    else:
        raise ValueError(f"unsupported updater method: {job.method}")
    options = updater_options(config)
    namespace = _updater_namespace(job)
    context = preflight_context(
        prompt,
        context_length=config["models"]["context_length"],
        max_output_tokens=config["models"]["updater_max_tokens"],
        safety_margin_tokens=config["context"]["safety_margin_tokens"],
        method=job.method,
        block_id=job.block_id,
        order=job.order,
        reject=False,
    )
    input_rubric_hash = sha256_text(_compact(input_rubric))
    request_hash = make_request_hash(
        config["models"]["updater"],
        prompt,
        options,
        response_schema,
        template.sha256,
        namespace,
    )
    return {
        "run_id": run_id,
        "stage": "build-rubrics",
        "method": job.method,
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
        "model": config["models"]["updater"],
        "model_options": options,
        "response_schema": response_schema,
        "cache_namespace": namespace,
        "request_status": "exact",
        "request_hash": request_hash,
        "rendered_prompt_hash": sha256_text(prompt),
        "prompt_template_hash": template.sha256,
        "input_rubric_hash": input_rubric_hash,
        "schema_hash": sha256_text(_compact(response_schema)),
        "model_options_hash": sha256_text(_compact(options)),
        "estimated_input_tokens": context["estimated_input_tokens"],
        "context_length": context["context_length"],
        "reserved_output_tokens": context["max_output_tokens"],
        "safety_margin_tokens": context["safety_margin_tokens"],
        "estimated_context_headroom": context["estimated_headroom_tokens"],
        "context_preflight_ok": context["allowed"],
        "context_failure_reason": (
            None if context["allowed"] else "estimated_input_exceeds_context_budget"
        ),
        "token_estimator": context["token_estimate_method"],
        "token_estimate_exact": False,
        "context_preflight": context,
    }


def sanitized_updater_request(request: dict[str, Any], cache_status: str) -> dict[str, Any]:
    excluded = {"rendered_prompt", "model_options", "response_schema", "context_preflight"}
    return {
        **{key: value for key, value in request.items() if key not in excluded},
        "cache_status": cache_status,
    }


def _updater_namespace(job: RubricJob) -> str:
    return f"pilot-updater:{job.method}:{job.block_id}:{job.order}:{job.replicate}"


def build_rubrics(
    config: dict[str, Any],
    run_id: str,
    selection: Selection,
    max_calls: int,
    client: OllamaClient | None = None,
) -> dict[str, int]:
    run_dir, _manifest, blocks, _heldout = load_prepared(config, run_id)
    jobs = plan_rubric_jobs(config, selection)
    planned_calls = sum(job.updater_calls for job in jobs)
    initial_path = resolve_project_path(config, config["rubric"]["initial"])
    initial_value = read_json(initial_path)
    is_v2 = config.get("schema_version") == "2.0"
    initial_rubric = (
        validate_rubric_v2(initial_value, initial_value)
        if is_v2
        else validate_rubric(initial_value)
    )
    response_schema = RUBRIC_V2_JSON_SCHEMA if is_v2 else RUBRIC_JSON_SCHEMA
    index_path = run_dir / "artifacts" / "rubrics.jsonl"
    if selection.allows_method("static") and selection.allows_replicate(0):
        static_path = run_dir / "artifacts" / "rubrics" / "static" / "initial.json"
        write_json(static_path, initial_rubric)
        static_rows = [
            _index_row("static", block["block_id"], order, 0, False, static_path, run_dir)
            for block in blocks
            if selection.allows_block(block["block_id"])
            for order in config["orders"]
            if selection.allows_order(order)
        ]
        merge_index(
            index_path,
            static_rows,
            ("method", "block_id", "order", "replicate"),
        )
    if not jobs:
        return {"planned_calls": 0, "actual_calls": 0, "cache_hits": 0}

    raw_client = client or OllamaClient(
        config["models"]["base_url"], config["models"]["timeout_seconds"]
    )
    health = raw_client.health_and_models([config["models"]["updater"]])
    if health["missing"]:
        raise RuntimeError(
            "required updater model is not installed in Ollama: " + ", ".join(health["missing"])
        )
    sequential_template = load_template(config, "sequential")
    batch_template = load_template(config, "batch")
    budgeted_client = InferenceBudgetClient(raw_client, max_calls)
    cache_hits = 0
    try:
        with CallStore(run_dir / "calls.sqlite3") as store:
            for job in jobs:
                block = blocks[job.block_id]
                examples = ordered_examples(block, job.order)
                rubric = initial_rubric
                if job.method == "batch":
                    request = build_updater_request(
                        config,
                        run_id=run_id,
                        job=job,
                        step=1,
                        input_rubric=initial_rubric,
                        examples=examples,
                        template=batch_template,
                        response_schema=response_schema,
                    )
                    final_step = 1
                    completed = _load_completed_step(
                        run_dir,
                        job,
                        final_step,
                        request,
                        initial_rubric,
                    )
                    if completed is not None:
                        rubric = completed
                    else:
                        _assert_updater_preflight(request)
                        _persist_materialized_request(run_dir, request, store)
                        rubric, cached, metadata = _update_once(
                            budgeted_client,
                            store,
                            config,
                            run_id,
                            job,
                            request=request,
                            initial_rubric=initial_rubric,
                        )
                        cache_hits += int(cached)
                        _write_step(
                            run_dir,
                            job,
                            final_step,
                            rubric,
                            metadata,
                            examples,
                            initial_rubric,
                        )
                else:
                    for step, example in enumerate(examples, start=1):
                        request = build_updater_request(
                            config,
                            run_id=run_id,
                            job=job,
                            step=step,
                            input_rubric=rubric,
                            examples=[example],
                            template=sequential_template,
                            response_schema=response_schema,
                        )
                        if step > 1:
                            previous_path = _step_path(run_dir, job, step - 1)
                            if not previous_path.exists():
                                raise ValueError(
                                    "sequential chain is missing the previous canonical rubric: "
                                    f"{previous_path}"
                                )
                            previous_hash = sha256_text(_compact(read_json(previous_path)))
                            if request["input_rubric_hash"] != previous_hash:
                                raise ValueError(
                                    "sequential input rubric hash does not match previous output"
                                )
                        completed = _load_completed_step(
                            run_dir,
                            job,
                            step,
                            request,
                            initial_rubric,
                        )
                        if completed is not None:
                            rubric = completed
                            continue
                        _assert_updater_preflight(request)
                        _persist_materialized_request(run_dir, request, store)
                        rubric, cached, metadata = _update_once(
                            budgeted_client,
                            store,
                            config,
                            run_id,
                            job,
                            request=request,
                            initial_rubric=initial_rubric,
                        )
                        cache_hits += int(cached)
                        _write_step(run_dir, job, step, rubric, metadata, [example], initial_rubric)
                    final_step = len(examples)
                final_path = _step_path(run_dir, job, final_step)
                merge_index(
                    index_path,
                    [
                        _index_row(
                            job.method,
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
        "planned_calls": planned_calls,
        "actual_calls": budgeted_client.calls,
        "cache_hits": cache_hits,
    }


def _load_completed_step(
    run_dir: Path,
    job: RubricJob,
    step: int,
    request: dict[str, Any],
    initial_rubric: dict[str, Any],
) -> dict[str, Any] | None:
    """Validate a completed artifact before allowing resume to advance past it."""
    path = _step_path(run_dir, job, step)
    meta_path = path.with_suffix(".meta.json")
    if not path.exists() and not meta_path.exists():
        return None
    if not path.exists() or not meta_path.exists():
        raise ValueError(f"incomplete rubric step artifact cannot be resumed: {path}")
    value = read_json(path)
    if value.get("schema_version") == "2.0":
        rubric = validate_rubric_v2(value, initial_rubric)
    else:
        rubric = validate_rubric(value)
    metadata = read_json(meta_path)
    canonical_hash = sha256_text(_compact(rubric))
    checks = {
        "request_hash": request["request_hash"],
        "input_rubric_hash": request["input_rubric_hash"],
        "output_canonical_rubric_hash": canonical_hash,
    }
    for field, expected in checks.items():
        if metadata.get(field) != expected:
            raise ValueError(
                f"completed rubric step has a broken canonical chain ({field}): {path}"
            )
    if not metadata.get("materialized_before_client_call"):
        raise ValueError(f"completed rubric step lacks pre-call materialization proof: {path}")
    return rubric


def _update_once(
    client: OllamaClient,
    store: CallStore,
    config: dict[str, Any],
    run_id: str,
    job: RubricJob,
    request: dict[str, Any],
    initial_rubric: dict[str, Any],
) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    is_v2 = config.get("schema_version") == "2.0"
    context_result = request["context_preflight"] if is_v2 else None
    parser = partial(parse_rubric_v2, initial_rubric=initial_rubric) if is_v2 else parse_rubric
    parsed, cached = cached_validated_call(
        client=client,
        store=store,
        context=CallContext(
            run_id=run_id,
            stage="build-rubrics",
            method=job.method,
            block_id=job.block_id,
            order_name=job.order,
            step=request["step"],
            replicate=job.replicate,
        ),
        model_name=config["models"]["updater"],
        prompt=request["rendered_prompt"],
        model_options=request["model_options"],
        response_schema=request["response_schema"],
        template_hash=request["prompt_template_hash"],
        parser=parser,
        max_validation_retries=config["rubric"]["max_validation_retries"],
        cache_namespace=request["cache_namespace"],
        schema_version=config.get("schema_version", "1.0"),
        prompt_version=config.get("prompt_version", "1.0"),
        context_preflight=context_result,
        minimum_context_headroom=(
            config["context"]["minimum_observed_headroom_tokens"] if is_v2 else None
        ),
        expected_request_hash=request["request_hash"],
    )
    metadata = store.metadata_for_request(request["request_hash"]) or {
        "schema_version": config.get("schema_version", "1.0"),
        "prompt_version": config.get("prompt_version", "1.0"),
        "estimated_input_tokens": (
            context_result["estimated_input_tokens"] if context_result else None
        ),
    }
    metadata["cache_hit"] = cached
    metadata.update(
        {
            "request_hash": request["request_hash"],
            "rendered_prompt_hash": request["rendered_prompt_hash"],
            "prompt_template_hash": request["prompt_template_hash"],
            "input_rubric_hash": request["input_rubric_hash"],
            "schema_hash": request["schema_hash"],
            "model_options_hash": request["model_options_hash"],
            "estimated_context_headroom": request["estimated_context_headroom"],
            "request_materialized_at": request.get("materialized_at"),
            "example_id": request["example_id"],
            "prompt_id": request["prompt_id"],
        }
    )
    return parsed, cached, metadata


def _assert_updater_preflight(request: dict[str, Any]) -> None:
    if not request["context_preflight_ok"]:
        raise ValueError(
            "updater request failed context preflight before model contact: "
            f"method={request['method']}, block={request['block_id']}, "
            f"order={request['order']}, replicate={request['replicate']}, "
            f"step={request['step']}, headroom={request['estimated_context_headroom']}"
        )


def _persist_materialized_request(run_dir: Path, request: dict[str, Any], store: CallStore) -> None:
    request["materialized_at"] = datetime.now(timezone.utc).isoformat()
    request["materialized_before_client_call"] = True
    record = sanitized_updater_request(request, store.cache_status(request["request_hash"]))
    merge_index(
        run_dir / "artifacts" / "request_plans" / "updater.jsonl",
        [record],
        ("method", "block_id", "order", "replicate", "step"),
    )


def _write_step(
    run_dir: Path,
    job: RubricJob,
    step: int,
    rubric: dict[str, Any],
    metadata: dict[str, Any],
    examples: list[dict[str, Any]],
    initial_rubric: dict[str, Any],
) -> None:
    path = _step_path(run_dir, job, step)
    write_json(path, rubric)
    canonical_hash = hashlib.sha256(_compact(rubric).encode("utf-8")).hexdigest()
    write_json(
        path.with_suffix(".meta.json"),
        {
            "method": job.method,
            "block_id": job.block_id,
            "order": job.order,
            "replicate": job.replicate,
            "step": step,
            "example_id": metadata.get("example_id"),
            "prompt_id": metadata.get("prompt_id"),
            "request_hash": metadata.get("request_hash"),
            "prompt_template_hash": metadata.get("prompt_template_hash"),
            "input_rubric_hash": metadata.get("input_rubric_hash"),
            "schema_version": metadata.get("schema_version", rubric["schema_version"]),
            "prompt_version": metadata.get("prompt_version"),
            "raw_response_sha256": metadata.get("raw_response_sha256"),
            "output_raw_rubric_hash": metadata.get("raw_response_sha256"),
            "canonical_rubric_sha256": canonical_hash,
            "output_canonical_rubric_hash": canonical_hash,
            "prompt_sha256": metadata.get("prompt_sha256"),
            "rendered_prompt_hash": metadata.get("rendered_prompt_hash"),
            "schema_hash": metadata.get("schema_hash"),
            "model_options_hash": metadata.get("model_options_hash"),
            "estimated_input_tokens": metadata.get("estimated_input_tokens"),
            "estimated_context_headroom": metadata.get("estimated_context_headroom"),
            "prompt_eval_count": metadata.get("prompt_eval_count"),
            "observed_headroom_tokens": metadata.get("observed_headroom_tokens"),
            "context_saturation": bool(metadata.get("context_saturation", False)),
            "backend_prompt_truncated": bool(metadata.get("backend_prompt_truncated", False)),
            "attempt": metadata.get("attempt"),
            "latency_seconds": metadata.get("latency_seconds"),
            "cache_namespace": metadata.get("cache_namespace"),
            "cache_hit": metadata.get("cache_hit", False),
            "cache_status": "hit" if metadata.get("cache_hit", False) else "fresh",
            "status": "success",
            "request_materialized_at": metadata.get("request_materialized_at"),
            "materialized_before_client_call": True,
            "drift_audit": drift_audit(rubric, initial_rubric, examples),
        },
    )


def _step_path(run_dir: Path, job: RubricJob, step: int) -> Path:
    return (
        run_dir
        / "artifacts"
        / "rubrics"
        / job.method
        / f"block_{job.block_id:02d}"
        / job.order
        / f"replicate_{job.replicate}"
        / f"step_{step:02d}.json"
    )


def _index_row(
    method: str,
    block_id: int,
    order: str,
    replicate: int,
    repeat_control: bool,
    rubric_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    content = rubric_path.read_bytes()
    return {
        "method": method,
        "block_id": block_id,
        "order": order,
        "replicate": replicate,
        "repeat_control": repeat_control,
        "rubric_path": rubric_path.relative_to(run_dir).as_posix(),
        "rubric_sha256": hashlib.sha256(content).hexdigest(),
        "canonical_rubric_sha256": hashlib.sha256(
            _compact(json.loads(content)).encode("utf-8")
        ).hexdigest(),
        "schema_version": json.loads(content)["schema_version"],
    }


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def plan_rubric_context(
    config: dict[str, Any],
    blocks: list[dict[str, Any]],
    jobs: list[RubricJob],
    *,
    run_id: str = "<plan>",
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Build a phase-aware updater plan without inventing sequential request hashes."""
    initial_path = resolve_project_path(config, config["rubric"]["initial"])
    initial = read_json(initial_path)
    schema = RUBRIC_V2_JSON_SCHEMA if config.get("schema_version") == "2.0" else RUBRIC_JSON_SCHEMA
    sequential = load_template(config, "sequential")
    batch = load_template(config, "batch")
    budget_rubric = (
        _maximum_budget_rubric(initial) if config.get("schema_version") == "2.0" else initial
    )
    options = updater_options(config)
    persisted_path = run_dir / "artifacts" / "request_plans" / "updater.jsonl" if run_dir else None
    persisted_rows = (
        {
            (
                row["method"],
                row["block_id"],
                row["order"],
                row["replicate"],
                row["step"],
            ): row
            for row in (read_jsonl(persisted_path) if persisted_path.exists() else [])
        }
        if persisted_path
        else {}
    )
    database = run_dir / "calls.sqlite3" if run_dir else None
    rows: list[dict[str, Any]] = []
    for job in jobs:
        examples = ordered_examples(blocks[job.block_id], job.order)
        if job.method == "batch":
            requests = [
                build_updater_request(
                    config,
                    run_id=run_id,
                    job=job,
                    step=1,
                    input_rubric=initial,
                    examples=examples,
                    template=batch,
                    response_schema=schema,
                )
            ]
        else:
            requests = []
            for step, example in enumerate(examples, start=1):
                if step == 1:
                    input_rubric = initial
                else:
                    previous_path = _step_path(run_dir, job, step - 1) if run_dir else None
                    input_rubric = (
                        read_json(previous_path)
                        if previous_path is not None and previous_path.exists()
                        else None
                    )
                if input_rubric is not None:
                    requests.append(
                        build_updater_request(
                            config,
                            run_id=run_id,
                            job=job,
                            step=step,
                            input_rubric=input_rubric,
                            examples=[example],
                            template=sequential,
                            response_schema=schema,
                        )
                    )
                    continue
                bound_prompt = sequential.render(
                    rubric_json=_compact(budget_rubric),
                    schema_json=_compact(schema),
                    prompt_json=json.dumps(example["prompt"], ensure_ascii=False),
                    response_json=json.dumps(example["response"], ensure_ascii=False),
                    label=example["human_score"],
                )
                bound = preflight_context(
                    bound_prompt,
                    context_length=config["models"]["context_length"],
                    max_output_tokens=config["models"]["updater_max_tokens"],
                    safety_margin_tokens=config["context"]["safety_margin_tokens"],
                    method=job.method,
                    block_id=job.block_id,
                    order=job.order,
                    reject=False,
                )
                rows.append(
                    {
                        "run_id": run_id,
                        "stage": "build-rubrics",
                        "method": job.method,
                        "block_id": job.block_id,
                        "order": job.order,
                        "replicate": job.replicate,
                        "repeat_control": job.repeat_control,
                        "step": step,
                        "example_id": example["example_id"],
                        "prompt_id": example["prompt_id"],
                        "request_status": "deferred",
                        "deferred_reason": "depends_on_previous_rubric",
                        "request_hash": None,
                        "rendered_prompt_hash": None,
                        "input_rubric_hash": None,
                        "prompt_template_hash": sequential.sha256,
                        "schema_hash": sha256_text(_compact(schema)),
                        "model_options_hash": sha256_text(_compact(options)),
                        "cache_status": "not_applicable",
                        "rubric_state_assumption": "maximum_v2_adaptive_shape",
                        "context_bound_is_conservative": True,
                        "estimated_input_tokens": bound["estimated_input_tokens"],
                        "context_length": bound["context_length"],
                        "reserved_output_tokens": bound["max_output_tokens"],
                        "safety_margin_tokens": bound["safety_margin_tokens"],
                        "estimated_context_headroom": bound["estimated_headroom_tokens"],
                        "context_preflight_ok": bound["allowed"],
                        "context_failure_reason": (
                            None
                            if bound["allowed"]
                            else "conservative_context_bound_exceeds_budget"
                        ),
                        "token_estimator": bound["token_estimate_method"],
                        "token_estimate_exact": False,
                    }
                )
        for request in requests:
            cache_status = (
                read_only_cache_status(database, request["request_hash"])
                if database is not None
                else "fresh"
            )
            record = sanitized_updater_request(request, cache_status)
            key = (
                job.method,
                job.block_id,
                job.order,
                job.replicate,
                request["step"],
            )
            persisted = persisted_rows.get(key)
            if persisted is not None:
                record["materialized_at"] = persisted.get("materialized_at")
                record["materialized_before_client_call"] = persisted.get(
                    "materialized_before_client_call", False
                )
                record["materialization_hash_match"] = (
                    persisted.get("request_hash") == record["request_hash"]
                )
            rows.append(record)
    exact = [row for row in rows if row["request_status"] == "exact"]
    deferred = [row for row in rows if row["request_status"] == "deferred"]
    invalid = [row for row in rows if not row["context_preflight_ok"]]
    cache_unavailable = sum(row.get("cache_status") == "unavailable" for row in exact)
    maximum = max((row["estimated_input_tokens"] for row in rows), default=0)
    return {
        "calls": rows,
        "requests": rows,
        "ready": not invalid and cache_unavailable == 0,
        "planned_physical_updater_calls": len(rows),
        "exact_requests": len(exact),
        "deferred_requests": len(deferred),
        "cache_hits": sum(row["cache_status"] == "hit" for row in exact),
        "fresh_requests": sum(row["cache_status"] == "fresh" for row in exact),
        "execution_fresh_calls": sum(row["cache_status"] == "fresh" for row in exact)
        + len(deferred),
        "cache_unavailable": cache_unavailable,
        "invalid_requests": len(invalid),
        "materialized_requests": sum(
            bool(row.get("materialized_before_client_call")) for row in exact
        ),
        "materialization_mismatches": sum(
            row.get("materialization_hash_match") is False for row in exact
        ),
        "batch_exact_requests": sum(
            row["method"] == "batch" and row["request_status"] == "exact" for row in rows
        ),
        "sequential_exact_requests": sum(
            row["method"] == "sequential" and row["request_status"] == "exact" for row in rows
        ),
        "sequential_deferred_requests": sum(
            row["method"] == "sequential" and row["request_status"] == "deferred" for row in rows
        ),
        "maximum_estimated_input_tokens": maximum,
        "minimum_estimated_headroom_tokens": min(
            (row["estimated_context_headroom"] for row in rows), default=None
        ),
        "refused_calls": invalid,
        "truncated_example_ids": truncated_example_ids(blocks),
    }


def _maximum_budget_rubric(initial: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(initial))
    for index in range(2):
        result["criteria"].append(
            {
                "name": "\\" * 79 + str(index),
                "description": "\\" * 239 + str(index),
                "positive_signals": ["\\" * 159 + str(item) for item in range(6)],
                "negative_signals": ["\\" * 158 + f"n{item}" for item in range(6)],
            }
        )
    return result
