from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import resolve_project_path
from .estimate import Selection
from .judge_validation import build_judge_request, sanitized_request_plan
from .ollama import InferenceBudgetClient, OllamaClient, judge_options
from .prompts import load_template
from .run import load_prepared, merge_index, read_json, read_jsonl, sync_manifest_model_calls
from .schemas import parse_judge_output
from .storage import CallContext, CallStore, cached_validated_call, read_only_cache_status


def build_pilot_judge_request(
    config: dict[str, Any],
    *,
    run_id: str,
    rubric_row: dict[str, Any],
    rubric: dict[str, Any],
    template: Any,
    example: dict[str, Any],
) -> dict[str, Any]:
    """Build one exact pilot judging request for both planning and execution."""
    namespace = _judge_namespace(rubric_row)
    request = build_judge_request(
        config,
        run_id=run_id,
        stage="judge",
        candidate="judge",
        model_name=config["models"]["judge"],
        rubric=rubric,
        template=template,
        example=example,
        cache_namespace=namespace,
    )
    request.update(
        {
            "method": rubric_row["method"],
            "block_id": rubric_row["block_id"],
            "order": rubric_row["order"],
            "replicate": rubric_row["replicate"],
            "repeat_control": bool(rubric_row["repeat_control"]),
            "indexed_rubric_sha256": rubric_row["rubric_sha256"],
            "indexed_canonical_rubric_sha256": rubric_row.get("canonical_rubric_sha256"),
        }
    )
    return request


def sanitized_pilot_judge_request(
    request: dict[str, Any], cache_status: str, *, physical_request: bool
) -> dict[str, Any]:
    result = sanitized_request_plan(request, cache_status)
    result.update(
        {
            key: request[key]
            for key in (
                "method",
                "block_id",
                "order",
                "replicate",
                "repeat_control",
                "indexed_rubric_sha256",
                "indexed_canonical_rubric_sha256",
                "cache_namespace",
            )
        }
    )
    result["physical_request"] = physical_request
    return result


def _judge_namespace(rubric_row: dict[str, Any]) -> str:
    if rubric_row["method"] == "static":
        return "pilot-judge:static"
    return (
        "pilot-judge:"
        f"{rubric_row['method']}:{rubric_row['block_id']}:"
        f"{rubric_row['order']}:{rubric_row['replicate']}"
    )


def plan_pilot_judge_requests(
    config: dict[str, Any],
    *,
    run_id: str,
    run_dir: Path,
    rubric_rows: list[dict[str, Any]],
    heldout: list[dict[str, Any]],
) -> dict[str, Any]:
    """Materialize every exact judge request without contacting or mutating Ollama."""
    template = load_template(config, "judge")
    if config.get("schema_version") in {"2.0", "3.0"}:
        _assert_approved_judge(config, template.sha256)
    database = run_dir / "calls.sqlite3"
    requests: list[dict[str, Any]] = []
    physical: dict[str, dict[str, Any]] = {}
    for rubric_row in rubric_rows:
        rubric = read_json(run_dir / rubric_row["rubric_path"])
        _assert_indexed_rubric(run_dir, rubric_row, rubric)
        for example in heldout:
            request = build_pilot_judge_request(
                config,
                run_id=run_id,
                rubric_row=rubric_row,
                rubric=rubric,
                template=template,
                example=example,
            )
            first_physical = request["request_hash"] not in physical
            cache_status = read_only_cache_status(database, request["request_hash"])
            record = sanitized_pilot_judge_request(
                request,
                cache_status,
                physical_request=first_physical,
            )
            requests.append(record)
            if first_physical:
                physical[request["request_hash"]] = record

    physical_rows = list(physical.values())
    invalid = [row for row in requests if not row["context_preflight_ok"]]
    cache_unavailable = sum(row["cache_status"] == "unavailable" for row in physical_rows)
    return {
        "status": "ready" if not invalid and cache_unavailable == 0 else "invalid",
        "ready": not invalid and cache_unavailable == 0 and bool(requests),
        "requests": requests,
        "exact_requests": len(requests),
        "logical_predictions": len(requests),
        "physical_requests": len(physical_rows),
        "planned_calls": len(physical_rows),
        "cache_hits": sum(row["cache_status"] == "hit" for row in physical_rows),
        "fresh_requests": sum(row["cache_status"] == "fresh" for row in physical_rows),
        "cache_unavailable": cache_unavailable,
        "invalid_requests": len(invalid),
        "maximum_estimated_input_tokens": max(
            (row["estimated_input_tokens"] for row in requests), default=0
        ),
        "minimum_estimated_headroom_tokens": min(
            (row["estimated_context_headroom"] for row in requests), default=None
        ),
        "prompt_template_hash": template.sha256,
        "request_hashes_unique": len(physical),
    }


def selected_rubric_rows(run_dir: Path, selection: Selection) -> list[dict[str, Any]]:
    index_path = run_dir / "artifacts" / "rubrics.jsonl"
    if not index_path.exists():
        raise FileNotFoundError("no stored rubrics; execute build-rubrics first")
    return [
        row
        for row in read_jsonl(index_path)
        if selection.allows_method(row["method"])
        and selection.allows_block(row["block_id"])
        and selection.allows_order(row["order"])
        and selection.allows_replicate(row["replicate"])
    ]


def judge_rubrics(
    config: dict[str, Any],
    run_id: str,
    selection: Selection,
    max_calls: int,
    client: OllamaClient | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    run_dir, _manifest, _blocks, heldout = load_prepared(config, run_id)
    rubric_rows = selected_rubric_rows(run_dir, selection)
    if not rubric_rows:
        raise ValueError("no stored rubrics match the selection")
    static_present = any(row["method"] == "static" for row in rubric_rows)
    non_static_count = sum(row["method"] != "static" for row in rubric_rows)
    planned_calls = len(heldout) * (non_static_count + int(static_present))
    raw_client = client or OllamaClient(
        config["models"]["base_url"], config["models"]["timeout_seconds"]
    )
    health = raw_client.health_and_models([config["models"]["judge"]])
    if health["missing"]:
        raise RuntimeError(
            "required judge model is not installed in Ollama: " + ", ".join(health["missing"])
        )
    template = load_template(config, "judge")
    if config.get("schema_version") in {"2.0", "3.0"}:
        _assert_approved_judge(config, template.sha256)
    predictions_path = run_dir / "artifacts" / "predictions.jsonl"
    budgeted_client = InferenceBudgetClient(raw_client, max_calls)
    cache_hits = 0
    seen_request_hashes: set[str] = set()
    stopped_at_budget = False
    try:
        with CallStore(run_dir / "calls.sqlite3") as store:
            for rubric_index, rubric_row in enumerate(rubric_rows, start=1):
                rubric = read_json(run_dir / rubric_row["rubric_path"])
                _assert_indexed_rubric(run_dir, rubric_row, rubric)
                predictions: list[dict[str, Any]] = []
                for example in heldout:
                    request = build_pilot_judge_request(
                        config,
                        run_id=run_id,
                        rubric_row=rubric_row,
                        rubric=rubric,
                        template=template,
                        example=example,
                    )
                    repeat = bool(rubric_row["repeat_control"])
                    if not request["context_preflight_ok"]:
                        raise ValueError(
                            "judge request failed context preflight before model contact: "
                            f"method={rubric_row['method']}, block={rubric_row['block_id']}, "
                            f"order={rubric_row['order']}, "
                            f"replicate={rubric_row['replicate']}, "
                            f"example_id={example['example_id']}"
                        )
                    physical_request = request["request_hash"] not in seen_request_hashes
                    seen_request_hashes.add(request["request_hash"])
                    cached_before = store.cached_success(request["request_hash"])
                    if cached_before is None and budgeted_client.calls >= max_calls:
                        stopped_at_budget = True
                        break
                    _persist_judge_request_plan(
                        run_dir,
                        request,
                        store,
                        physical_request=physical_request,
                    )
                    parsed, cached = cached_validated_call(
                        client=budgeted_client,
                        store=store,
                        context=CallContext(
                            run_id=run_id,
                            stage="judge",
                            method=rubric_row["method"],
                            block_id=rubric_row["block_id"],
                            order_name=rubric_row["order"],
                            step=None,
                            replicate=rubric_row["replicate"],
                        ),
                        model_name=config["models"]["judge"],
                        prompt=request["rendered_prompt"],
                        model_options=request["model_options"],
                        response_schema=request["response_schema"],
                        template_hash=request["prompt_template_hash"],
                        parser=parse_judge_output,
                        max_validation_retries=config["rubric"]["max_validation_retries"],
                        cache_namespace=request["cache_namespace"],
                        schema_version=config.get("schema_version", "1.0"),
                        prompt_version=config.get("prompt_version", "1.0"),
                        context_preflight=request["context_preflight"],
                        minimum_context_headroom=(
                            config["context"]["minimum_observed_headroom_tokens"]
                            if config.get("schema_version") in {"2.0", "3.0"}
                            else None
                        ),
                        expected_request_hash=request["request_hash"],
                    )
                    cache_hits += int(cached)
                    predictions.append(
                        {
                            "method": rubric_row["method"],
                            "block_id": rubric_row["block_id"],
                            "order": rubric_row["order"],
                            "replicate": rubric_row["replicate"],
                            "repeat_control": repeat,
                            "rubric_sha256": rubric_row["rubric_sha256"],
                            "example_id": example["example_id"],
                            "prompt_id": example["prompt_id"],
                            "human_score": example["human_score"],
                            "predicted_score": parsed["score"],
                            "cache_hit": cached,
                            "schema_version": config.get("schema_version", "1.0"),
                            "prompt_version": config.get("prompt_version", "1.0"),
                            "judge_prompt_sha256": template.sha256,
                            "request_hash": request["request_hash"],
                            "rubric_canonical_sha256": request["rubric_hash"],
                            "schema_sha256": request["schema_hash"],
                            "model_options_sha256": request["model_options_hash"],
                        }
                    )
                merge_index(
                    predictions_path,
                    predictions,
                    ("method", "block_id", "order", "replicate", "example_id"),
                )
                if progress_callback is not None:
                    progress_callback(
                        {
                            "rubrics_completed": rubric_index,
                            "rubrics_total": len(rubric_rows),
                            "method": rubric_row["method"],
                            "block_id": rubric_row["block_id"],
                            "order": rubric_row["order"],
                            "replicate": rubric_row["replicate"],
                            "predictions_saved": len(predictions),
                        }
                    )
                if stopped_at_budget:
                    break
    finally:
        try:
            if config["models"].get("unload_after_stage", True):
                raw_client.unload(config["models"]["judge"])
        finally:
            sync_manifest_model_calls(run_dir)
    return {
        "planned_calls": planned_calls,
        "actual_calls": budgeted_client.calls,
        "cache_hits": cache_hits,
        "status": "partial_budget_stop" if stopped_at_budget else "complete",
        "resume_safe": True,
    }


def _assert_indexed_rubric(
    run_dir: Path, rubric_row: dict[str, Any], rubric: dict[str, Any]
) -> None:
    path = run_dir / rubric_row["rubric_path"]
    raw_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    canonical_hash = hashlib.sha256(
        json.dumps(rubric, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if raw_hash != rubric_row["rubric_sha256"]:
        raise ValueError(f"indexed rubric raw hash mismatch: {path}")
    if canonical_hash != rubric_row.get("canonical_rubric_sha256"):
        raise ValueError(f"indexed rubric canonical hash mismatch: {path}")


def _persist_judge_request_plan(
    run_dir: Path,
    request: dict[str, Any],
    store: CallStore,
    *,
    physical_request: bool,
) -> None:
    record = sanitized_pilot_judge_request(
        request,
        store.cache_status(request["request_hash"]),
        physical_request=physical_request,
    )
    record["materialized_at"] = datetime.now(timezone.utc).isoformat()
    record["materialized_before_client_call"] = True
    merge_index(
        run_dir / "artifacts" / "request_plans" / "judge.jsonl",
        [record],
        ("method", "block_id", "order", "replicate", "example_id"),
    )


def _assert_approved_judge(config: dict[str, Any], prompt_sha256: str) -> None:
    configured = config.get("approved_judge_artifact")
    if not configured:
        raise ValueError("v2 judging requires approved_judge_artifact")
    path = resolve_project_path(config, configured)
    if not path.exists():
        raise FileNotFoundError(f"approved judge artifact does not exist: {path}")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("verdict") != "JUDGE_APPROVED":
        raise ValueError("judge validation artifact is not JUDGE_APPROVED")
    chosen = artifact.get("chosen", {})
    if chosen.get("model") != config["models"]["judge"]:
        raise ValueError("configured judge model differs from approved artifact")
    if chosen.get("prompt_sha256") != prompt_sha256:
        raise ValueError("configured judge prompt hash differs from approved artifact")
    approved_options = chosen.get("model_options")
    if approved_options is not None and approved_options != judge_options(config):
        raise ValueError("configured judge options differ from approved artifact")
