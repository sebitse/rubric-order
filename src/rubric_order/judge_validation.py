from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import public_config, resolve_project_path
from .context import preflight_context
from .data import dataset_fingerprint, load_dataset
from .exclusions import apply_exclusions, load_exclusions
from .metrics import agreement_metrics
from .ollama import InferenceBudgetClient, OllamaClient, judge_options
from .prompts import load_template
from .run import (
    merge_index,
    read_json,
    read_jsonl,
    run_directory,
    write_json,
    write_jsonl,
)
from .schemas import JUDGE_JSON_SCHEMA, parse_judge_output, validate_rubric_v2
from .storage import (
    CallContext,
    CallStore,
    cached_validated_call,
    make_request_hash,
    read_only_cache_status,
)

DEV_STAGE = "judge-validation-dev"
VALIDATION_STAGE = "judge-validation-holdout"


def select_judge_subsets(
    rows: list[dict[str, Any]],
    *,
    dev_per_score: int = 10,
    validation_per_score: int = 20,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select exact balanced, response-level subsets with globally distinct prompts."""
    by_score: dict[int, list[dict[str, Any]]] = {score: [] for score in range(5)}
    for row in rows:
        by_score[row["human_score"]].append(row)
    for score, candidates in by_score.items():
        candidates.sort(key=lambda row: (row["prompt_id"], row["example_id"]))
        random.Random(seed + 7_919 * score).shuffle(candidates)
    used_prompts: set[str] = set()
    dev: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for subset, count in ((dev, dev_per_score), (validation, validation_per_score)):
        for score in range(5):
            selected: list[dict[str, Any]] = []
            for row in by_score[score]:
                if row["prompt_id"] in used_prompts:
                    continue
                selected.append(row)
                used_prompts.add(row["prompt_id"])
                if len(selected) == count:
                    break
            if len(selected) != count:
                raise ValueError(
                    f"cannot construct judge subset: score {score} needs {count} distinct "
                    f"prompts but only {len(selected)} remain"
                )
            subset.extend(selected)
    _assert_subset_contract(dev, dev_per_score)
    _assert_subset_contract(validation, validation_per_score)
    if {row["prompt_id"] for row in dev} & {row["prompt_id"] for row in validation}:
        raise AssertionError("judge-dev and judge-validation are not prompt-disjoint")
    return sorted(dev, key=_subset_sort_key), sorted(validation, key=_subset_sort_key)


def prepare_judge_validation(
    config: dict[str, Any], run_id: str | None = None, *, resume: bool = False
) -> dict[str, Any]:
    settings = config["judge_validation"]
    selected_run_id = run_id or settings["run_id"]
    run_dir = run_directory(config, selected_run_id)
    manifest_path = run_dir / "manifest.json"
    if run_dir.exists():
        if not resume:
            raise FileExistsError(f"judge-validation run already exists: {run_dir}")
        if not manifest_path.exists():
            raise ValueError("judge-validation directory has no manifest")
        return read_json(manifest_path)
    train, validation = load_dataset(config)
    if validation is None:
        raise ValueError("judge validation requires an official validation split")
    excluded_examples, excluded_prompts, exclusion_report = load_exclusions(config)
    eligible, removal = apply_exclusions(validation, excluded_examples, excluded_prompts)
    dev, holdout = select_judge_subsets(
        eligible,
        dev_per_score=settings["dev_per_score"],
        validation_per_score=settings["validation_per_score"],
        seed=config["seed"],
    )
    run_dir.mkdir(parents=True)
    write_json(run_dir / "config.resolved.json", public_config(config))
    write_jsonl(run_dir / "subsets" / "judge_dev.jsonl", [_concise_row(row) for row in dev])
    write_jsonl(
        run_dir / "subsets" / "judge_validation.jsonl", [_concise_row(row) for row in holdout]
    )
    manifest = {
        "schema_version": config["schema_version"],
        "prompt_version": config["prompt_version"],
        "run_id": selected_run_id,
        "workflow": "judge_validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": dataset_fingerprint(train, validation),
        "exclusions": {**exclusion_report, "validation": removal},
        "dev_response_count": len(dev),
        "validation_response_count": len(holdout),
        "dev_label_counts": _label_counts(dev),
        "validation_label_counts": _label_counts(holdout),
        "prompt_disjoint": True,
        "model_calls_made": 0,
    }
    write_json(manifest_path, manifest)
    return manifest


def estimate_judge_validation(
    config: dict[str, Any], include_qwen_fallback: bool = False
) -> dict[str, int]:
    settings = config["judge_validation"]
    dev = 5 * settings["dev_per_score"]
    holdout = 5 * settings["validation_per_score"]
    candidate_count = len(settings["candidates"])
    llama_dev = candidate_count * dev
    normal = llama_dev + holdout
    fallback = candidate_count * dev if include_qwen_fallback else 0
    return {
        "llama_dev_calls": llama_dev,
        "untouched_validation_calls": holdout,
        "normal_total_calls": normal,
        "optional_qwen_dev_calls": fallback,
        "total_with_requested_fallback": normal + fallback,
    }


def build_judge_request(
    config: dict[str, Any],
    *,
    run_id: str,
    stage: str,
    candidate: str,
    model_name: str,
    rubric: dict[str, Any],
    template: Any,
    example: dict[str, Any],
    response_schema: dict[str, Any] | None = None,
    cache_namespace: str | None = None,
) -> dict[str, Any]:
    """Build the deterministic request consumed by planning and execution."""
    schema = JUDGE_JSON_SCHEMA if response_schema is None else response_schema
    options = judge_options(config)
    rubric_json = _canonical_json(rubric)
    prompt = template.render(
        output_example=_canonical_json({"score": 0}),
        rubric_json=rubric_json,
        prompt_json=json.dumps(example["prompt"], ensure_ascii=False),
        response_json=json.dumps(example["response"], ensure_ascii=False),
    )
    context = preflight_context(
        prompt,
        context_length=config["models"]["context_length"],
        max_output_tokens=config["models"]["judge_max_tokens"],
        safety_margin_tokens=config["context"]["safety_margin_tokens"],
        method=candidate,
        block_id=None,
        order=None,
        reject=False,
    )
    namespace = cache_namespace or f"judge-validation:{stage}:{model_name}:{candidate}"
    request_hash = make_request_hash(
        model_name,
        prompt,
        options,
        schema,
        template.sha256,
        namespace,
    )
    return {
        "run_id": run_id,
        "stage": stage,
        "candidate": candidate,
        "model": model_name,
        "example_id": example["example_id"],
        "prompt_id": example["prompt_id"],
        "rendered_prompt": prompt,
        "model_options": options,
        "response_schema": schema,
        "cache_namespace": namespace,
        "rendered_prompt_hash": _sha256_text(prompt),
        "prompt_template_hash": template.sha256,
        "rubric_hash": _sha256_text(rubric_json),
        "schema_hash": _sha256_text(_canonical_json(schema)),
        "model_options_hash": _sha256_text(_canonical_json(options)),
        "request_hash": request_hash,
        "estimated_input_tokens": context["estimated_input_tokens"],
        "context_length": context["context_length"],
        "reserved_output_tokens": context["max_output_tokens"],
        "safety_margin_tokens": context["safety_margin_tokens"],
        "estimated_context_headroom": context["estimated_headroom_tokens"],
        "context_preflight_ok": context["allowed"],
        "context_failure_reason": (
            None if context["allowed"] else "estimated_input_exceeds_context_budget"
        ),
        "token_estimate_exact": False,
        "token_estimator": context["token_estimate_method"],
        "context_preflight": context,
    }


def sanitized_request_plan(request: dict[str, Any], cache_status: str) -> dict[str, Any]:
    """Remove dataset text and execution-only fields from a request plan."""
    fields = (
        "candidate",
        "model",
        "example_id",
        "prompt_id",
        "request_hash",
        "rendered_prompt_hash",
        "prompt_template_hash",
        "rubric_hash",
        "schema_hash",
        "model_options_hash",
        "estimated_input_tokens",
        "context_length",
        "reserved_output_tokens",
        "safety_margin_tokens",
        "estimated_context_headroom",
        "context_preflight_ok",
        "context_failure_reason",
        "token_estimate_exact",
        "token_estimator",
    )
    return {**{field: request[field] for field in fields}, "cache_status": cache_status}


def plan_judge_requests(
    config: dict[str, Any],
    *,
    run_id: str,
    stage: str,
    candidates: list[str],
    model_name: str,
    rows: list[dict[str, Any]],
    cache_database: Path | None = None,
    cache_available: bool = True,
) -> dict[str, Any]:
    """Plan judge requests without constructing or contacting an Ollama client."""
    initial_value = read_json(resolve_project_path(config, config["rubric"]["initial"]))
    rubric = validate_rubric_v2(initial_value, initial_value)
    requests: list[dict[str, Any]] = []
    prompt_hashes: dict[str, str] = {}
    for candidate in candidates:
        template = load_template(config, candidate)
        prompt_hashes[candidate] = template.sha256
        for example in rows:
            request = build_judge_request(
                config,
                run_id=run_id,
                stage=stage,
                candidate=candidate,
                model_name=model_name,
                rubric=rubric,
                template=template,
                example=example,
            )
            if not cache_available:
                cache_status = "unavailable"
            elif cache_database is None:
                cache_status = "fresh"
            else:
                cache_status = read_only_cache_status(cache_database, request["request_hash"])
            requests.append(sanitized_request_plan(request, cache_status))
    logical = len(requests)
    invalid = sum(not row["context_preflight_ok"] for row in requests)
    cache_hits = sum(row["cache_status"] == "hit" for row in requests)
    unavailable = sum(row["cache_status"] == "unavailable" for row in requests)
    fresh = sum(row["cache_status"] == "fresh" for row in requests)
    first = requests[0] if requests else {}
    return {
        "stage": stage,
        "run_id": run_id,
        "inference_executed": False,
        "executed": False,
        "actual_calls": 0,
        "ready": invalid == 0 and unavailable == 0 and logical > 0,
        "logical_calls": logical,
        "planned_calls": logical,
        "fresh_calls": fresh,
        "cache_hits": cache_hits,
        "cache_unavailable": unavailable,
        "invalid_requests": invalid,
        "maximum_estimated_input_tokens": max(
            (row["estimated_input_tokens"] for row in requests), default=0
        ),
        "minimum_estimated_headroom": min(
            (row["estimated_context_headroom"] for row in requests), default=None
        ),
        "token_estimate_exact": first.get("token_estimate_exact"),
        "token_estimator": first.get("token_estimator"),
        "prompt_template_hashes": prompt_hashes,
        "rubric_hash": first.get("rubric_hash"),
        "schema_hash": first.get("schema_hash"),
        "model_options_hash": first.get("model_options_hash"),
        "model": model_name,
        "candidates": list(candidates),
        "requests": requests,
    }


def plan_judge_validation(
    config: dict[str, Any],
    run_id: str,
    *,
    candidate: str | None = None,
    model: str = "llama",
) -> dict[str, Any]:
    if model not in {"llama", "qwen"}:
        raise ValueError("model must be llama or qwen")
    candidates = [candidate] if candidate else list(config["judge_validation"]["candidates"])
    _validate_candidates(config, candidates)
    model_name = config["models"]["judge"] if model == "llama" else config["models"]["updater"]
    dev, validation_count, fingerprint, exclusion_report = _planning_dev_rows(config, run_id)
    run_dir = run_directory(config, run_id)
    cache_available = _run_cache_is_compatible(config, run_dir)
    database = run_dir / "calls.sqlite3" if run_dir.exists() else None
    plan = plan_judge_requests(
        config,
        run_id=run_id,
        stage=DEV_STAGE,
        candidates=candidates,
        model_name=model_name,
        rows=dev,
        cache_database=database,
        cache_available=cache_available,
    )
    return {
        **plan,
        "dev_responses": len(dev),
        "validation_responses": validation_count,
        "estimate": estimate_judge_validation(config),
        "dataset_fingerprint": fingerprint,
        "exclusions": exclusion_report,
        "config_hash": _object_hash(public_config(config)),
    }


def run_judge_dev(
    config: dict[str, Any],
    run_id: str,
    *,
    execute_model_calls: bool = False,
    max_calls: int = 0,
    candidate: str | None = None,
    model: str = "llama",
    allow_qwen_fallback: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    candidates = [candidate] if candidate else list(config["judge_validation"]["candidates"])
    _validate_candidates(config, candidates)
    model_name = _validation_model(config, model, allow_qwen_fallback)
    if model == "qwen" and execute_model_calls:
        llama_summary = summarize_judge_dev(config, run_id, model_name=config["models"]["judge"])
        expected = len(config["judge_validation"]["candidates"])
        complete = [row for row in llama_summary["candidates"] if row["complete"]]
        if len(complete) != expected or any(row["gate"]["passed"] for row in complete):
            raise ValueError(
                "Qwen fallback is allowed only after both complete Llama candidates fail the gate"
            )
    if not execute_model_calls:
        plan = plan_judge_validation(
            config,
            run_id,
            candidate=candidate,
            model=model,
        )
        details = {
            **plan,
            "max_calls": max_calls,
            "execution_requested": False,
        }
        path = _write_judge_dry_run(config, run_id, DEV_STAGE, details)
        return {**details, "dry_run_manifest": str(path)}
    if max_calls <= 0:
        raise ValueError("judge-dev execution requires --max-calls greater than zero")
    result = _run_subset(
        config,
        run_id,
        subset_name="judge_dev",
        stage=DEV_STAGE,
        candidates=candidates,
        model_name=model_name,
        max_calls=max_calls,
        client=client,
    )
    report = summarize_judge_dev(config, run_id, model_name=model_name)
    return {**result, "dev_summary": report}


def summarize_judge_dev(
    config: dict[str, Any], run_id: str, *, model_name: str | None = None
) -> dict[str, Any]:
    run_dir = run_directory(config, run_id)
    path = run_dir / "artifacts" / "judge_validation_predictions.jsonl"
    rows = read_jsonl(path) if path.exists() else []
    rows = [row for row in rows if row["stage"] == DEV_STAGE]
    if model_name is not None:
        rows = [row for row in rows if row["model"] == model_name]
    expected = 5 * config["judge_validation"]["dev_per_score"]
    candidate_rows: list[dict[str, Any]] = []
    for (model, candidate), group in sorted(_group(rows, ("model", "candidate")).items()):
        summary = evaluate_judge_predictions(group, expected_count=expected)
        template = load_template(config, candidate)
        row = {
            "model": model,
            "candidate": candidate,
            "prompt_sha256": template.sha256,
            **summary,
        }
        row["gate"] = judge_gate(row, config["judge_validation"]["gate"])
        candidate_rows.append(row)
    complete = [row for row in candidate_rows if row["complete"]]
    chosen = choose_judge_candidate(complete) if complete else None
    gate = judge_gate(chosen, config["judge_validation"]["gate"]) if chosen else _missing_gate()
    disagreements = _dev_disagreements(config, run_id, rows, chosen) if chosen else []
    payload = {
        "stage": DEV_STAGE,
        "candidates": candidate_rows,
        "chosen": chosen,
        "gate": gate,
        "ready_for_untouched_validation": bool(chosen and gate["passed"]),
        "largest_dev_disagreements": disagreements,
    }
    write_json(run_dir / "metrics" / "judge_dev_metrics.json", payload)
    return payload


def run_judge_holdout(
    config: dict[str, Any],
    run_id: str,
    *,
    execute_model_calls: bool = False,
    max_calls: int = 0,
    client: Any | None = None,
) -> dict[str, Any]:
    run_dir = run_directory(config, run_id)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"judge-validation run is not prepared: {run_dir}")
    dev = summarize_judge_dev(config, run_id)
    chosen = dev.get("chosen")
    if not chosen or not dev["gate"]["passed"]:
        raise ValueError("judge-dev has no gate-passing frozen candidate")
    if not execute_model_calls:
        subset_ids = read_jsonl(run_dir / "subsets" / "judge_validation.jsonl")
        rows = _reload_subset_rows(config, subset_ids)
        plan = plan_judge_requests(
            config,
            run_id=run_id,
            stage=VALIDATION_STAGE,
            candidates=[chosen["candidate"]],
            model_name=chosen["model"],
            rows=rows,
            cache_database=run_dir / "calls.sqlite3",
            cache_available=_run_cache_is_compatible(config, run_dir),
        )
        manifest = read_json(manifest_path)
        details = {
            **plan,
            "candidate": chosen["candidate"],
            "max_calls": max_calls,
            "execution_requested": False,
            "dev_responses": manifest.get("dev_response_count"),
            "validation_responses": len(rows),
            "estimate": estimate_judge_validation(config),
            "dataset_fingerprint": manifest.get("dataset_fingerprint"),
            "exclusions": manifest.get("exclusions", {}),
            "config_hash": _object_hash(public_config(config)),
        }
        path = _write_judge_dry_run(config, run_id, VALIDATION_STAGE, details)
        return {**details, "dry_run_manifest": str(path)}
    if max_calls <= 0:
        raise ValueError("judge-validation execution requires --max-calls greater than zero")
    result = _run_subset(
        config,
        run_id,
        subset_name="judge_validation",
        stage=VALIDATION_STAGE,
        candidates=[chosen["candidate"]],
        model_name=chosen["model"],
        max_calls=max_calls,
        client=client,
    )
    summary = finalize_judge_validation(config, run_id)
    return {**result, "judge_validation_summary": summary}


def finalize_judge_validation(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    run_dir = run_directory(config, run_id)
    dev = summarize_judge_dev(config, run_id)
    chosen = dev.get("chosen")
    if not chosen:
        raise ValueError("cannot finalize without a selected dev candidate")
    predictions = read_jsonl(run_dir / "artifacts" / "judge_validation_predictions.jsonl")
    rows = [
        row
        for row in predictions
        if row["stage"] == VALIDATION_STAGE
        and row["model"] == chosen["model"]
        and row["candidate"] == chosen["candidate"]
    ]
    expected = 5 * config["judge_validation"]["validation_per_score"]
    metrics = evaluate_judge_predictions(rows, expected_count=expected)
    gate = judge_gate(metrics, config["judge_validation"]["gate"])
    verdict = "JUDGE_APPROVED" if gate["passed"] else "JUDGE_NOT_APPROVED"
    template = load_template(config, chosen["candidate"])
    frozen = {
        "model": chosen["model"],
        "candidate": chosen["candidate"],
        "prompt_sha256": template.sha256,
        "model_options": judge_options(config),
    }
    payload = {
        "verdict": verdict,
        "chosen": frozen,
        "dev": dev,
        "untouched_validation": metrics,
        "gate": gate,
        "threshold_scope": "internal project gate, not a universal standard",
    }
    write_json(run_dir / "judge_validation_summary.json", payload)
    write_json(run_dir / "verdict.json", {"verdict": verdict})
    return payload


def evaluate_judge_predictions(
    rows: list[dict[str, Any]], *, expected_count: int | None = None
) -> dict[str, Any]:
    keys = [row["example_id"] for row in rows]
    duplicate_count = len(keys) - len(set(keys))
    unique = {row["example_id"]: row for row in rows}
    deduplicated = list(unique.values())
    metrics = agreement_metrics(deduplicated) if deduplicated else _empty_metrics()
    confusion = [[0 for _ in range(5)] for _ in range(5)]
    for row in deduplicated:
        confusion[int(row["human_score"])][int(row["predicted_score"])] += 1
    predicted_counts = Counter(int(row["predicted_score"]) for row in deduplicated)
    complete = expected_count is None or (
        len(deduplicated) == expected_count and duplicate_count == 0
    )
    return {
        **metrics,
        "confusion_matrix": confusion,
        "human_score_distribution": {
            str(score): sum(int(row["human_score"]) == score for row in deduplicated)
            for score in range(5)
        },
        "predicted_score_distribution": {
            str(score): predicted_counts.get(score, 0) for score in range(5)
        },
        "constant_predictions": len(predicted_counts) <= 1 and bool(deduplicated),
        "undefined_metrics": [
            name for name in ("quadratic_weighted_kappa", "spearman") if metrics[name] is None
        ],
        "prediction_count": len(deduplicated),
        "expected_count": expected_count,
        "duplicate_predictions": duplicate_count,
        "missing_predictions": max(0, (expected_count or len(deduplicated)) - len(deduplicated)),
        "unresolved_schema_errors": 0,
        "complete": complete,
    }


def choose_judge_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no complete judge candidates are available")

    def key(row: dict[str, Any]) -> tuple[float, float, float, str, str]:
        qwk = row.get("quadratic_weighted_kappa")
        spearman = row.get("spearman")
        return (
            -(float(qwk) if qwk is not None else float("-inf")),
            -(float(spearman) if spearman is not None else float("-inf")),
            float(row["mean_absolute_error"]),
            str(row.get("candidate", "")),
            str(row.get("model", "")),
        )

    return dict(sorted(rows, key=key)[0])


def judge_gate(metrics: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    reasons: list[str] = []
    qwk = metrics.get("quadratic_weighted_kappa")
    spearman = metrics.get("spearman")
    mae = metrics.get("mean_absolute_error")
    if qwk is None:
        reasons.append("undefined_qwk")
    if spearman is None:
        reasons.append("undefined_spearman")
    if metrics.get("constant_predictions"):
        reasons.append("constant_predictions")
    if metrics.get("missing_predictions", 0):
        reasons.append("missing_predictions")
    if metrics.get("unresolved_schema_errors", 0):
        reasons.append("unresolved_schema_errors")
    if qwk is not None and qwk < thresholds["minimum_qwk"]:
        reasons.append("qwk_below_threshold")
    if spearman is not None and spearman < thresholds["minimum_spearman"]:
        reasons.append("spearman_below_threshold")
    if mae is None or mae > thresholds["maximum_mae"]:
        reasons.append("mae_above_threshold")
    return {"passed": not reasons, "reasons": reasons, "thresholds": dict(thresholds)}


def _run_subset(
    config: dict[str, Any],
    run_id: str,
    *,
    subset_name: str,
    stage: str,
    candidates: list[str],
    model_name: str,
    max_calls: int,
    client: Any | None,
) -> dict[str, Any]:
    run_dir = run_directory(config, run_id)
    subset_ids = read_jsonl(run_dir / "subsets" / f"{subset_name}.jsonl")
    rows = _reload_subset_rows(config, subset_ids)
    initial_value = read_json(resolve_project_path(config, config["rubric"]["initial"]))
    rubric = validate_rubric_v2(initial_value, initial_value)
    raw_client = client or OllamaClient(
        config["models"]["base_url"], config["models"]["timeout_seconds"]
    )
    health = raw_client.health_and_models([model_name])
    if health["missing"]:
        raise RuntimeError(f"required judge model is not installed: {model_name}")
    budget = InferenceBudgetClient(raw_client, max_calls)
    output_path = run_dir / "artifacts" / "judge_validation_predictions.jsonl"
    existing = read_jsonl(output_path) if output_path.exists() else []
    completed = {
        (row["stage"], row["model"], row["candidate"], row["example_id"]) for row in existing
    }
    cache_hits = 0
    try:
        with CallStore(run_dir / "calls.sqlite3") as store:
            stop = False
            for candidate in candidates:
                template = load_template(config, candidate)
                additions: list[dict[str, Any]] = []
                for row in rows:
                    key = (stage, model_name, candidate, row["example_id"])
                    if key in completed:
                        continue
                    if budget.calls >= max_calls:
                        stop = True
                        break
                    request = build_judge_request(
                        config,
                        run_id=run_id,
                        stage=stage,
                        candidate=candidate,
                        model_name=model_name,
                        rubric=rubric,
                        template=template,
                        example=row,
                    )
                    if not request["context_preflight_ok"]:
                        raise ValueError(
                            "judge request failed context preflight: "
                            f"example_id={row['example_id']}, candidate={candidate}, "
                            f"headroom={request['estimated_context_headroom']}"
                        )
                    parsed, cached = cached_validated_call(
                        client=budget,
                        store=store,
                        context=CallContext(run_id, stage, candidate, None, None, None, 0),
                        model_name=model_name,
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
                    additions.append(
                        {
                            "stage": stage,
                            "model": model_name,
                            "candidate": candidate,
                            "prompt_sha256": template.sha256,
                            "request_hash": request["request_hash"],
                            "schema_version": config["schema_version"],
                            "prompt_version": config["prompt_version"],
                            "example_id": row["example_id"],
                            "prompt_id": row["prompt_id"],
                            "human_score": row["human_score"],
                            "predicted_score": parsed["score"],
                            "cache_hit": cached,
                        }
                    )
                merge_index(
                    output_path,
                    additions,
                    ("stage", "model", "candidate", "example_id"),
                )
                if stop:
                    break
    finally:
        if config["models"].get("unload_after_stage", True):
            raw_client.unload(model_name)
        _sync_call_count(run_dir)
    return {
        "executed": True,
        "actual_calls": budget.calls,
        "cache_hits": cache_hits,
        "resume_supported": True,
        "complete": not stop,
    }


def _reload_subset_rows(
    config: dict[str, Any], subset_ids: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    _train, validation = load_dataset(config)
    if validation is None:
        raise ValueError("official validation split is unavailable")
    lookup = {row["example_id"]: row for row in validation}
    rows: list[dict[str, Any]] = []
    for item in subset_ids:
        row = lookup.get(item["example_id"])
        if row is None or _row_hash(row) != item["row_sha256"]:
            raise ValueError(f"judge subset row hash mismatch: {item['example_id']}")
        rows.append(row)
    return rows


def compact_judge_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Return the terminal/notebook summary; full request records stay in the manifest."""
    fields = (
        "stage",
        "run_id",
        "executed",
        "ready",
        "logical_calls",
        "planned_calls",
        "fresh_calls",
        "cache_hits",
        "cache_unavailable",
        "invalid_requests",
        "maximum_estimated_input_tokens",
        "minimum_estimated_headroom",
        "prompt_template_hashes",
        "rubric_hash",
        "schema_hash",
        "model_options_hash",
        "dry_run_manifest",
    )
    return {field: plan[field] for field in fields if field in plan}


def _planning_dev_rows(
    config: dict[str, Any], run_id: str
) -> tuple[list[dict[str, Any]], int, str, dict[str, Any]]:
    """Recreate preparation read-only, or verify and reload an existing prepared subset."""
    train, validation = load_dataset(config)
    if validation is None:
        raise ValueError("judge validation requires an official validation split")
    fingerprint = dataset_fingerprint(train, validation)
    run_dir = run_directory(config, run_id)
    if run_dir.exists():
        manifest_path = run_dir / "manifest.json"
        dev_path = run_dir / "subsets" / "judge_dev.jsonl"
        holdout_path = run_dir / "subsets" / "judge_validation.jsonl"
        if not manifest_path.exists() or not dev_path.exists() or not holdout_path.exists():
            raise ValueError("existing judge-validation directory is not a prepared run")
        manifest = read_json(manifest_path)
        if manifest.get("workflow") != "judge_validation" or manifest.get("run_id") != run_id:
            raise ValueError("existing directory is not the requested judge-validation run")
        if manifest.get("dataset_fingerprint") != fingerprint:
            raise ValueError("existing judge-validation run uses a different dataset")
        dev = _reload_subset_rows(config, read_jsonl(dev_path))
        return (
            dev,
            len(read_jsonl(holdout_path)),
            fingerprint,
            dict(manifest.get("exclusions", {})),
        )

    excluded_examples, excluded_prompts, exclusion_report = load_exclusions(config)
    eligible, removal = apply_exclusions(validation, excluded_examples, excluded_prompts)
    settings = config["judge_validation"]
    dev, holdout = select_judge_subsets(
        eligible,
        dev_per_score=settings["dev_per_score"],
        validation_per_score=settings["validation_per_score"],
        seed=config["seed"],
    )
    return dev, len(holdout), fingerprint, {**exclusion_report, "validation": removal}


def _run_cache_is_compatible(config: dict[str, Any], run_dir: Path) -> bool:
    if not run_dir.exists():
        return True
    resolved_path = run_dir / "config.resolved.json"
    manifest_path = run_dir / "manifest.json"
    if not resolved_path.exists() or not manifest_path.exists():
        return False
    try:
        resolved = read_json(resolved_path)
        manifest = read_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        manifest.get("workflow") == "judge_validation"
        and _object_hash(resolved) == _object_hash(public_config(config))
    )


def _write_judge_dry_run(
    config: dict[str, Any], run_id: str, stage: str, plan: dict[str, Any]
) -> Path:
    dry_dir = Path(config["_project_root"]) / "dry_runs" / run_id
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = dry_dir / f"{stage}_{stamp}.json"
    payload = {
        **plan,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inference_executed": False,
        "model_responses": 0,
        "prompt_templates": {
            candidate: {
                "path": config["prompts"][candidate],
                "sha256": digest,
            }
            for candidate, digest in plan["prompt_template_hashes"].items()
        },
    }
    write_json(path, payload)
    return path


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _object_hash(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


def _validation_model(config: dict[str, Any], model: str, allow_qwen_fallback: bool) -> str:
    if model == "llama":
        return config["models"]["judge"]
    if model != "qwen":
        raise ValueError("model must be llama or qwen")
    if not allow_qwen_fallback:
        raise ValueError("Qwen fallback requires explicit allow_qwen_fallback=True")
    return config["models"]["updater"]


def _dev_disagreements(
    config: dict[str, Any],
    run_id: str,
    prediction_rows: list[dict[str, Any]],
    chosen: dict[str, Any],
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in prediction_rows
        if row["model"] == chosen["model"] and row["candidate"] == chosen["candidate"]
    ]
    subset_ids = read_jsonl(run_directory(config, run_id) / "subsets" / "judge_dev.jsonl")
    content = {row["example_id"]: row for row in _reload_subset_rows(config, subset_ids)}
    largest = sorted(
        selected,
        key=lambda row: (
            -abs(int(row["predicted_score"]) - int(row["human_score"])),
            row["example_id"],
        ),
    )[:10]
    return [
        {
            "example_id": row["example_id"],
            "prompt_id": row["prompt_id"],
            "human_score": row["human_score"],
            "predicted_score": row["predicted_score"],
            "absolute_error": abs(int(row["predicted_score"]) - int(row["human_score"])),
            "prompt_preview": _truncate(content[row["example_id"]]["prompt"]),
            "response_preview": _truncate(content[row["example_id"]]["response"]),
        }
        for row in largest
    ]


def _validate_candidates(config: dict[str, Any], candidates: list[str]) -> None:
    allowed = set(config["judge_validation"]["candidates"])
    if not candidates or not set(candidates) <= allowed:
        raise ValueError(f"judge candidate must be one of {sorted(allowed)}")


def _concise_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "example_id": row["example_id"],
        "prompt_id": row["prompt_id"],
        "human_score": row["human_score"],
        "row_sha256": _row_hash(row),
    }


def _row_hash(row: dict[str, Any]) -> str:
    text = json.dumps(
        [row["example_id"], row["prompt_id"], row["prompt"], row["response"], row["human_score"]],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _assert_subset_contract(rows: list[dict[str, Any]], per_score: int) -> None:
    if _label_counts(rows) != {str(score): per_score for score in range(5)}:
        raise AssertionError("judge subset is not exactly label-balanced")
    if len({row["prompt_id"] for row in rows}) != len(rows):
        raise AssertionError("judge subset contains more than one response per prompt")


def _label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {str(score): sum(row["human_score"] == score for row in rows) for score in range(5)}


def _subset_sort_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return row["human_score"], row["prompt_id"], row["example_id"]


def _group(
    rows: list[dict[str, Any]], fields: tuple[str, ...]
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    result: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[tuple(row[field] for field in fields)].append(row)
    return result


def _empty_metrics() -> dict[str, None]:
    return {
        "quadratic_weighted_kappa": None,
        "spearman": None,
        "mean_absolute_error": None,
        "mean_signed_error": None,
    }


def _missing_gate() -> dict[str, Any]:
    return {"passed": False, "reasons": ["no_complete_candidate"], "thresholds": {}}


def _sync_call_count(run_dir: Path) -> None:
    manifest = read_json(run_dir / "manifest.json")
    database = run_dir / "calls.sqlite3"
    import sqlite3

    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro&immutable=1", uri=True)
    try:
        manifest["model_calls_made"] = int(
            connection.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
        )
    finally:
        connection.close()
    write_json(run_dir / "manifest.json", manifest)


def _truncate(text: str, length: int = 240) -> str:
    return text if len(text) <= length else text[: length - 1] + "…"
