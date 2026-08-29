from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import resolve_project_path
from .context import normalize_text, sha256_text
from .data import (
    dataset_fingerprint,
    example_id_for_record,
    load_dataset,
    load_dataset_with_lineage,
    prompt_id_for_text,
)
from .judging import build_pilot_judge_request
from .prompts import load_template
from .run import file_sha256, read_json, read_jsonl, write_json


def resolve_dataset_lineage(
    config: dict[str, Any],
    *,
    prompt_id: str | None = None,
    example_id: str | None = None,
    predictions_path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve internal IDs with the project's canonical loader, normalization, and hashes."""
    if prompt_id is None and example_id is None:
        raise ValueError("provide prompt_id, example_id, or both")
    train, validation = load_dataset(config)
    lineage_train, lineage_validation = load_dataset_with_lineage(config)
    all_lineage = lineage_train + (lineage_validation or [])
    matches = [
        row
        for row in all_lineage
        if (prompt_id is None or row["normalized_record"]["prompt_id"] == prompt_id)
        and (example_id is None or row["normalized_record"]["example_id"] == example_id)
    ]
    prediction_context = (
        _load_prediction_context(config, Path(predictions_path).resolve())
        if predictions_path is not None
        else None
    )
    records = [_resolve_record(config, row, prediction_context) for row in matches]
    return {
        "query": {"prompt_id": prompt_id, "example_id": example_id},
        "match_count": len(records),
        "dataset_fingerprint": dataset_fingerprint(train, validation),
        "dataset_path": str(resolve_project_path(config, config["dataset"]["path"])),
        "column_mapping": dict(config["dataset"]["columns"]),
        "records": records,
        "read_only": True,
        "files_modified": 0,
        "ollama_contacted": False,
    }


def audit_extreme_pair_lineage(
    config: dict[str, Any],
    *,
    prompt_ids: Sequence[str],
    example_ids: Sequence[str],
    predictions_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Audit an explicit set of prompt/example IDs and optionally write JSON/Markdown reports."""
    unique_examples = list(dict.fromkeys(example_ids))
    results = [
        resolve_dataset_lineage(
            config,
            example_id=value,
            predictions_path=predictions_path,
        )
        for value in unique_examples
    ]
    records = [row for result in results for row in result["records"]]
    found_examples = {row["example_id"] for row in records}
    found_prompts = {row["prompt_id"] for row in records}
    failures: list[str] = []
    if found_examples != set(unique_examples):
        failures.append("one or more example IDs did not reconstruct uniquely")
    if found_prompts != set(prompt_ids):
        failures.append("reconstructed prompt IDs differ from the requested prompt set")
    for row in records:
        checks = row["checks"]
        required = (
            "prompt_id_recalculated_matches",
            "example_id_recalculated_matches",
            "prepared_prompt_matches_processed_source",
            "prepared_response_matches_processed_source",
            "prepared_human_score_matches_helpfulness",
            "prediction_prompt_ids_match",
            "prediction_human_scores_match_helpfulness",
            "request_hashes_reconstructed_match",
            "rendered_prompt_hashes_match_plan",
            "successful_call_prompt_hashes_match",
            "judge_backend_reported_no_prompt_truncation",
            "responses_not_swapped",
        )
        failures.extend(
            f"{row['example_id']}: {name}" for name in required if checks.get(name) is not True
        )
    verdict = "DATA_LINEAGE_APPROVED" if not failures else "DATA_MAPPING_BUG"
    report = {
        "verdict": verdict,
        "prompt_ids": list(prompt_ids),
        "example_ids": unique_examples,
        "dataset_fingerprint": results[0]["dataset_fingerprint"] if results else None,
        "records": records,
        "failures": failures,
        "hashing_implementation": (
            "rubric_order.data.prompt_id_for_text/example_id_for_record, called by the "
            "canonical normalize_records loader"
        ),
        "prediction_artifact": str(Path(predictions_path).resolve()),
        "prediction_artifact_sha256": file_sha256(predictions_path),
        "read_only": True,
        "ollama_contacted": False,
    }
    if output_dir is not None:
        destination = Path(output_dir).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        json_path = destination / "extreme_pair_lineage_audit.json"
        markdown_path = destination / "extreme_pair_lineage_audit.md"
        write_json(json_path, report)
        markdown_path.write_text(_lineage_markdown(report), encoding="utf-8")
        report["artifacts"] = {
            "json": str(json_path),
            "markdown": str(markdown_path),
        }
    return report


def _resolve_record(
    config: dict[str, Any],
    lineage: dict[str, Any],
    prediction_context: dict[str, Any] | None,
) -> dict[str, Any]:
    raw = lineage["raw_record"]
    normalized = lineage["normalized_record"]
    source = lineage["source_split"]
    index = int(lineage["source_row_index"])
    columns = config["dataset"]["columns"]
    label_column = columns["label"]
    recalculated_prompt = prompt_id_for_text(raw[columns["prompt"]])
    recalculated_example = example_id_for_record(
        source,
        index,
        raw[columns["prompt"]],
        raw[columns["response"]],
        normalized["human_score"],
    )
    prepared = None
    predictions: list[dict[str, Any]] = []
    verification = _empty_prediction_verification()
    if prediction_context is not None:
        prepared = prediction_context["heldout"].get(normalized["example_id"])
        predictions = prediction_context["predictions"].get(normalized["example_id"], [])
        verification = _verify_prediction_rows(
            config,
            normalized,
            raw,
            prepared,
            predictions,
            prediction_context,
        )
    processed_prompt = normalize_text(normalized["prompt"])
    processed_response = normalize_text(normalized["response"])
    processing = prepared.get("text_processing", {}) if prepared else {}
    other_dimensions = {
        key: value
        for key, value in raw.items()
        if key not in {columns["prompt"], columns["response"], label_column}
    }
    checks = {
        "prompt_id_recalculated_matches": recalculated_prompt == normalized["prompt_id"],
        "example_id_recalculated_matches": recalculated_example == normalized["example_id"],
        "prepared_prompt_matches_processed_source": bool(
            prepared and prepared["prompt"] == processed_prompt
        ),
        "prepared_response_matches_processed_source": bool(
            prepared and prepared["response"] == processed_response
        ),
        "prepared_human_score_matches_helpfulness": bool(
            prepared and prepared["human_score"] == raw[label_column]
        ),
        **verification["checks"],
    }
    return {
        "source_dataset": lineage["source_dataset"],
        "source_split": source,
        "source_row_index": index,
        "prompt_id": normalized["prompt_id"],
        "example_id": normalized["example_id"],
        "recalculated_prompt_id": recalculated_prompt,
        "recalculated_example_id": recalculated_example,
        "prompt": raw[columns["prompt"]],
        "response": raw[columns["response"]],
        "original_helpfulness": raw[label_column],
        "prediction_human_score": (
            sorted({row["human_score"] for row in predictions}) if predictions else []
        ),
        "other_helpsteer2_dimensions": other_dimensions,
        "prepared_prompt": prepared.get("prompt") if prepared else None,
        "prepared_response": prepared.get("response") if prepared else None,
        "text_processing": processing,
        "normalization": {
            "prompt_modified": normalized["prompt"] != processed_prompt,
            "response_modified": normalized["response"] != processed_response,
            "prompt_original_sha256": sha256_text(normalized["prompt"]),
            "prompt_normalized_sha256": sha256_text(processed_prompt),
            "response_original_sha256": sha256_text(normalized["response"]),
            "response_normalized_sha256": sha256_text(processed_response),
        },
        "prediction_summary": verification["summary"],
        "checks": checks,
    }


def _load_prediction_context(config: dict[str, Any], path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"prediction artifact does not exist: {path}")
    settings = config.get("downstream_audit", {})
    source = resolve_project_path(config, settings.get("source_run", "runs/pilot-002"))
    heldout_rows = read_jsonl(source / "prepared" / "heldout.jsonl")
    rubric_rows = read_jsonl(source / "artifacts" / "rubrics.jsonl")
    predictions = read_jsonl(path)
    request_plan_path = path.parent / "request_plans" / "judge.jsonl"
    request_plan = read_jsonl(request_plan_path) if request_plan_path.exists() else []
    database = path.parents[1] / "calls.sqlite3"
    return {
        "source_run": source,
        "heldout": {row["example_id"]: row for row in heldout_rows},
        "heldout_by_prompt": _group_by(heldout_rows, "prompt_id"),
        "rubrics": {
            (row["method"], row["block_id"], row["order"], row["replicate"]): row
            for row in rubric_rows
        },
        "predictions": _group_by(predictions, "example_id"),
        "request_plan": {row["request_hash"]: row for row in request_plan},
        "database": database,
        "template": load_template(config, "judge"),
    }


def _verify_prediction_rows(
    config: dict[str, Any],
    normalized: dict[str, Any],
    raw: dict[str, Any],
    prepared: dict[str, Any] | None,
    predictions: list[dict[str, Any]],
    context: dict[str, Any],
) -> dict[str, Any]:
    reconstructed: list[dict[str, Any]] = []
    for prediction in predictions:
        key = (
            prediction["method"],
            prediction["block_id"],
            prediction["order"],
            prediction["replicate"],
        )
        rubric_row = context["rubrics"].get(key)
        if rubric_row is None or prepared is None:
            continue
        rubric = read_json(context["source_run"] / rubric_row["rubric_path"])
        request = build_pilot_judge_request(
            config,
            run_id="pilot-002-downstream-audit",
            rubric_row=rubric_row,
            rubric=rubric,
            template=context["template"],
            example=prepared,
        )
        plan = context["request_plan"].get(prediction["request_hash"])
        reconstructed.append(
            {
                "prediction": prediction,
                "request": request,
                "plan": plan,
            }
        )
    expected_prompt_hashes = {
        row["prediction"]["request_hash"]: row["request"]["rendered_prompt_hash"]
        for row in reconstructed
    }
    call_rows = _read_call_verification(context["database"], expected_prompt_hashes)
    label_column = config["dataset"]["columns"]["label"]
    request_hashes_match = bool(reconstructed) and all(
        row["request"]["request_hash"] == row["prediction"]["request_hash"] for row in reconstructed
    )
    rendered_hashes_match = bool(reconstructed) and all(
        row["plan"] is not None
        and row["request"]["rendered_prompt_hash"] == row["plan"]["rendered_prompt_hash"]
        for row in reconstructed
    )
    call_hashes_match = bool(call_rows) and all(
        row["status"] == "success" and row["prompt_sha256"] == row["expected_prompt_sha256"]
        for row in call_rows
    )
    no_backend_truncation = bool(call_rows) and all(
        row["backend_prompt_truncated"] == 0 for row in call_rows
    )
    same_prompt_responses = {
        row["example_id"]: row["response"]
        for row in context["heldout_by_prompt"].get(normalized["prompt_id"], [])
    }
    prepared_response = prepared.get("response") if prepared else None
    response_owner = [
        example
        for example, response in same_prompt_responses.items()
        if normalize_text(response) == prepared_response
    ]
    return {
        "checks": {
            "prediction_prompt_ids_match": bool(predictions)
            and all(row["prompt_id"] == normalized["prompt_id"] for row in predictions),
            "prediction_human_scores_match_helpfulness": bool(predictions)
            and all(row["human_score"] == raw[label_column] for row in predictions),
            "request_hashes_reconstructed_match": request_hashes_match,
            "rendered_prompt_hashes_match_plan": rendered_hashes_match,
            "successful_call_prompt_hashes_match": call_hashes_match,
            "judge_backend_reported_no_prompt_truncation": no_backend_truncation,
            "responses_not_swapped": response_owner == [normalized["example_id"]],
        },
        "summary": {
            "prediction_count": len(predictions),
            "condition_count_reconstructed": len(reconstructed),
            "predicted_scores": sorted({row["predicted_score"] for row in predictions}),
            "request_hash_count": len({row["request_hash"] for row in predictions}),
            "successful_call_count": len(call_rows),
            "response_owner_example_ids": response_owner,
        },
    }


def _read_call_verification(
    database: Path, expected_prompt_hashes: dict[str, str]
) -> list[dict[str, Any]]:
    if not database.exists() or not expected_prompt_hashes:
        return []
    request_hashes = sorted(expected_prompt_hashes)
    connection = sqlite3.connect(
        f"file:{database.resolve().as_posix()}?mode=ro&immutable=1", uri=True
    )
    try:
        placeholders = ",".join("?" for _ in request_hashes)
        rows = connection.execute(
            f"SELECT request_hash, status, prompt_sha256, backend_prompt_truncated "
            f"FROM calls WHERE request_hash IN ({placeholders})",
            request_hashes,
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "request_hash": request_hash,
            "status": status,
            "prompt_sha256": prompt_hash,
            "expected_prompt_sha256": expected_prompt_hashes[request_hash],
            "backend_prompt_truncated": backend_truncated,
        }
        for request_hash, status, prompt_hash, backend_truncated in rows
    ]


def _empty_prediction_verification() -> dict[str, Any]:
    return {
        "checks": {
            "prediction_prompt_ids_match": None,
            "prediction_human_scores_match_helpfulness": None,
            "request_hashes_reconstructed_match": None,
            "rendered_prompt_hashes_match_plan": None,
            "successful_call_prompt_hashes_match": None,
            "judge_backend_reported_no_prompt_truncation": None,
            "responses_not_swapped": None,
        },
        "summary": {},
    }


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return grouped


def _lineage_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Extreme-pair data-lineage audit",
        "",
        f"**Verdict:** `{report['verdict']}`",
        "",
        f"Dataset fingerprint: `{report['dataset_fingerprint']}`",
        "",
        "IDs are recalculated through the same functions used by `normalize_records`; no "
        "approximate hashing implementation is used.",
        "",
    ]
    for row in report["records"]:
        dimensions = json.dumps(row["other_helpsteer2_dimensions"], sort_keys=True)
        lines.extend(
            [
                f"## `{row['example_id']}`",
                "",
                f"- Source: `{row['source_split']}` row `{row['source_row_index']}`",
                f"- Prompt ID: `{row['prompt_id']}`",
                f"- Helpfulness: `{row['original_helpfulness']}`; predicted score(s): "
                f"`{row['prediction_summary'].get('predicted_scores', [])}`",
                f"- Other dimensions: `{dimensions}`",
                f"- Prompt normalized: `{row['normalization']['prompt_modified']}`; response "
                f"normalized: `{row['normalization']['response_modified']}`",
                f"- Prompt truncated: `{row['text_processing'].get('prompt_truncated')}`; response "
                f"truncated: `{row['text_processing'].get('response_truncated')}`",
                f"- All lineage checks passed: `{all(row['checks'].values())}`",
                "",
                "### Prompt",
                "",
                row["prompt"],
                "",
                "### Response",
                "",
                row["response"],
                "",
            ]
        )
    if report["failures"]:
        lines.extend(["## Failures", "", *[f"- {value}" for value in report["failures"]], ""])
    return "\n".join(lines)
