from __future__ import annotations

import ast
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from .config import resolve_project_path
from .estimate import Selection, plan_rubric_jobs
from .ollama import judge_options
from .prompts import load_template
from .run import read_json, read_jsonl, run_directory, write_json
from .sampling import assert_order_invariants
from .schemas import validate_rubric_v2


def readiness_report(
    config: dict[str, Any], run_id: str, *, write_report: bool = True
) -> dict[str, Any]:
    """Audit a completed v2 pilot without contacting a model server."""
    run_dir = run_directory(config, run_id)
    failures: list[str] = []
    checks: dict[str, bool] = {}

    def check(name: str, condition: bool, detail: str) -> None:
        checks[name] = bool(condition)
        if not condition:
            failures.append(f"{name}: {detail}")

    manifest_path = run_dir / "manifest.json"
    check("manifest_exists", manifest_path.exists(), "missing manifest.json")
    if not manifest_path.exists():
        return _finish(run_dir, checks, failures, write_report)
    manifest = read_json(manifest_path)
    check("schema_v2", manifest.get("schema_version") == "2.0", "run is not schema v2")
    check(
        "partial_analysis_disabled",
        config.get("allow_partial_results") is False,
        "configuration permits partial results",
    )
    _check_judge_artifact(config, checks, failures)
    blocks_path = run_dir / "prepared" / "calibration_blocks.json"
    heldout_path = run_dir / "prepared" / "heldout.jsonl"
    check(
        "prepared_data_exists",
        blocks_path.exists() and heldout_path.exists(),
        "prepared data missing",
    )
    if not blocks_path.exists() or not heldout_path.exists():
        return _finish(run_dir, checks, failures, write_report)
    blocks = read_json(blocks_path)
    heldout = read_jsonl(heldout_path)
    order_ok = True
    evidence_hashes_ok = True
    for block in blocks:
        try:
            assert_order_invariants(block["orders"])
        except ValueError:
            order_ok = False
        versions = {
            row["example_id"]: (
                row.get("text_processing", {}).get("used_prompt_sha256"),
                row.get("text_processing", {}).get("used_response_sha256"),
            )
            for row in block["examples"]
        }
        evidence_hashes_ok &= all(all(pair) for pair in versions.values())
    check("same_evidence_all_orders", order_ok, "an order changes the calibration multiset")
    check(
        "processed_evidence_hashes_present",
        evidence_hashes_ok,
        "processed text hashes are missing",
    )
    initial = read_json(resolve_project_path(config, config["rubric"]["initial"]))
    jobs = plan_rubric_jobs(config, Selection())
    expected_rubrics = {(job.method, job.block_id, job.order, job.replicate) for job in jobs} | {
        ("static", block["block_id"], order, 0) for block in blocks for order in config["orders"]
    }
    rubric_path = run_dir / "artifacts" / "rubrics.jsonl"
    rubric_rows = read_jsonl(rubric_path) if rubric_path.exists() else []
    rubric_keys = [
        (row["method"], row["block_id"], row["order"], row["replicate"]) for row in rubric_rows
    ]
    check(
        "rubrics_complete_unique",
        set(rubric_keys) == expected_rubrics and len(rubric_keys) == len(set(rubric_keys)),
        f"expected {len(expected_rubrics)} unique rubric keys, found {len(rubric_keys)}",
    )
    valid_rubrics = True
    rubric_hashes_ok = True
    for row in rubric_rows:
        path = run_dir / row["rubric_path"]
        if not path.exists():
            valid_rubrics = rubric_hashes_ok = False
            continue
        try:
            canonical = validate_rubric_v2(read_json(path), initial)
        except ValueError:
            valid_rubrics = False
            continue
        rubric_hashes_ok &= hashlib.sha256(path.read_bytes()).hexdigest() == row["rubric_sha256"]
        compact = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        rubric_hashes_ok &= hashlib.sha256(compact.encode("utf-8")).hexdigest() == row.get(
            "canonical_rubric_sha256"
        )
    check("all_rubrics_valid_v2", valid_rubrics, "invalid core, anchors, duplicates, or schema")
    check("rubric_hashes_match", rubric_hashes_ok, "rubric raw/canonical hash mismatch")
    trajectory_ok = True
    example_count = config["sampling"]["examples_per_score"] * 5
    for job in jobs:
        if job.method != "sequential":
            continue
        directory = (
            run_dir
            / "artifacts"
            / "rubrics"
            / "sequential"
            / f"block_{job.block_id:02d}"
            / job.order
            / f"replicate_{job.replicate}"
        )
        steps = [path for path in directory.glob("step_*.json") if ".meta." not in path.name]
        trajectory_ok &= len(steps) == example_count
    check("sequential_trajectories_complete", trajectory_ok, "a trajectory is not exactly 10 steps")
    prediction_path = run_dir / "artifacts" / "predictions.jsonl"
    predictions = read_jsonl(prediction_path) if prediction_path.exists() else []
    expected_prediction_count = len(expected_rubrics) * len(heldout)
    prediction_keys = [
        (row["method"], row["block_id"], row["order"], row["replicate"], row["example_id"])
        for row in predictions
    ]
    check(
        "predictions_complete_unique",
        len(predictions) == expected_prediction_count
        and len(prediction_keys) == len(set(prediction_keys)),
        f"expected {expected_prediction_count} unique predictions, found {len(predictions)}",
    )
    _check_call_context(run_dir, checks, failures)
    exclusions = manifest.get("exclusions", {})
    check(
        "development_exclusions_respected",
        exclusions.get("all_selected_subsets_disjoint") is True,
        "manifest lacks a positive disjointness assertion",
    )
    _check_metrics(config, run_dir, checks, failures)
    _check_prompt_freeze(config, manifest, checks, failures)
    _check_notebooks(Path(config["_project_root"]), checks, failures)
    return _finish(run_dir, checks, failures, write_report)


def _check_judge_artifact(
    config: dict[str, Any], checks: dict[str, bool], failures: list[str]
) -> None:
    configured = config.get("approved_judge_artifact")
    path = resolve_project_path(config, configured) if configured else None
    artifact = read_json(path) if path and path.exists() else None
    passed = bool(artifact and artifact.get("verdict") == "JUDGE_APPROVED")
    checks["judge_approved"] = passed
    if not passed:
        failures.append("judge_approved: approved judge artifact missing or negative")
        return
    chosen = artifact["chosen"]
    template = load_template(config, "judge")
    match = (
        chosen.get("prompt_sha256") == template.sha256
        and chosen.get("model") == config["models"]["judge"]
        and chosen.get("model_options") == judge_options(config)
    )
    checks["judge_configuration_frozen"] = match
    if not match:
        failures.append("judge_configuration_frozen: prompt, model, or options changed")


def _check_call_context(run_dir: Path, checks: dict[str, bool], failures: list[str]) -> None:
    path = run_dir / "calls.sqlite3"
    if not path.exists():
        checks["call_context_complete"] = False
        failures.append("call_context_complete: calls.sqlite3 is missing")
        return
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(calls)")}
        required = {
            "schema_version",
            "prompt_version",
            "estimated_input_tokens",
            "prompt_eval_count",
            "observed_headroom_tokens",
            "context_saturation",
            "backend_prompt_truncated",
        }
        complete = required <= columns
        bad = 1
        if complete:
            bad = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM calls
                    WHERE status = 'context_error' OR (
                        status = 'success' AND (
                            schema_version != '2.0' OR prompt_version IS NULL
                            OR estimated_input_tokens IS NULL OR prompt_eval_count IS NULL
                            OR observed_headroom_tokens IS NULL OR context_saturation != 0
                            OR backend_prompt_truncated != 0
                        )
                    )
                    """
                ).fetchone()[0]
            )
        checks["call_context_complete"] = complete and bad == 0
        if not complete or bad:
            failures.append(
                f"call_context_complete: missing metadata columns or {bad} invalid successful calls"
            )
    finally:
        connection.close()


def _check_metrics(
    config: dict[str, Any], run_dir: Path, checks: dict[str, bool], failures: list[str]
) -> None:
    summary_path = run_dir / "metrics" / "summary.json"
    repeat_path = run_dir / "metrics" / "repeat_variation.csv"
    between_path = run_dir / "metrics" / "between_vs_repeat.csv"
    metrics_exist = summary_path.exists() and repeat_path.exists() and between_path.exists()
    checks["required_metrics_exist"] = metrics_exist
    if not metrics_exist:
        failures.append("required_metrics_exist: analysis or repeat metrics are missing")
        return
    summary = read_json(summary_path)
    static = summary.get("primary", {}).get("static", {})
    zero = all(
        static.get(metric, {}).get("estimate") == 0
        for metric in (
            "calibration_order_instability",
            "score_change_rate",
            "mean_score_range",
            "pairwise_ranking_flip_rate",
            "tie_change_rate",
        )
    )
    checks["static_order_metrics_zero"] = zero
    if not zero:
        failures.append("static_order_metrics_zero: static baseline is not exactly zero")
    repeat_expected = len(config["repeat_control"]["blocks"]) * len(
        config["repeat_control"]["methods"]
    )
    repeat_lines = max(0, len(repeat_path.read_text(encoding="utf-8").splitlines()) - 1)
    checks["repeat_controls_complete"] = repeat_lines == repeat_expected
    if repeat_lines != repeat_expected:
        failures.append(
            "repeat_controls_complete: "
            f"expected {repeat_expected} repeat rows, found {repeat_lines}"
        )


def _check_prompt_freeze(
    config: dict[str, Any], manifest: dict[str, Any], checks: dict[str, bool], failures: list[str]
) -> None:
    frozen = manifest.get("prompts", {})
    current = {name: load_template(config, name).sha256 for name in config["prompts"]}
    match = all(frozen.get(name, {}).get("sha256") == digest for name, digest in current.items())
    checks["prompts_frozen"] = match
    if not match:
        failures.append("prompts_frozen: configured prompt content changed after preparation")


def _check_notebooks(root: Path, checks: dict[str, bool], failures: list[str]) -> None:
    expected = {
        "02_rubric_generation.ipynb": {"EXECUTE_MODEL_CALLS": False, "MAX_CALLS": 0},
        "03_judging.ipynb": {"EXECUTE_MODEL_CALLS": False, "MAX_CALLS": 0},
        "04_analysis.ipynb": {"ALLOW_PARTIAL_RESULTS": False},
    }
    safe = True
    for name, values in expected.items():
        notebook = json.loads((root / "notebooks" / name).read_text(encoding="utf-8"))
        sources = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        assignments = _literal_assignments(sources)
        safe &= all(assignments.get(key) == value for key, value in values.items())
    checks["notebooks_safe"] = safe
    if not safe:
        failures.append("notebooks_safe: an execution or partial-results default is unsafe")


def _literal_assignments(source: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
    return result


def _finish(
    run_dir: Path, checks: dict[str, bool], failures: list[str], write_report: bool
) -> dict[str, Any]:
    verdict = "READY_FOR_MAIN" if not failures else "NOT_READY_FOR_MAIN"
    payload = {"verdict": verdict, "checks": checks, "failures": failures}
    if write_report and run_dir.exists():
        write_json(run_dir / "readiness_report.json", payload)
    return payload
