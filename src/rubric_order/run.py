from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import public_config, resolve_project_path
from .context import preprocess_examples, sha256_text, truncated_example_ids
from .data import dataset_fingerprint, load_dataset
from .exclusions import apply_exclusions, assert_disjoint, load_exclusions
from .sampling import build_calibration_blocks, select_partitions
from .schemas import JUDGE_JSON_SCHEMA, RUBRIC_V2_JSON_SCHEMA


def run_directory(config: dict[str, Any], run_id: str) -> Path:
    root = resolve_project_path(config, config["output_root"])
    return root / run_id


def prepare_run(config: dict[str, Any], run_id: str, resume: bool = False) -> dict[str, Any]:
    if config.get("schema_version") == "3.0" and config.get("paired_source"):
        raise ValueError(
            "paired v3 runs may not resample; use the paired-source preparation workflow"
        )
    run_dir = run_directory(config, run_id)
    manifest_path = run_dir / "manifest.json"
    config_hash = _object_hash(public_config(config))
    if run_dir.exists():
        if not resume:
            raise FileExistsError(
                f"run directory already exists: {run_dir}; pass --resume to reuse it"
            )
        if not manifest_path.exists():
            raise ValueError(f"existing run directory has no manifest: {run_dir}")
        manifest = read_json(manifest_path)
        if manifest.get("config_hash") != config_hash:
            raise ValueError("existing run uses a different resolved configuration")
        return manifest

    train, validation = load_dataset(config)
    original_fingerprint = dataset_fingerprint(train, validation)
    excluded_examples, excluded_prompts, exclusion_report = load_exclusions(config)
    train, train_exclusion = apply_exclusions(train, excluded_examples, excluded_prompts)
    if validation is not None:
        validation, validation_exclusion = apply_exclusions(
            validation, excluded_examples, excluded_prompts
        )
    else:
        validation_exclusion = None
    sampling = config["sampling"]
    calibration_pool, heldout, partition_metadata = select_partitions(
        train,
        validation,
        heldout_prompts=sampling["heldout_prompts"],
        responses_per_prompt=sampling["responses_per_prompt"],
        seed=config["seed"],
    )
    if config.get("schema_version") in {"2.0", "3.0"}:
        calibration_pool = preprocess_examples(calibration_pool, config["context"])
        heldout = preprocess_examples(heldout, config["context"])
    blocks, block_metadata = build_calibration_blocks(
        calibration_pool,
        block_count=sampling["calibration_blocks"],
        examples_per_score=sampling["examples_per_score"],
        seed=config["seed"],
    )
    assert_disjoint(
        [row for block in blocks for row in block["examples"]] + heldout,
        excluded_examples,
        excluded_prompts,
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    prepared_dir = run_dir / "prepared"
    prepared_dir.mkdir()
    write_json(prepared_dir / "calibration_blocks.json", blocks)
    write_jsonl(prepared_dir / "heldout.jsonl", heldout)
    write_json(run_dir / "config.resolved.json", public_config(config))
    manifest = {
        "schema_version": config.get("schema_version", "1.0"),
        "prompt_version": config.get("prompt_version", "1.0"),
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash,
        "dataset_fingerprint": original_fingerprint,
        "post_exclusion_dataset_fingerprint": dataset_fingerprint(train, validation),
        "config_path": config["_config_path"],
        "runtime_overrides": config.get("_runtime_overrides", {}),
        "prepared": True,
        "partition": partition_metadata,
        "calibration": block_metadata,
        "block_count": len(blocks),
        "examples_per_block": len(blocks[0]["examples"]),
        "heldout_response_count": len(heldout),
        "model_calls_made": 0,
        "exclusions": {
            **exclusion_report,
            "train": train_exclusion,
            "validation": validation_exclusion,
            "all_selected_subsets_disjoint": True,
        },
    }
    if config.get("schema_version") in {"2.0", "3.0"}:
        manifest.update(_v2_provenance(config, blocks))
    write_json(manifest_path, manifest)
    return manifest


def load_prepared(
    config: dict[str, Any], run_id: str
) -> tuple[Path, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    run_dir = run_directory(config, run_id)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"run is not prepared: {run_dir}")
    manifest = read_json(manifest_path)
    config_hash = _object_hash(public_config(config))
    if manifest.get("config_hash") != config_hash:
        raise ValueError("run manifest does not match the resolved configuration")
    blocks = read_json(run_dir / "prepared" / "calibration_blocks.json")
    heldout = read_jsonl(run_dir / "prepared" / "heldout.jsonl")
    return run_dir, manifest, blocks, heldout


def write_dry_run(run_dir: Path, stage: str, details: dict[str, Any]) -> Path:
    dry_dir = run_dir / "dry_runs"
    dry_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = dry_dir / f"{stage}_{stamp}.json"
    payload = {
        "stage": stage,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inference_executed": False,
        **details,
    }
    write_json(path, payload)
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row is not an object: {path}")
                rows.append(value)
    return rows


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write(destination, content)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(destination, content)


def merge_index(path: Path, new_rows: list[dict[str, Any]], key_fields: tuple[str, ...]) -> None:
    existing = read_jsonl(path) if path.exists() else []
    indexed = {tuple(row.get(field) for field in key_fields): row for row in existing}
    for row in new_rows:
        indexed[tuple(row.get(field) for field in key_fields)] = row
    ordered = [
        indexed[key] for key in sorted(indexed, key=lambda item: tuple(str(v) for v in item))
    ]
    write_jsonl(path, ordered)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sync_manifest_model_calls(run_dir: Path) -> int:
    database = run_dir / "calls.sqlite3"
    manifest_path = run_dir / "manifest.json"
    if not database.exists() or not manifest_path.exists():
        return 0
    connection = sqlite3.connect(database)
    try:
        count = int(connection.execute("SELECT COUNT(*) FROM calls").fetchone()[0])
        columns = {row[1] for row in connection.execute("PRAGMA table_info(calls)")}
        context_summary = None
        if {"context_saturation", "observed_headroom_tokens"} <= columns:
            row = connection.execute(
                """
                SELECT SUM(CASE WHEN context_saturation = 1 THEN 1 ELSE 0 END),
                       MIN(observed_headroom_tokens), MAX(estimated_input_tokens)
                FROM calls WHERE status = 'success'
                """
            ).fetchone()
            context_summary = {
                "saturated_successful_calls": int(row[0] or 0),
                "minimum_observed_headroom_tokens": row[1],
                "maximum_estimated_input_tokens": row[2],
            }
    finally:
        connection.close()
    manifest = read_json(manifest_path)
    manifest["model_calls_made"] = count
    if context_summary is not None:
        manifest["call_context_summary"] = context_summary
    write_json(manifest_path, manifest)
    return count


def _object_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _v2_provenance(config: dict[str, Any], blocks: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_records: dict[str, dict[str, str]] = {}
    for name, configured in sorted(config["prompts"].items()):
        path = resolve_project_path(config, configured)
        content = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        prompt_records[name] = {"content": content, "sha256": sha256_text(content)}
    initial_path = resolve_project_path(config, config["rubric"]["initial"])
    initial = read_json(initial_path)
    schema_payload = {
        "rubric": RUBRIC_V2_JSON_SCHEMA,
        "judge": JUDGE_JSON_SCHEMA,
    }
    schema_text = json.dumps(
        schema_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    project_root = Path(config["_project_root"])
    return {
        "prompts": prompt_records,
        "json_schema": {"content": schema_payload, "sha256": sha256_text(schema_text)},
        "initial_rubric": {
            "content": initial,
            "sha256": _object_hash(initial),
        },
        "seeds": {
            "experiment": config["seed"],
            "model": config["models"]["seed"],
        },
        "model_provenance": {
            "updater": {"name": config["models"]["updater"], "digest": None, "quantization": None},
            "judge": {"name": config["models"]["judge"], "digest": None, "quantization": None},
            "ollama_version": None,
            "captured_without_model_contact": True,
        },
        "model_options": {
            "context_length": config["models"]["context_length"],
            "updater_max_tokens": config["models"]["updater_max_tokens"],
            "judge_max_tokens": config["models"]["judge_max_tokens"],
            "temperature": config["models"]["temperature"],
            "think": config["models"].get("think", False),
        },
        "environment": {
            "python": platform.python_version(),
            "packages": _package_versions(("pytest", "ruff", "nbformat", "matplotlib")),
        },
        "code": _code_provenance(project_root),
        "text_processing": {
            "truncated_example_ids": truncated_example_ids(blocks),
            "truncated_example_count": len(truncated_example_ids(blocks)),
        },
    }


def _package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _code_provenance(root: Path) -> dict[str, Any]:
    relevant: list[Path] = []
    for directory in ("src", "configs", "prompts", "rubrics"):
        candidate = root / directory
        if candidate.exists():
            relevant.extend(
                path
                for path in candidate.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            )
    inventory = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(relevant, key=lambda item: item.relative_to(root).as_posix())
    ]
    canonical = json.dumps(
        inventory, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    return {
        "git_commit": None,
        "git_dirty": None,
        "file_tree_sha256": digest,
        "content_inventory_sha256": digest,
        "content_inventory_algorithm": (
            "sha256(canonical-json([{path,sha256(file-bytes)}], sorted by POSIX path))"
        ),
        "content_inventory_file_count": len(inventory),
    }


def _atomic_write(destination: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
