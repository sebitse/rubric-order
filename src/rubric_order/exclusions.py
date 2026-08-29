from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .config import resolve_project_path


def load_exclusions(config: dict[str, Any]) -> tuple[set[str], set[str], dict[str, Any]]:
    """Load development exclusions without opening or modifying any run database."""
    example_ids: set[str] = set()
    prompt_ids: set[str] = set()
    source_rows: list[dict[str, Any]] = []
    example_sources: Counter[str] = Counter()
    prompt_sources: Counter[str] = Counter()
    for entry in config.get("exclude_manifests", []):
        spec = {"path": entry} if isinstance(entry, str) else dict(entry)
        path = resolve_project_path(config, spec["path"])
        required = bool(spec.get("required", True))
        if not path.exists():
            if required:
                raise FileNotFoundError(f"required exclusion manifest does not exist: {path}")
            source_rows.append({"path": str(path), "present": False, "required": False})
            continue
        found_examples, found_prompts = _ids_from_source(path)
        source_key = path.as_posix()
        example_overlap = len(example_ids & found_examples)
        prompt_overlap = len(prompt_ids & found_prompts)
        example_ids.update(found_examples)
        prompt_ids.update(found_prompts)
        for value in found_examples:
            example_sources[value] += 1
        for value in found_prompts:
            prompt_sources[value] += 1
        source_rows.append(
            {
                "path": str(path),
                "present": True,
                "required": required,
                "sha256": _file_hash(path),
                "example_ids": len(found_examples),
                "prompt_ids": len(found_prompts),
                "example_collisions_with_prior_sources": example_overlap,
                "prompt_collisions_with_prior_sources": prompt_overlap,
                "source": source_key,
            }
        )
    report = {
        "excluded_example_ids": len(example_ids),
        "excluded_prompt_ids": len(prompt_ids),
        "sources": source_rows,
        "example_ids_in_multiple_sources": sorted(
            value for value, count in example_sources.items() if count > 1
        ),
        "prompt_ids_in_multiple_sources": sorted(
            value for value, count in prompt_sources.items() if count > 1
        ),
    }
    return example_ids, prompt_ids, report


def apply_exclusions(
    rows: list[dict[str, Any]], example_ids: set[str], prompt_ids: set[str]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    kept = [
        row
        for row in rows
        if row["example_id"] not in example_ids and row["prompt_id"] not in prompt_ids
    ]
    removed = len(rows) - len(kept)
    return kept, {"input_rows": len(rows), "removed_rows": removed, "remaining_rows": len(kept)}


def assert_disjoint(
    selected_rows: list[dict[str, Any]], example_ids: set[str], prompt_ids: set[str]
) -> None:
    leaked_examples = {row["example_id"] for row in selected_rows} & example_ids
    leaked_prompts = {row["prompt_id"] for row in selected_rows} & prompt_ids
    if leaked_examples or leaked_prompts:
        raise AssertionError(
            "development exclusion failure: "
            f"example_ids={sorted(leaked_examples)}, prompt_ids={sorted(leaked_prompts)}"
        )


def _ids_from_source(path: Path) -> tuple[set[str], set[str]]:
    value = _read_json(path)
    example_ids, prompt_ids = _extract_ids(value)
    base = path.parent
    if path.name == "manifest.json":
        candidates = (
            base / "prepared" / "calibration_blocks.json",
            base / "prepared" / "heldout.jsonl",
            base / "subsets" / "judge_dev.jsonl",
            base / "subsets" / "judge_validation.jsonl",
        )
        for candidate in candidates:
            if not candidate.exists():
                continue
            child_examples, child_prompts = _extract_ids(_read_json(candidate))
            example_ids.update(child_examples)
            prompt_ids.update(child_prompts)
    return example_ids, prompt_ids


def _read_json(path: Path) -> Any:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_ids(value: Any) -> tuple[set[str], set[str]]:
    example_ids: set[str] = set()
    prompt_ids: set[str] = set()
    if isinstance(value, dict):
        if isinstance(value.get("example_id"), str):
            example_ids.add(value["example_id"])
        if isinstance(value.get("prompt_id"), str):
            prompt_ids.add(value["prompt_id"])
        for key in ("example_ids", "excluded_example_ids"):
            items = value.get(key, [])
            if isinstance(items, list):
                example_ids.update(item for item in items if isinstance(item, str))
        for key in ("prompt_ids", "excluded_prompt_ids"):
            items = value.get(key, [])
            if isinstance(items, list):
                prompt_ids.update(item for item in items if isinstance(item, str))
        for item in value.values():
            child_examples, child_prompts = _extract_ids(item)
            example_ids.update(child_examples)
            prompt_ids.update(child_prompts)
    elif isinstance(value, list):
        for item in value:
            child_examples, child_prompts = _extract_ids(item)
            example_ids.update(child_examples)
            prompt_ids.update(child_prompts)
    return example_ids, prompt_ids


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
