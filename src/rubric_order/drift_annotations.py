from __future__ import annotations

import hashlib
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .run import file_sha256, read_json, read_jsonl, write_json, write_jsonl

ANNOTATION_FIELDS = {
    "blind_id",
    "domain_general",
    "task_specificity",
    "copies_example_terms",
    "duplicate_or_near_duplicate",
    "short_rationale",
}


def plan_blinded_drift_export(source_run: str | Path) -> dict[str, Any]:
    source = Path(source_run).resolve()
    index = read_jsonl(source / "artifacts" / "rubrics.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = {}
    rubric_content: dict[str, dict[str, Any]] = {}
    for row in index:
        digest = row["canonical_rubric_sha256"]
        grouped.setdefault(digest, []).append(row)
        rubric_content.setdefault(digest, read_json(source / row["rubric_path"]))
    records = []
    mapping = []
    for digest in sorted(grouped):
        blind_id = "DRIFT-" + hashlib.sha256(f"drift-v1:{digest}".encode()).hexdigest()[:12]
        rubric = rubric_content[digest]
        records.append(
            {
                "blind_id": blind_id,
                "adaptive_criteria": rubric.get("criteria", [])[3:],
                "annotation_fields": {
                    "domain_general": None,
                    "task_specificity": None,
                    "copies_example_terms": None,
                    "duplicate_or_near_duplicate": None,
                    "short_rationale": "",
                },
            }
        )
        mapping.append(
            {
                "blind_id": blind_id,
                "canonical_rubric_sha256": digest,
                "artifacts": [
                    {
                        "method": row["method"],
                        "block_id": row["block_id"],
                        "order": row["order"],
                        "replicate": row["replicate"],
                        "rubric_path": row["rubric_path"],
                    }
                    for row in grouped[digest]
                ],
            }
        )
    return {
        "source_run": str(source),
        "artifact_count": len(index),
        "blinded_item_count": len(records),
        "records": records,
        "mapping_records": mapping,
        "blinding_unit": "unique canonical rubric",
        "replicates_are_not_independent": True,
    }


def export_blinded_drift(source_run: str | Path, output_dir: str | Path) -> dict[str, Any]:
    plan = plan_blinded_drift_export(source_run)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"drift export directory already exists: {output}")
    output.mkdir(parents=True)
    blinded_path = output / "blinded_rubrics.jsonl"
    mapping_path = output / "blind_mapping.json"
    write_jsonl(blinded_path, plan["records"])
    mapping_payload = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run": plan["source_run"],
        "blinded_export_sha256": file_sha256(blinded_path),
        "records": plan["mapping_records"],
    }
    write_json(mapping_path, mapping_payload)
    return {
        "blinded_export": str(blinded_path),
        "mapping": str(mapping_path),
        "blinded_export_sha256": mapping_payload["blinded_export_sha256"],
        "mapping_sha256": file_sha256(mapping_path),
        "items": plan["blinded_item_count"],
    }


def validate_annotation_import(
    annotations_path: str | Path,
    mapping_path: str | Path,
    *,
    expected_mapping_sha256: str | None = None,
) -> dict[str, Any]:
    annotations_file = Path(annotations_path)
    mapping_file = Path(mapping_path)
    mapping = read_json(mapping_file)
    mapping_hash = file_sha256(mapping_file)
    if expected_mapping_sha256 is not None and mapping_hash != expected_mapping_sha256:
        raise ValueError("blind mapping hash differs from the expected frozen hash")
    blinded_path = mapping_file.parent / "blinded_rubrics.jsonl"
    if not blinded_path.exists():
        raise FileNotFoundError("blinded export referenced by mapping is missing")
    if file_sha256(blinded_path) != mapping.get("blinded_export_sha256"):
        raise ValueError("blinded export hash differs from mapping")
    expected = {row["blind_id"] for row in mapping.get("records", [])}
    annotations = read_jsonl(annotations_file)
    seen: set[str] = set()
    for index, row in enumerate(annotations):
        if set(row) != ANNOTATION_FIELDS:
            raise ValueError(
                f"annotation row {index} must contain exactly {sorted(ANNOTATION_FIELDS)}"
            )
        blind_id = row["blind_id"]
        if blind_id not in expected:
            raise ValueError(f"annotation contains unknown blind_id: {blind_id}")
        if blind_id in seen:
            raise ValueError(f"annotation contains duplicate blind_id: {blind_id}")
        seen.add(blind_id)
        if row["domain_general"] not in {"yes", "no"}:
            raise ValueError(f"invalid domain_general label for {blind_id}")
        if row["task_specificity"] not in {0, 1, 2}:
            raise ValueError(f"invalid task_specificity label for {blind_id}")
        for field in ("copies_example_terms", "duplicate_or_near_duplicate"):
            if row[field] not in {"yes", "no"}:
                raise ValueError(f"invalid {field} label for {blind_id}")
        rationale = row["short_rationale"]
        if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 300:
            raise ValueError(f"invalid short_rationale for {blind_id}")
    missing = expected - seen
    if missing:
        raise ValueError(f"annotations are incomplete; missing {len(missing)} blind IDs")
    return {
        "status": "valid",
        "annotations": annotations,
        "annotation_count": len(annotations),
        "annotations_sha256": file_sha256(annotations_file),
        "mapping_sha256": mapping_hash,
        "blinded_export_sha256": mapping["blinded_export_sha256"],
    }


def annotation_agreement(
    annotation_paths: list[str | Path],
    mapping_path: str | Path,
    adjudicated_path: str | Path | None = None,
) -> dict[str, Any]:
    if not annotation_paths:
        raise ValueError("at least one annotation file is required")
    validated = [validate_annotation_import(path, mapping_path) for path in annotation_paths]
    warning = None
    agreement: dict[str, float | None] = {}
    disagreements: list[dict[str, Any]] = []
    if len(validated) == 1:
        warning = "Only one annotator is available; inter-rater reliability cannot be estimated."
    elif len(validated) == 2:
        left = {row["blind_id"]: row for row in validated[0]["annotations"]}
        right = {row["blind_id"]: row for row in validated[1]["annotations"]}
        for field in (
            "domain_general",
            "task_specificity",
            "copies_example_terms",
            "duplicate_or_near_duplicate",
        ):
            agreement[field] = cohens_kappa(
                [left[key][field] for key in sorted(left)],
                [right[key][field] for key in sorted(left)],
            )
        disagreements = [
            {
                "blind_id": key,
                "annotator_1": {
                    field: left[key][field] for field in ANNOTATION_FIELDS - {"blind_id"}
                },
                "annotator_2": {
                    field: right[key][field] for field in ANNOTATION_FIELDS - {"blind_id"}
                },
            }
            for key in sorted(left)
            if any(
                left[key][field] != right[key][field]
                for field in ANNOTATION_FIELDS - {"blind_id", "short_rationale"}
            )
        ]
    else:
        raise ValueError("this workflow supports one or two primary annotators")
    adjudicated = (
        validate_annotation_import(adjudicated_path, mapping_path)
        if adjudicated_path is not None
        else None
    )
    return {
        "annotator_count": len(validated),
        "methodological_warning": warning,
        "cohens_kappa": agreement,
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
        "adjudicated": adjudicated is not None,
        "adjudicated_sha256": adjudicated["annotations_sha256"] if adjudicated else None,
    }


def cohens_kappa(left: list[Any], right: list[Any]) -> float | None:
    if len(left) != len(right) or not left:
        raise ValueError("Cohen's kappa inputs must have equal non-zero length")
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    labels = set(left_counts) | set(right_counts)
    expected = sum(
        left_counts[label] / len(left) * right_counts[label] / len(right) for label in labels
    )
    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def freeze_annotations(
    annotation_paths: list[str | Path],
    mapping_path: str | Path,
    destination: str | Path,
    *,
    adjudicated_path: str | Path | None = None,
) -> dict[str, Any]:
    report = annotation_agreement(annotation_paths, mapping_path, adjudicated_path)
    payload = {
        "schema_version": "1.0",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "mapping_path": str(Path(mapping_path).resolve()),
        "mapping_sha256": file_sha256(mapping_path),
        "annotation_files": [
            {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}
            for path in annotation_paths
        ],
        "adjudicated": (
            {
                "path": str(Path(adjudicated_path).resolve()),
                "sha256": file_sha256(adjudicated_path),
            }
            if adjudicated_path is not None
            else None
        ),
        "agreement": report,
        "automated_flags_are_ground_truth": False,
    }
    write_json(destination, payload)
    return payload


def annotation_freeze_gate(
    freeze_path: str | Path,
    *,
    judging_started_at: str | None = None,
    minimum_annotators: int = 1,
) -> dict[str, Any]:
    path = Path(freeze_path)
    if not path.exists():
        return {"ready": False, "failures": ["annotation freeze artifact is missing"]}
    value = read_json(path)
    failures = []
    mapping_path = Path(value.get("mapping_path", ""))
    if not mapping_path.is_file() or file_sha256(mapping_path) != value.get("mapping_sha256"):
        failures.append("frozen blind mapping hash mismatch")
    for row in value.get("annotation_files", []):
        annotation_path = Path(row["path"])
        if not annotation_path.exists() or file_sha256(annotation_path) != row["sha256"]:
            failures.append(f"frozen annotation hash mismatch: {annotation_path}")
    annotators = len(value.get("annotation_files", []))
    if annotators < minimum_annotators:
        failures.append(f"requires at least {minimum_annotators} annotator(s)")
    frozen_at = datetime.fromisoformat(value["frozen_at"])
    adjudicated = value.get("adjudicated")
    if adjudicated is not None:
        adjudicated_path = Path(adjudicated["path"])
        if not adjudicated_path.exists() or file_sha256(adjudicated_path) != adjudicated["sha256"]:
            failures.append("frozen adjudicated annotation hash mismatch")
    if judging_started_at is not None:
        judging_time = datetime.fromisoformat(judging_started_at)
        if frozen_at > judging_time:
            failures.append("annotations were frozen after judging started")
    return {
        "ready": not failures,
        "frozen_at": value.get("frozen_at"),
        "freeze_sha256": file_sha256(path),
        "annotator_count": annotators,
        "methodological_warning": value.get("agreement", {}).get("methodological_warning"),
        "failures": failures,
    }


def lexical_drift_flags(
    rubric: dict[str, Any], calibration_examples: list[dict[str, Any]]
) -> dict[str, Any]:
    """Return inspectable lexical heuristics, never human labels or semantic truth."""
    source = " ".join(
        f"{row.get('prompt', '')} {row.get('response', '')}" for row in calibration_examples
    ).casefold()
    source_tokens = set(re.findall(r"(?u)\b\w{4,}\b", source))
    generic = {
        "response",
        "request",
        "information",
        "correctness",
        "clarity",
        "helpfulness",
        "relevant",
        "accurate",
        "complete",
    }
    flags = []
    for index, criterion in enumerate(rubric.get("criteria", [])[3:], start=3):
        text = " ".join(
            [str(criterion.get("name", "")), str(criterion.get("description", ""))]
            + [str(value) for value in criterion.get("positive_signals", [])]
            + [str(value) for value in criterion.get("negative_signals", [])]
        )
        tokens = set(re.findall(r"(?u)\b\w{4,}\b", text.casefold())) - generic
        overlap = sorted(tokens & source_tokens)
        if overlap:
            flags.append(
                {
                    "criterion_index": index,
                    "code": "lexical_overlap_with_calibration_examples",
                    "overlap_tokens": overlap,
                }
            )
        lowered = text.casefold()
        workflow_terms = sorted(
            term
            for term in ("prior step", "previous step", "next step", "extracted information")
            if term in lowered
        )
        if workflow_terms:
            flags.append(
                {
                    "criterion_index": index,
                    "code": "workflow_or_extraction_phrase",
                    "phrases": workflow_terms,
                }
            )
    return {
        "heuristic_only": True,
        "semantic_validator": False,
        "ground_truth": False,
        "human_annotation_required": True,
        "flags": flags,
    }
