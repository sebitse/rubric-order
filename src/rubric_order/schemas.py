from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from .context import sha256_text, unusual_field_tokens

RUBRIC_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["schema_version", "title", "criteria", "score_anchors"],
    "additionalProperties": False,
    "properties": {
        "schema_version": {"type": "string", "const": "1.0"},
        "title": {"type": "string", "minLength": 1, "maxLength": 100},
        "criteria": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {
                "type": "object",
                "required": [
                    "name",
                    "description",
                    "positive_signals",
                    "negative_signals",
                ],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "minLength": 1, "maxLength": 80},
                    "description": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 240,
                    },
                    "positive_signals": {
                        "type": "array",
                        "maxItems": 6,
                        "items": {"type": "string", "minLength": 1, "maxLength": 160},
                    },
                    "negative_signals": {
                        "type": "array",
                        "maxItems": 6,
                        "items": {"type": "string", "minLength": 1, "maxLength": 160},
                    },
                },
            },
        },
        "score_anchors": {
            "type": "object",
            "required": ["0", "1", "2", "3", "4"],
            "additionalProperties": False,
            "properties": {
                str(score): {"type": "string", "minLength": 1, "maxLength": 300}
                for score in range(5)
            },
        },
    },
}

JUDGE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["score"],
    "additionalProperties": False,
    "properties": {"score": {"type": "integer", "minimum": 0, "maximum": 4}},
}

RUBRIC_V2_JSON_SCHEMA: dict[str, Any] = deepcopy(RUBRIC_JSON_SCHEMA)
RUBRIC_V2_JSON_SCHEMA["properties"]["schema_version"]["const"] = "2.0"
RUBRIC_V2_JSON_SCHEMA["properties"]["criteria"]["minItems"] = 3


def parse_json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"response is not valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("response must be a JSON object")
    return value


def validate_rubric(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("rubric must be an object")
    required = {"schema_version", "title", "criteria", "score_anchors"}
    missing = required - value.keys()
    if missing:
        raise ValueError(f"rubric is missing fields: {sorted(missing)}")
    extra = value.keys() - required
    if extra:
        raise ValueError(f"rubric has unknown fields: {sorted(extra)}")
    if value["schema_version"] != "1.0":
        raise ValueError("rubric schema_version must be '1.0'")
    if not _short_string(value["title"], 100):
        raise ValueError("rubric title must be a non-empty string of at most 100 characters")
    criteria = value["criteria"]
    if not isinstance(criteria, list) or not 1 <= len(criteria) <= 5:
        raise ValueError("rubric criteria must contain between 1 and 5 entries")
    clean_criteria: list[dict[str, Any]] = []
    criterion_fields = {"name", "description", "positive_signals", "negative_signals"}
    for index, criterion in enumerate(criteria):
        if not isinstance(criterion, dict) or set(criterion) != criterion_fields:
            raise ValueError(f"criterion {index} must contain exactly {sorted(criterion_fields)}")
        if not _short_string(criterion["name"], 80):
            raise ValueError(f"criterion {index} name is invalid or too long")
        if not _short_string(criterion["description"], 240):
            raise ValueError(f"criterion {index} description is invalid or too long")
        for field in ("positive_signals", "negative_signals"):
            signals = criterion[field]
            if not isinstance(signals, list) or len(signals) > 6:
                raise ValueError(f"criterion {index} {field} must be a list of at most 6 strings")
            if any(not _short_string(signal, 160) for signal in signals):
                raise ValueError(f"criterion {index} {field} contains an invalid or long signal")
        clean_criteria.append(dict(criterion))
    anchors = value["score_anchors"]
    expected_anchors = {str(score) for score in range(5)}
    if not isinstance(anchors, dict) or set(anchors) != expected_anchors:
        raise ValueError("score_anchors must contain exactly keys 0, 1, 2, 3, and 4")
    if any(not _short_string(anchors[key], 300) for key in expected_anchors):
        raise ValueError("all score anchors must be non-empty strings of at most 300 characters")
    return {
        "schema_version": "1.0",
        "title": value["title"],
        "criteria": clean_criteria,
        "score_anchors": {str(score): anchors[str(score)] for score in range(5)},
    }


def parse_rubric(raw: str) -> dict[str, Any]:
    return validate_rubric(parse_json_object(raw))


def validate_rubric_v2(value: Any, initial_rubric: dict[str, Any]) -> dict[str, Any]:
    """Validate the lexical v2 contract against the immutable initial rubric."""
    clean = _validate_rubric_shape(value, schema_version="2.0", minimum_criteria=3)
    initial = _validate_rubric_shape(initial_rubric, schema_version="2.0", minimum_criteria=3)
    if clean["title"] != initial["title"]:
        raise ValueError("rubric title differs from the immutable initial title")
    if clean["criteria"][:3] != initial["criteria"][:3]:
        raise ValueError("the three core criteria must remain unchanged and in fixed order")
    if clean["score_anchors"] != initial["score_anchors"]:
        raise ValueError("score anchors must remain unchanged")
    names = [_normalized_name(item["name"]) for item in clean["criteria"]]
    if len(names) != len(set(names)):
        raise ValueError("criterion names must be unique after case-insensitive normalization")
    descriptions = [_normalized_text(item["description"]) for item in clean["criteria"]]
    if len(descriptions) != len(set(descriptions)):
        raise ValueError("criterion descriptions must be unique")
    for field in ("positive_signals", "negative_signals"):
        signals = [
            _normalized_text(signal)
            for criterion in clean["criteria"]
            for signal in criterion[field]
        ]
        if len(signals) != len(set(signals)):
            raise ValueError(f"exact duplicate {field} are not permitted")
    return canonicalize_rubric_v2(clean, initial)


def parse_rubric_v2(raw: str, initial_rubric: dict[str, Any]) -> dict[str, Any]:
    return validate_rubric_v2(parse_json_object(raw), initial_rubric)


def canonicalize_rubric_v2(value: dict[str, Any], initial_rubric: dict[str, Any]) -> dict[str, Any]:
    core = deepcopy(initial_rubric["criteria"][:3])
    adaptive = sorted(
        (deepcopy(item) for item in value["criteria"][3:]),
        key=lambda item: (_normalized_name(item["name"]), item["name"]),
    )
    return {
        "schema_version": "2.0",
        "title": initial_rubric["title"],
        "criteria": core + adaptive,
        "score_anchors": {
            str(score): initial_rubric["score_anchors"][str(score)] for score in range(5)
        },
    }


def rubric_hashes(raw_response: str, canonical: dict[str, Any]) -> dict[str, str]:
    canonical_text = json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return {
        "raw_response_sha256": sha256_text(raw_response),
        "canonical_rubric_sha256": sha256_text(canonical_text),
    }


def drift_audit(
    rubric: dict[str, Any],
    initial_rubric: dict[str, Any],
    examples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Flag lexical drift for human inspection; this is not a semantic verdict."""
    flags: list[dict[str, Any]] = []
    criteria = rubric.get("criteria") if isinstance(rubric, dict) else None
    if not isinstance(criteria, list) or criteria[:3] != initial_rubric["criteria"][:3]:
        flags.append({"code": "core_changed_or_missing"})
    if rubric.get("score_anchors") != initial_rubric["score_anchors"]:
        flags.append({"code": "anchors_changed"})
    adaptive = criteria[3:] if isinstance(criteria, list) else []
    example_tokens = [
        _lexical_tokens(f"{row.get('prompt', '')} {row.get('response', '')}")
        for row in (examples or [])
    ]
    for index, criterion in enumerate(adaptive, start=3):
        if not isinstance(criterion, dict):
            continue
        name = str(criterion.get("name", ""))
        description = str(criterion.get("description", ""))
        combined = f"{name} {description}"
        if unusual_field_tokens(combined):
            flags.append(
                {
                    "code": "field_name_or_snake_case",
                    "criterion": index,
                    "tokens": sorted(unusual_field_tokens(combined)),
                }
            )
        criterion_tokens = _lexical_tokens(combined)
        for example_index, tokens in enumerate(example_tokens):
            overlap = len(criterion_tokens & tokens) / max(1, len(criterion_tokens))
            if len(criterion_tokens) >= 3 and overlap >= 0.75:
                flags.append(
                    {
                        "code": "high_single_example_overlap",
                        "criterion": index,
                        "example_index": example_index,
                        "overlap": round(overlap, 4),
                    }
                )
            if _normalized_name(name) and _normalized_name(name) in _normalized_text(
                f"{(examples or [])[example_index].get('prompt', '')} "
                f"{(examples or [])[example_index].get('response', '')}"
            ):
                flags.append(
                    {
                        "code": "literal_name_from_example",
                        "criterion": index,
                        "example_index": example_index,
                    }
                )
        if len(name) > 60 or len(description) > 200:
            flags.append({"code": "excessive_text_length", "criterion": index})
    if len(adaptive) == 2:
        left = _lexical_tokens(
            f"{adaptive[0].get('name', '')} {adaptive[0].get('description', '')}"
        )
        right = _lexical_tokens(
            f"{adaptive[1].get('name', '')} {adaptive[1].get('description', '')}"
        )
        similarity = len(left & right) / max(1, len(left | right))
        if similarity >= 0.6:
            flags.append(
                {"code": "adaptive_criteria_high_similarity", "jaccard": round(similarity, 4)}
            )
    return {
        "semantic_verdict": None,
        "requires_human_inspection": bool(flags),
        "flags": flags,
    }


def validate_judge_output(value: Any) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != {"score"}:
        raise ValueError("judge output must contain exactly the score field")
    score = value["score"]
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 4:
        raise ValueError("judge score must be an integer from 0 to 4")
    return {"score": score}


def parse_judge_output(raw: str) -> dict[str, int]:
    return validate_judge_output(parse_json_object(raw))


def _short_string(value: Any, limit: int) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= limit


def _validate_rubric_shape(
    value: Any, *, schema_version: str, minimum_criteria: int
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("rubric must be an object")
    required = {"schema_version", "title", "criteria", "score_anchors"}
    if set(value) != required:
        raise ValueError(f"rubric must contain exactly {sorted(required)}; got {sorted(value)}")
    if value["schema_version"] != schema_version:
        raise ValueError(f"rubric schema_version must be {schema_version!r}")
    if not _short_string(value["title"], 100):
        raise ValueError("rubric title must be a non-empty string of at most 100 characters")
    criteria = value["criteria"]
    if not isinstance(criteria, list) or not minimum_criteria <= len(criteria) <= 5:
        raise ValueError(f"rubric criteria must contain between {minimum_criteria} and 5 entries")
    criterion_fields = {"name", "description", "positive_signals", "negative_signals"}
    clean_criteria: list[dict[str, Any]] = []
    for index, criterion in enumerate(criteria):
        if not isinstance(criterion, dict) or set(criterion) != criterion_fields:
            raise ValueError(f"criterion {index} must contain exactly {sorted(criterion_fields)}")
        if not _short_string(criterion["name"], 80):
            raise ValueError(f"criterion {index} name is invalid or too long")
        if not _short_string(criterion["description"], 240):
            raise ValueError(f"criterion {index} description is invalid or too long")
        clean = dict(criterion)
        for field in ("positive_signals", "negative_signals"):
            signals = criterion[field]
            if not isinstance(signals, list) or len(signals) > 6:
                raise ValueError(f"criterion {index} {field} must be a list of at most 6 strings")
            if any(not _short_string(signal, 160) for signal in signals):
                raise ValueError(f"criterion {index} {field} contains an invalid or long signal")
            clean[field] = list(signals)
        clean_criteria.append(clean)
    anchors = value["score_anchors"]
    expected_anchors = {str(score) for score in range(5)}
    if not isinstance(anchors, dict) or set(anchors) != expected_anchors:
        raise ValueError("score_anchors must contain exactly keys 0, 1, 2, 3, and 4")
    if any(not _short_string(anchors[key], 300) for key in expected_anchors):
        raise ValueError("all score anchors must be non-empty strings of at most 300 characters")
    return {
        "schema_version": schema_version,
        "title": value["title"],
        "criteria": clean_criteria,
        "score_anchors": {str(score): anchors[str(score)] for score in range(5)},
    }


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _lexical_tokens(value: str) -> set[str]:
    return set(re.findall(r"(?u)\b\w\w+\b", value.casefold()))
