from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable
from typing import Any

TOKEN_ESTIMATE_METHOD = "conservative_chars_div_3_v1"
TRUNCATION_MARKER = "\n[... deterministically truncated ...]\n"


class ContextBudgetError(ValueError):
    """Raised before inference when a rendered prompt exceeds its declared budget."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    """Normalize line endings and trailing horizontal whitespace exactly once."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip(" \t") for line in normalized.split("\n")).strip()


def estimate_tokens(text: str, tokenizer: Callable[[str], int] | None = None) -> int:
    """Return an exact local tokenizer count, or a deliberately conservative estimate."""
    if tokenizer is not None:
        value = tokenizer(text)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("tokenizer must return a non-negative integer")
        return value
    return math.ceil(len(text) / 3)


def truncate_text(text: str, max_characters: int | None) -> tuple[str, bool]:
    if max_characters is None:
        return text, False
    if max_characters < len(TRUNCATION_MARKER) + 2:
        raise ValueError("text limit is too small for deterministic truncation marker")
    if len(text) <= max_characters:
        return text, False
    available = max_characters - len(TRUNCATION_MARKER)
    head = (available + 1) // 2
    tail = available // 2
    return text[:head] + TRUNCATION_MARKER + text[-tail:], True


def preprocess_example(
    row: dict[str, Any],
    *,
    max_prompt_characters: int | None = None,
    max_response_characters: int | None = None,
    tokenizer: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    """Create the single processed representation used by every experimental order."""
    original_prompt = row["prompt"]
    original_response = row["response"]
    prompt = normalize_text(original_prompt)
    response = normalize_text(original_response)
    used_prompt, prompt_truncated = truncate_text(prompt, max_prompt_characters)
    used_response, response_truncated = truncate_text(response, max_response_characters)
    result = dict(row)
    result.update(
        {
            "prompt": used_prompt,
            "response": used_response,
            "text_processing": {
                "method": TOKEN_ESTIMATE_METHOD if tokenizer is None else "local_exact_tokenizer",
                "original_prompt_sha256": sha256_text(original_prompt),
                "original_response_sha256": sha256_text(original_response),
                "normalized_prompt_sha256": sha256_text(prompt),
                "normalized_response_sha256": sha256_text(response),
                "used_prompt_sha256": sha256_text(used_prompt),
                "used_response_sha256": sha256_text(used_response),
                "estimated_prompt_tokens": estimate_tokens(used_prompt, tokenizer),
                "estimated_response_tokens": estimate_tokens(used_response, tokenizer),
                "prompt_truncated": prompt_truncated,
                "response_truncated": response_truncated,
            },
        }
    )
    return result


def preprocess_examples(
    rows: list[dict[str, Any]], context_config: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        preprocess_example(
            row,
            max_prompt_characters=context_config.get("max_prompt_characters"),
            max_response_characters=context_config.get("max_response_characters"),
        )
        for row in rows
    ]


def preflight_context(
    prompt: str,
    *,
    context_length: int,
    max_output_tokens: int,
    safety_margin_tokens: int,
    method: str,
    block_id: int | None,
    order: str | None,
    reject: bool = True,
) -> dict[str, Any]:
    estimated = estimate_tokens(prompt)
    input_limit = context_length - max_output_tokens - safety_margin_tokens
    headroom = input_limit - estimated
    result = {
        "token_estimate_method": TOKEN_ESTIMATE_METHOD,
        "estimated_input_tokens": estimated,
        "context_length": context_length,
        "max_output_tokens": max_output_tokens,
        "safety_margin_tokens": safety_margin_tokens,
        "input_token_limit": input_limit,
        "estimated_headroom_tokens": headroom,
        "allowed": headroom >= 0,
        "method": method,
        "block_id": block_id,
        "order": order,
    }
    if reject and not result["allowed"]:
        raise ContextBudgetError(
            "context preflight rejected updater call: "
            f"method={method}, block={block_id}, order={order}, "
            f"estimated_input_tokens={estimated}, input_limit={input_limit}"
        )
    return result


def postflight_context(
    prompt_eval_count: int | None,
    *,
    context_length: int,
    minimum_headroom_tokens: int,
) -> dict[str, Any]:
    if prompt_eval_count is None:
        return {
            "prompt_eval_count": None,
            "observed_headroom_tokens": None,
            "context_saturation": True,
            "backend_prompt_truncated": None,
            "critical": True,
            "reason": "missing_prompt_eval_count",
        }
    headroom = context_length - prompt_eval_count
    exact_limit = prompt_eval_count >= context_length
    saturation = exact_limit or headroom < minimum_headroom_tokens
    return {
        "prompt_eval_count": prompt_eval_count,
        "observed_headroom_tokens": headroom,
        "context_saturation": saturation,
        "backend_prompt_truncated": exact_limit,
        "critical": exact_limit,
        "reason": (
            "context_limit_reached"
            if exact_limit
            else "low_context_headroom"
            if saturation
            else None
        ),
    }


def truncated_example_ids(blocks: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            row["example_id"]
            for block in blocks
            for row in block["examples"]
            if any(
                row.get("text_processing", {}).get(key, False)
                for key in ("prompt_truncated", "response_truncated")
            )
        }
    )


def unusual_field_tokens(text: str) -> set[str]:
    return set(re.findall(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b", text))
