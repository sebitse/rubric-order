from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "schema_version": "1.0",
    "prompt_version": "1.0",
    "seed": 42,
    "dataset": {
        "path": None,
        "format": "auto",
        "columns": {"prompt": "prompt", "response": "response", "label": "helpfulness"},
        "train_split": "train",
        "validation_split": "validation",
    },
    "sampling": {
        "calibration_blocks": 2,
        "examples_per_score": 2,
        "heldout_prompts": 10,
        "responses_per_prompt": 2,
    },
    "orders": ["ascending", "descending", "random", "reverse_random"],
    "methods": ["static", "batch", "sequential"],
    "models": {
        "updater": "qwen3:14b",
        "judge": "llama3.1:8b",
        "base_url": "http://localhost:11434",
        "temperature": 0,
        "context_length": 8192,
        "updater_max_tokens": 650,
        "judge_max_tokens": 25,
        "seed": 42,
        "timeout_seconds": 300,
        "unload_after_stage": True,
        "think": False,
    },
    "rubric": {
        "initial": "rubrics/initial_helpfulness.json",
        "max_validation_retries": 2,
    },
    "prompts": {
        "sequential": "prompts/update_sequential.txt",
        "batch": "prompts/induce_batch.txt",
        "judge": "prompts/judge.txt",
    },
    "analysis": {"bootstrap_repetitions": 100, "confidence": 0.95, "make_plots": True},
    "context": {
        "safety_margin_tokens": 1024,
        "minimum_observed_headroom_tokens": 256,
        "max_prompt_characters": 1200,
        "max_response_characters": 2400,
    },
    "repeat_control": {
        "enabled": False,
        "blocks": [],
        "orders": ["random"],
        "methods": ["batch", "sequential"],
        "replicates": [1],
    },
    "output_root": "runs",
    "exclude_manifests": [],
    "approved_judge_artifact": None,
    "allow_partial_results": False,
}


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                f"{path} is YAML rather than JSON-compatible YAML; install PyYAML with "
                "`python -m pip install -e '.[yaml]'`"
            ) from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be an object: {path}")
    return value


def load_config(path: str | Path, data_path: str | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = _deep_merge(DEFAULTS, _read_yaml_or_json(config_path))
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(config_path.parent.parent.resolve())
    if data_path is not None:
        config["dataset"]["path"] = data_path
    _apply_approved_judge(config)
    _validate_config(config)
    return config


def apply_config_overrides(
    config: dict[str, Any], overrides: dict[str, Any] | None
) -> dict[str, Any]:
    """Return a validated copy with dotted-key runtime overrides applied."""
    result = copy.deepcopy(config)
    normalized = _json_compatible(overrides or {})
    for dotted_key, value in normalized.items():
        if dotted_key.startswith("_"):
            raise ValueError(f"runtime override cannot target internal key {dotted_key!r}")
        parts = dotted_key.split(".")
        target = result
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"runtime override has unknown path: {dotted_key}")
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"runtime override has unknown key: {dotted_key}")
        target[parts[-1]] = value
    result["_runtime_overrides"] = normalized
    _validate_config(result)
    return result


def _validate_config(config: dict[str, Any]) -> None:
    if config["schema_version"] not in {"1.0", "2.0", "3.0"}:
        raise ValueError("schema_version must be '1.0', '2.0', or '3.0'")
    if not isinstance(config["prompt_version"], str) or not config["prompt_version"]:
        raise ValueError("prompt_version must be a non-empty string")
    sampling = config["sampling"]
    for key in ("calibration_blocks", "examples_per_score", "heldout_prompts"):
        if not isinstance(sampling.get(key), int) or sampling[key] < 1:
            raise ValueError(f"sampling.{key} must be a positive integer")
    orders = config["orders"]
    expected = {"ascending", "descending", "random", "reverse_random"}
    if set(orders) != expected or len(orders) != 4:
        raise ValueError(f"orders must contain exactly {sorted(expected)}")
    methods = config["methods"]
    if not methods or not set(methods) <= {"static", "batch", "sequential"}:
        raise ValueError("methods may only contain static, batch, and sequential")
    if config["rubric"]["max_validation_retries"] > 2:
        raise ValueError("rubric.max_validation_retries may not exceed 2")
    execution_safety = config.get("execution_safety")
    if execution_safety is not None:
        if not isinstance(execution_safety, dict):
            raise ValueError("execution_safety must be an object")
        planned = execution_safety.get("planned_physical_calls")
        ceiling = execution_safety.get("hard_total_call_ceiling")
        if not isinstance(planned, int) or planned < 1:
            raise ValueError("execution_safety.planned_physical_calls must be positive")
        if not isinstance(ceiling, int) or ceiling < planned:
            raise ValueError("execution_safety.hard_total_call_ceiling must cover planned calls")
        stage_ceilings = execution_safety.get("stage_call_ceilings")
        if (
            not isinstance(stage_ceilings, dict)
            or set(stage_ceilings) != {"build-rubrics", "judge"}
            or not all(isinstance(value, int) and value > 0 for value in stage_ceilings.values())
            or sum(stage_ceilings.values()) != ceiling
        ):
            raise ValueError(
                "execution_safety.stage_call_ceilings must be positive updater/judge ceilings "
                "that sum to the hard total ceiling"
            )
        retries = execution_safety.get("max_retries_per_request")
        if retries != config["rubric"]["max_validation_retries"] or retries != 1:
            raise ValueError("final execution safety requires exactly one retry per request")
        error_rate = execution_safety.get("max_error_rate")
        if not isinstance(error_rate, (int, float)) or not 0 < error_rate <= 0.01:
            raise ValueError("execution_safety.max_error_rate must be in (0, 0.01]")
        if execution_safety.get("max_consecutive_same_error") != 3:
            raise ValueError("execution_safety.max_consecutive_same_error must be exactly 3")
    if config["analysis"]["bootstrap_repetitions"] < 0:
        raise ValueError("analysis.bootstrap_repetitions must be non-negative")
    context = config["context"]
    for key in ("safety_margin_tokens", "minimum_observed_headroom_tokens"):
        if not isinstance(context[key], int) or context[key] < 0:
            raise ValueError(f"context.{key} must be a non-negative integer")
    for key in ("max_prompt_characters", "max_response_characters"):
        if context[key] is not None and (not isinstance(context[key], int) or context[key] < 100):
            raise ValueError(f"context.{key} must be null or an integer of at least 100")
    if config["schema_version"] in {"2.0", "3.0"}:
        if config["models"]["context_length"] != 16384:
            raise ValueError("v2 updater context_length must be 16384")
        if config["models"]["updater_max_tokens"] != 650:
            raise ValueError("v2 updater_max_tokens must be 650")
        if config["models"]["judge_max_tokens"] != 25:
            raise ValueError("v2 judge_max_tokens must be 25")
        if config["models"]["temperature"] != 0:
            raise ValueError("v2 models.temperature must be 0")
        if config["models"].get("think") is not False:
            raise ValueError("v2 models.think must be false")
    if config["schema_version"] == "3.0":
        inventory = config.get("inventory_v3")
        if not isinstance(inventory, dict) or not isinstance(inventory.get("path"), str):
            raise ValueError("v3 configuration requires inventory_v3.path")
        if inventory.get("maximum_selected") != 2:
            raise ValueError("v3 inventory_v3.maximum_selected must be exactly 2")
        if config.get("updater_mode") not in {"inventory_constrained", "mixed"}:
            raise ValueError("v3 updater_mode must be 'inventory_constrained' or 'mixed'")


def resolve_project_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(config["_project_root"]) / path
    return path.resolve()


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _apply_approved_judge(config: dict[str, Any]) -> None:
    configured = config.get("approved_judge_artifact")
    if not configured:
        return
    path = Path(configured)
    if not path.is_absolute():
        path = Path(config["_project_root"]) / path
    if not path.exists():
        return
    artifact = _read_yaml_or_json(path)
    if artifact.get("verdict") != "JUDGE_APPROVED":
        return
    chosen = artifact.get("chosen", {})
    candidate = chosen.get("candidate")
    model = chosen.get("model")
    if candidate not in config["prompts"]:
        raise ValueError(f"approved judge candidate has no configured prompt: {candidate!r}")
    if not isinstance(model, str) or not model:
        raise ValueError("approved judge artifact has no model name")
    config["prompts"]["judge"] = config["prompts"][candidate]
    config["models"]["judge"] = model
    config["_approved_judge_selection"] = {
        "artifact": str(path.resolve()),
        "candidate": candidate,
        "model": model,
    }
