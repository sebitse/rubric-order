from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, TextIO


def load_dataset(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    dataset_config = config["dataset"]
    raw_path = dataset_config.get("path")
    if not raw_path:
        raise ValueError("dataset.path is unset; provide it in the config or with --data-path")
    path = Path(raw_path).expanduser()
    if not path.is_absolute() and config.get("_project_root"):
        path = Path(config["_project_root"]) / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"dataset path does not exist: {path}")
    raw_train, raw_validation = _load_raw_splits(path, dataset_config)
    columns = dataset_config["columns"]
    train = normalize_records(raw_train, columns, source="train")
    validation = (
        normalize_records(raw_validation, columns, source="validation")
        if raw_validation is not None
        else None
    )
    if not train:
        raise ValueError("training dataset is empty")
    if validation is not None and not validation:
        raise ValueError("validation split is present but empty")
    return train, validation


def normalize_records(
    records: Iterable[Mapping[str, Any]], columns: Mapping[str, str], source: str
) -> list[dict[str, Any]]:
    required_mappings = {"prompt", "response", "label"}
    missing_mappings = required_mappings - columns.keys()
    if missing_mappings:
        raise ValueError(f"dataset column mapping is missing: {sorted(missing_mappings)}")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        missing_columns = [columns[key] for key in required_mappings if columns[key] not in row]
        if missing_columns:
            raise ValueError(f"row {index} is missing columns: {sorted(missing_columns)}")
        prompt = row[columns["prompt"]]
        response = row[columns["response"]]
        label = _validate_label(row[columns["label"]], index)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"row {index} has a missing or non-text prompt")
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"row {index} has a missing or non-text response")
        normalized.append(
            {
                "example_id": example_id_for_record(source, index, prompt, response, label),
                "prompt_id": prompt_id_for_text(prompt),
                "prompt": prompt,
                "response": response,
                "human_score": label,
            }
        )
    return normalized


def prompt_id_for_text(prompt: str) -> str:
    """Return the one canonical prompt ID used by dataset normalization."""
    return _hash_text(prompt)


def example_id_for_record(
    source: str,
    index: int,
    prompt: str,
    response: str,
    label: int,
) -> str:
    """Return the one canonical example ID used by dataset normalization."""
    identity = json.dumps(
        [source, index, prompt, response, label], ensure_ascii=False, separators=(",", ":")
    )
    return _hash_text(identity)


def load_dataset_with_lineage(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Load raw and normalized rows together without changing either representation."""
    dataset_config = config["dataset"]
    raw_path = dataset_config.get("path")
    if not raw_path:
        raise ValueError("dataset.path is unset; provide it in the config or with --data-path")
    path = Path(raw_path).expanduser()
    if not path.is_absolute() and config.get("_project_root"):
        path = Path(config["_project_root"]) / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"dataset path does not exist: {path}")
    raw_train_iter, raw_validation_iter = _load_raw_splits(path, dataset_config)
    raw_train = list(raw_train_iter)
    raw_validation = list(raw_validation_iter) if raw_validation_iter is not None else None
    columns = dataset_config["columns"]
    train = _attach_lineage(raw_train, columns, "train", path)
    validation = (
        _attach_lineage(raw_validation, columns, "validation", path)
        if raw_validation is not None
        else None
    )
    return train, validation


def _attach_lineage(
    raw_rows: list[Mapping[str, Any]],
    columns: Mapping[str, str],
    source: str,
    dataset_path: Path,
) -> list[dict[str, Any]]:
    normalized = normalize_records(raw_rows, columns, source=source)
    return [
        {
            "source_dataset": str(dataset_path),
            "source_split": source,
            "source_row_index": index,
            "raw_record": dict(raw),
            "normalized_record": row,
        }
        for index, (raw, row) in enumerate(zip(raw_rows, normalized, strict=True))
    ]


def dataset_fingerprint(*splits: list[dict[str, Any]] | None) -> str:
    digest = hashlib.sha256()
    for split in splits:
        if split is None:
            digest.update(b"<none>")
            continue
        for row in split:
            digest.update(row["example_id"].encode())
            digest.update(b"\n")
    return digest.hexdigest()


def _validate_label(value: Any, index: int) -> int:
    if value is None or isinstance(value, bool):
        raise ValueError(f"row {index} has a missing or non-numeric helpfulness label")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"row {index} has a missing helpfulness label")
        try:
            numeric = float(stripped)
        except ValueError as exc:
            raise ValueError(f"row {index} has a non-numeric helpfulness label: {value!r}") from exc
    elif isinstance(value, (int, float)):
        numeric = float(value)
    else:
        raise ValueError(f"row {index} has a non-numeric helpfulness label: {value!r}")
    if not numeric.is_integer():
        raise ValueError(f"row {index} helpfulness label is not an integer: {value!r}")
    label = int(numeric)
    if not 0 <= label <= 4:
        raise ValueError(f"row {index} helpfulness label is outside 0-4: {label}")
    return label


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _load_raw_splits(
    path: Path, dataset_config: dict[str, Any]
) -> tuple[Iterable[Mapping[str, Any]], Iterable[Mapping[str, Any]] | None]:
    configured_format = str(dataset_config.get("format", "auto")).lower()
    if path.is_dir() and _looks_like_huggingface(path, configured_format):
        return _load_huggingface(path, dataset_config)
    if path.is_dir():
        return _load_split_directory(path, configured_format)
    return list(_read_tabular(path, configured_format)), None


def _looks_like_huggingface(path: Path, configured_format: str) -> bool:
    return configured_format in {"huggingface", "hf", "save_to_disk"} or any(
        (path / marker).exists() for marker in ("dataset_dict.json", "state.json")
    )


def _load_huggingface(
    path: Path, dataset_config: dict[str, Any]
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]] | None]:
    try:
        from datasets import DatasetDict, load_from_disk  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face save_to_disk input requires `python -m pip install -e '.[huggingface]'`"
        ) from exc
    loaded = load_from_disk(str(path))
    if isinstance(loaded, DatasetDict):
        train_name = dataset_config.get("train_split", "train")
        validation_name = dataset_config.get("validation_split", "validation")
        if train_name not in loaded:
            raise ValueError(f"Hugging Face dataset has no {train_name!r} training split")
        validation = list(loaded[validation_name]) if validation_name in loaded else None
        return list(loaded[train_name]), validation
    return list(loaded), None


def _load_split_directory(
    path: Path, configured_format: str
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]] | None]:
    train_file = _find_split_file(path, "train", configured_format)
    validation_file = _find_split_file(path, "validation", configured_format)
    if train_file is None:
        files = [entry for entry in path.iterdir() if entry.is_file() and _supported(entry)]
        if len(files) == 1:
            return list(_read_tabular(files[0], configured_format)), None
        raise ValueError(f"could not identify a train split in dataset directory: {path}")
    train = list(_read_tabular(train_file, configured_format))
    validation = (
        list(_read_tabular(validation_file, configured_format)) if validation_file else None
    )
    return train, validation


def _find_split_file(path: Path, split: str, configured_format: str) -> Path | None:
    suffixes = [".jsonl", ".jsonl.gz", ".json", ".json.gz", ".csv", ".csv.gz", ".parquet"]
    if configured_format not in {"auto", "directory"}:
        suffixes = [suffix for suffix in suffixes if configured_format in suffix]
    for suffix in suffixes:
        candidate = path / f"{split}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _supported(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in (".json", ".jsonl", ".csv", ".parquet", ".gz"))


def _read_tabular(path: Path, configured_format: str) -> Iterator[Mapping[str, Any]]:
    kind = _infer_format(path, configured_format)
    if kind == "jsonl":
        with _open_text(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                yield value
        return
    if kind == "json":
        with _open_text(path) as handle:
            value = json.load(handle)
        if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
            raise ValueError(f"JSON dataset must be an array of objects: {path}")
        yield from value
        return
    if kind == "csv":
        with _open_text(path) as handle:
            yield from csv.DictReader(handle)
        return
    if kind == "parquet":
        yield from _read_parquet(path)
        return
    raise ValueError(f"unsupported dataset format for {path}")


def _infer_format(path: Path, configured_format: str) -> str:
    aliases = {"jsonlines": "jsonl", "hf": "huggingface"}
    configured_format = aliases.get(configured_format, configured_format)
    if configured_format not in {"auto", "directory"}:
        return configured_format
    name = path.name.lower()
    if name.endswith((".jsonl", ".jsonl.gz")):
        return "jsonl"
    if name.endswith((".json", ".json.gz")):
        return "json"
    if name.endswith((".csv", ".csv.gz")):
        return "csv"
    if name.endswith(".parquet"):
        return "parquet"
    raise ValueError(f"cannot infer dataset format from {path}")


def _open_text(path: Path) -> TextIO:
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _read_parquet(path: Path) -> Iterator[Mapping[str, Any]]:
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-not-found]
    except ImportError:
        try:
            import pandas as pd  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Parquet input requires `python -m pip install -e '.[parquet]'`"
            ) from exc
        yield from pd.read_parquet(path).to_dict(orient="records")
        return
    yield from parquet.read_table(path).to_pylist()
