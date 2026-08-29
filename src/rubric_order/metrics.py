from __future__ import annotations

import csv
import itertools
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .run import load_prepared, read_json, read_jsonl, write_json

PRIMARY_METRICS = (
    "calibration_order_instability",
    "score_change_rate",
    "mean_score_range",
    "pairwise_ranking_flip_rate",
    "tie_change_rate",
)
AGREEMENT_METRICS = (
    "quadratic_weighted_kappa",
    "spearman",
    "mean_absolute_error",
    "mean_signed_error",
    "accuracy",
)


def primary_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    by_example: dict[str, list[int]] = defaultdict(list)
    orders: set[str] = set()
    for row in rows:
        by_example[row["example_id"]].append(int(row["predicted_score"]))
        orders.add(row["order"])
    if len(orders) < 2:
        raise ValueError("primary instability metrics require at least two orders")
    expected_count = len(orders)
    if any(len(scores) != expected_count for scores in by_example.values()):
        raise ValueError("orders do not contain identical held-out evidence")
    instability = statistics.fmean(statistics.pstdev(scores) for scores in by_example.values())
    change_rate = statistics.fmean(float(len(set(scores)) > 1) for scores in by_example.values())
    mean_range = statistics.fmean(max(scores) - min(scores) for scores in by_example.values())

    score_lookup = {
        (row["prompt_id"], row["example_id"], row["order"]): int(row["predicted_score"])
        for row in rows
    }
    prompt_examples: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        prompt_examples[row["prompt_id"]].add(row["example_id"])
    flips = 0
    tie_changes = 0
    comparisons = 0
    sorted_orders = sorted(orders)
    for prompt_id, example_ids in prompt_examples.items():
        for first, second in itertools.combinations(sorted(example_ids), 2):
            for left_order, right_order in itertools.combinations(sorted_orders, 2):
                left = _sign(
                    score_lookup[(prompt_id, first, left_order)]
                    - score_lookup[(prompt_id, second, left_order)]
                )
                right = _sign(
                    score_lookup[(prompt_id, first, right_order)]
                    - score_lookup[(prompt_id, second, right_order)]
                )
                comparisons += 1
                if left * right == -1:
                    flips += 1
                elif (left == 0) != (right == 0):
                    tie_changes += 1
    flip_rate = flips / comparisons if comparisons else 0.0
    tie_rate = tie_changes / comparisons if comparisons else 0.0
    return {
        "calibration_order_instability": instability,
        "score_change_rate": change_rate,
        "mean_score_range": mean_range,
        "pairwise_ranking_flip_rate": flip_rate,
        "tie_change_rate": tie_rate,
        "ranking_flips": flips,
        "ranking_tie_changes": tie_changes,
        "ranking_comparisons": comparisons,
        "heldout_responses": len(by_example),
    }


def agreement_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    human = [int(row["human_score"]) for row in rows]
    predicted = [int(row["predicted_score"]) for row in rows]
    if not human:
        raise ValueError("agreement metrics require at least one prediction")
    qwk = quadratic_weighted_kappa(human, predicted)
    spearman = spearman_correlation(human, predicted)
    return {
        "quadratic_weighted_kappa": qwk,
        "spearman": spearman,
        "mean_absolute_error": statistics.fmean(
            abs(left - right) for left, right in zip(human, predicted, strict=True)
        ),
        "mean_signed_error": statistics.fmean(
            right - left for left, right in zip(human, predicted, strict=True)
        ),
        "accuracy": statistics.fmean(
            left == right for left, right in zip(human, predicted, strict=True)
        ),
        "human_score_distribution": {str(score): human.count(score) for score in range(5)},
        "predicted_score_distribution": {str(score): predicted.count(score) for score in range(5)},
        "undefined_qwk": qwk is None,
        "undefined_spearman": spearman is None,
    }


def quadratic_weighted_kappa(actual: list[int], predicted: list[int]) -> float | None:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("kappa inputs must have equal non-zero length")
    size = 5
    observed = [[0 for _ in range(size)] for _ in range(size)]
    actual_counts = [0] * size
    predicted_counts = [0] * size
    for left, right in zip(actual, predicted, strict=True):
        if not 0 <= left < size or not 0 <= right < size:
            raise ValueError("kappa scores must be from 0 to 4")
        observed[left][right] += 1
        actual_counts[left] += 1
        predicted_counts[right] += 1
    denominator = 0.0
    numerator = 0.0
    count = len(actual)
    for left in range(size):
        for right in range(size):
            weight = ((left - right) / (size - 1)) ** 2
            numerator += weight * observed[left][right]
            denominator += weight * actual_counts[left] * predicted_counts[right] / count
    if denominator == 0:
        return 1.0 if numerator == 0 else None
    return 1.0 - numerator / denominator


def spearman_correlation(left: list[int], right: list[int]) -> float | None:
    if len(left) != len(right) or not left:
        raise ValueError("Spearman inputs must have equal non-zero length")
    return _pearson(_ranks(left), _ranks(right))


def analyze_run(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    run_dir, _manifest, _blocks, _heldout = load_prepared(config, run_id)
    predictions_path = run_dir / "artifacts" / "predictions.jsonl"
    if not predictions_path.exists():
        raise FileNotFoundError("no predictions found; execute judge first")
    predictions = read_jsonl(predictions_path)
    normal = [row for row in predictions if row["replicate"] == 0]
    primary_rows = calculate_primary_table(normal)
    agreement_rows = calculate_agreement_table(normal)
    paired_rows = calculate_paired_table(primary_rows)
    repeat_rows = calculate_repeat_control_table(predictions)
    repeat_variation_rows = calculate_repeat_variation_table(predictions)
    between_repeat_rows = calculate_between_repeat_table(primary_rows, repeat_variation_rows)
    rubric_text_rows, rubric_similarity_rows = rubric_diagnostics(run_dir)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_csv(metrics_dir / "primary_metrics.csv", primary_rows)
    write_csv(metrics_dir / "agreement_metrics.csv", agreement_rows)
    write_csv(metrics_dir / "paired_differences.csv", paired_rows)
    write_csv(metrics_dir / "repeat_control.csv", repeat_rows)
    write_csv(metrics_dir / "repeat_variation.csv", repeat_variation_rows)
    write_csv(metrics_dir / "between_vs_repeat.csv", between_repeat_rows)
    write_csv(metrics_dir / "rubric_text.csv", rubric_text_rows)
    write_csv(metrics_dir / "rubric_similarity.csv", rubric_similarity_rows)
    summary = bootstrap_summary(
        normal,
        repetitions=config["analysis"]["bootstrap_repetitions"],
        seed=config["seed"],
        confidence=config["analysis"]["confidence"],
    )
    summary["agreement"] = aggregate_agreement_table(agreement_rows)
    summary["agreement_pooled_descriptive"] = {
        method: agreement_metrics(_deduplicate_static(rows) if method == "static" else rows)
        for method in sorted({row["method"] for row in normal})
        for rows in [[row for row in normal if row["method"] == method]]
    }
    summary["repeat_variation"] = repeat_variation_rows
    summary["between_vs_repeat"] = between_repeat_rows
    summary["definitions"] = {
        "calibration_order_instability": (
            "Mean held-out-response population SD across order-derived scores."
        ),
        "score_change_rate": (
            "Fraction of held-out responses with more than one score across orders."
        ),
        "mean_score_range": "Mean per-response maximum minus minimum score across orders.",
        "pairwise_ranking_flip_rate": (
            "Strict reversals divided by response-pair/order-pair comparisons within prompts."
        ),
        "tie_change_rate": (
            "Tie/non-tie changes divided by response-pair/order-pair comparisons within prompts."
        ),
        "bootstrap_unit": "Calibration blocks and held-out prompt groups, resampled jointly.",
    }
    summary["plots"] = []
    summary["plot_warning"] = None
    write_json(metrics_dir / "summary.json", summary)
    if config["analysis"].get("make_plots", True):
        try:
            from .plotting import create_figures

            summary["plots"] = create_figures(summary, agreement_rows, normal, run_dir / "figures")
        except ImportError:
            summary["plot_warning"] = (
                "Matplotlib is unavailable; install with `python -m pip install -e '.[plots]'` "
                "and rerun analyze."
            )
        write_json(metrics_dir / "summary.json", summary)
    return summary


def calculate_primary_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group(rows, ("method", "block_id", "replicate"))
    table: list[dict[str, Any]] = []
    for (method, block_id, replicate), group_rows in sorted(grouped.items()):
        result = primary_metrics(group_rows)
        table.append(
            {
                "method": method,
                "block_id": block_id,
                "replicate": replicate,
                **result,
            }
        )
    return table


def calculate_agreement_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    non_static = [row for row in rows if row["method"] != "static"]
    grouped = _group(non_static, ("method", "block_id", "order", "replicate"))
    table: list[dict[str, Any]] = []
    for (method, block_id, order, replicate), group_rows in sorted(grouped.items()):
        table.append(
            {
                "method": method,
                "block_id": block_id,
                "order": order,
                "replicate": replicate,
                **agreement_metrics(group_rows),
            }
        )
    static = _deduplicate_static([row for row in rows if row["method"] == "static"])
    if static:
        table.append(
            {
                "method": "static",
                "block_id": "deduplicated",
                "order": "deduplicated",
                "replicate": 0,
                **agreement_metrics(static),
            }
        )
    return table


def aggregate_agreement_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Primary agreement is the unweighted mean of block-by-order conditions."""
    grouped = _group(rows, ("method",))
    result: dict[str, Any] = {}
    for (method,), method_rows in sorted(grouped.items()):
        result[method] = {
            metric: statistics.fmean(
                float(row[metric]) for row in method_rows if row[metric] is not None
            )
            if any(row[metric] is not None for row in method_rows)
            else None
            for metric in AGREEMENT_METRICS
        }
        result[method]["condition_count"] = len(method_rows)
        result[method]["aggregation"] = "mean_of_block_order_conditions"
        result[method]["undefined_metric_flags"] = {
            metric: sum(row[metric] is None for row in method_rows)
            for metric in ("quadratic_weighted_kappa", "spearman")
        }
    return result


def calculate_paired_table(primary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["method"], row["block_id"]): row for row in primary_rows}
    block_ids = sorted(
        {row["block_id"] for row in primary_rows if row["method"] == "sequential"}
        & {row["block_id"] for row in primary_rows if row["method"] == "batch"}
    )
    return [
        {
            "block_id": block_id,
            "contrast": "sequential_minus_batch",
            **{
                metric: lookup[("sequential", block_id)][metric]
                - lookup[("batch", block_id)][metric]
                for metric in PRIMARY_METRICS
            },
        }
        for block_id in block_ids
    ]


def calculate_repeat_control_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = {
        (row["method"], row["block_id"], row["order"], row["example_id"]): row
        for row in rows
        if row["replicate"] == 0
    }
    grouped: dict[tuple[Any, ...], list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        if row["replicate"] == 0:
            continue
        key = (row["method"], row["block_id"], row["order"], row["example_id"])
        if key in baseline:
            group_key = (row["method"], row["block_id"], row["order"], row["replicate"])
            grouped[group_key].append(
                (int(baseline[key]["predicted_score"]), int(row["predicted_score"]))
            )
    return [
        {
            "method": key[0],
            "block_id": key[1],
            "order": key[2],
            "replicate": key[3],
            "score_change_rate": statistics.fmean(float(a != b) for a, b in pairs),
            "mean_absolute_change": statistics.fmean(abs(a - b) for a, b in pairs),
            "responses": len(pairs),
        }
        for key, pairs in sorted(grouped.items())
    ]


def repeat_variation_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    replicates = sorted({int(row["replicate"]) for row in rows})
    if len(replicates) < 2:
        raise ValueError("repeat variation requires at least two replicates")
    converted = [{**row, "order": f"replicate_{row['replicate']}"} for row in rows]
    values = primary_metrics(converted)
    pair_rows = calculate_repeat_control_table(rows)
    exact = (
        statistics.fmean(1.0 - float(row["score_change_rate"]) for row in pair_rows)
        if pair_rows
        else None
    )
    return {
        "repeat_instability": values["calibration_order_instability"],
        "repeat_score_change_rate": values["score_change_rate"],
        "repeat_mean_score_range": values["mean_score_range"],
        "repeat_strict_ranking_flip_rate": values["pairwise_ranking_flip_rate"],
        "repeat_tie_change_rate": values["tie_change_rate"],
        "exact_replicate_agreement": exact,
        "replicate_count": len(replicates),
    }


def calculate_repeat_variation_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group(
        [row for row in rows if row["method"] in {"batch", "sequential"}],
        ("method", "block_id", "order"),
    )
    table: list[dict[str, Any]] = []
    for (method, block_id, order), group_rows in sorted(grouped.items()):
        if len({row["replicate"] for row in group_rows}) < 2:
            continue
        table.append(
            {
                "method": method,
                "block_id": block_id,
                "order": order,
                **repeat_variation_metrics(group_rows),
            }
        )
    return table


def calculate_between_repeat_table(
    primary_rows: list[dict[str, Any]], repeat_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    repeat_lookup = {(row["method"], row["block_id"]): row for row in repeat_rows}
    table: list[dict[str, Any]] = []
    for row in primary_rows:
        repeat = repeat_lookup.get((row["method"], row["block_id"]))
        if repeat is None:
            continue
        between = float(row["calibration_order_instability"])
        noise = float(repeat["repeat_instability"])
        table.append(
            {
                "method": row["method"],
                "block_id": row["block_id"],
                "between_order_instability": between,
                "repeat_instability": noise,
                "between_minus_repeat": between - noise,
                "between_over_repeat": between / noise if noise else None,
                "between_exceeds_zero_observed_noise": bool(noise == 0 and between > 0),
                "zero_repeat_ratio_convention": "ratio omitted when repeat instability is zero",
            }
        )
    return table


def rubric_diagnostics(
    run_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index_path = run_dir / "artifacts" / "rubrics.jsonl"
    if not index_path.exists():
        return [], []
    records: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    text_rows: list[dict[str, Any]] = []
    for index_row in read_jsonl(index_path):
        if index_row["replicate"] != 0:
            continue
        rubric = read_json(run_dir / index_row["rubric_path"])
        text = json.dumps(rubric, ensure_ascii=False, sort_keys=True)
        records.append((index_row, rubric, text))
        text_rows.append(
            {
                "method": index_row["method"],
                "block_id": index_row["block_id"],
                "order": index_row["order"],
                "replicate": index_row["replicate"],
                "characters": len(text),
                "approximate_tokens": len(_tokens(text)),
                "criteria_count": len(rubric["criteria"]),
            }
        )
    similarity_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[tuple[dict[str, Any], dict[str, Any], str]]] = defaultdict(
        list
    )
    for record in records:
        grouped[(record[0]["method"], record[0]["block_id"])].append(record)
    for (method, block_id), group in sorted(grouped.items()):
        ordered_group = sorted(group, key=lambda item: item[0]["order"])
        common_similarities = common_tfidf_cosines([item[2] for item in ordered_group])
        for left, right in itertools.combinations(ordered_group, 2):
            left_names = {criterion["name"].casefold() for criterion in left[1]["criteria"]}
            right_names = {criterion["name"].casefold() for criterion in right[1]["criteria"]}
            union = left_names | right_names
            similarity_rows.append(
                {
                    "method": method,
                    "block_id": block_id,
                    "left_order": left[0]["order"],
                    "right_order": right[0]["order"],
                    "criterion_name_jaccard": len(left_names & right_names) / len(union)
                    if union
                    else 1.0,
                    "tfidf_cosine": common_similarities[
                        (ordered_group.index(left), ordered_group.index(right))
                    ],
                    "tfidf_fit_scope": "all_rubrics_in_method_block",
                }
            )
    return text_rows, similarity_rows


def tfidf_cosine(left: str, right: str) -> float:
    return common_tfidf_cosines([left, right])[(0, 1)]


def common_tfidf_cosines(documents: list[str]) -> dict[tuple[int, int], float]:
    """Fit one IDF vocabulary over the complete comparison set."""
    if len(documents) < 2:
        return {}
    counts_by_document = [Counter(_tokens(document)) for document in documents]
    vocabulary = set().union(*counts_by_document)
    vectors: list[dict[str, float]] = []
    document_count = len(documents)
    for counts in counts_by_document:
        total = sum(counts.values()) or 1
        vector: dict[str, float] = {}
        for token in vocabulary:
            document_frequency = sum(token in other for other in counts_by_document)
            inverse_document_frequency = (
                math.log((1 + document_count) / (1 + document_frequency)) + 1
            )
            vector[token] = counts[token] / total * inverse_document_frequency
        vectors.append(vector)
    result: dict[tuple[int, int], float] = {}
    for left, right in itertools.combinations(range(document_count), 2):
        dot = sum(vectors[left][token] * vectors[right][token] for token in vocabulary)
        left_norm = math.sqrt(sum(value * value for value in vectors[left].values()))
        right_norm = math.sqrt(sum(value * value for value in vectors[right].values()))
        result[(left, right)] = dot / (left_norm * right_norm) if left_norm and right_norm else 0.0
    return result


def bootstrap_summary(
    rows: list[dict[str, Any]], repetitions: int, seed: int, confidence: float
) -> dict[str, Any]:
    methods = sorted({row["method"] for row in rows})
    summary: dict[str, Any] = {"bootstrap_repetitions": repetitions, "primary": {}, "agreement": {}}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        primary_estimate = _mean_primary(method_rows)
        draws = _bootstrap_clustered(
            method_rows,
            repetitions,
            seed + _stable_offset(method),
            _mean_primary,
        )
        summary["primary"][method] = {
            metric: _interval(
                primary_estimate[metric], [draw[metric] for draw in draws], confidence
            )
            for metric in PRIMARY_METRICS
        }
        summary["agreement"][method] = {}
        for order in sorted({row["order"] for row in method_rows}):
            order_rows = [row for row in method_rows if row["order"] == order]
            estimate = agreement_metrics(order_rows)
            agreement_draws = _bootstrap_clustered(
                order_rows,
                repetitions,
                seed + _stable_offset(f"{method}:{order}"),
                agreement_metrics,
            )
            summary["agreement"][method][order] = {
                metric: _interval(
                    estimate[metric],
                    [draw[metric] for draw in agreement_draws if draw[metric] is not None],
                    confidence,
                )
                for metric in AGREEMENT_METRICS
            }
    summary["paired_sequential_minus_batch"] = {}
    if {"sequential", "batch"} <= set(methods):
        paired_estimate = _paired_primary(rows)
        paired_draws = _bootstrap_paired(
            rows,
            repetitions,
            seed + 9_973,
        )
        summary["paired_sequential_minus_batch"] = {
            metric: _interval(
                paired_estimate[metric], [draw[metric] for draw in paired_draws], confidence
            )
            for metric in PRIMARY_METRICS
        }
    return summary


def _bootstrap_clustered(
    rows: list[dict[str, Any]],
    repetitions: int,
    seed: int,
    statistic: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> list[dict[str, Any]]:
    if repetitions == 0:
        return []
    blocks = sorted({int(row["block_id"]) for row in rows})
    prompts = sorted({row["prompt_id"] for row in rows})
    rng = random.Random(seed)
    draws: list[dict[str, Any]] = []
    for _ in range(repetitions):
        sampled_blocks = rng.choices(blocks, k=len(blocks))
        sampled_prompts = rng.choices(prompts, k=len(prompts))
        sampled = _cluster_sample(rows, sampled_blocks, sampled_prompts)
        draws.append(statistic(sampled))
    return draws


def _bootstrap_paired(
    rows: list[dict[str, Any]], repetitions: int, seed: int
) -> list[dict[str, float]]:
    relevant = [row for row in rows if row["method"] in {"sequential", "batch"}]
    if repetitions == 0:
        return []
    sequential_blocks = {int(row["block_id"]) for row in relevant if row["method"] == "sequential"}
    batch_blocks = {int(row["block_id"]) for row in relevant if row["method"] == "batch"}
    blocks = sorted(sequential_blocks & batch_blocks)
    prompts = sorted({row["prompt_id"] for row in relevant})
    rng = random.Random(seed)
    draws: list[dict[str, float]] = []
    for _ in range(repetitions):
        sampled = _cluster_sample(
            relevant,
            rng.choices(blocks, k=len(blocks)),
            rng.choices(prompts, k=len(prompts)),
        )
        draws.append(_paired_primary(sampled))
    return draws


def _cluster_sample(
    rows: list[dict[str, Any]], sampled_blocks: list[int], sampled_prompts: list[str]
) -> list[dict[str, Any]]:
    lookup: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        lookup[(int(row["block_id"]), row["prompt_id"])].append(row)
    sampled: list[dict[str, Any]] = []
    for block_occurrence, block_id in enumerate(sampled_blocks):
        for prompt_occurrence, prompt_id in enumerate(sampled_prompts):
            for row in lookup[(block_id, prompt_id)]:
                copied = dict(row)
                copied["block_id"] = block_occurrence
                copied["prompt_id"] = f"{prompt_occurrence}:{prompt_id}"
                copied["example_id"] = f"{prompt_occurrence}:{row['example_id']}"
                sampled.append(copied)
    return sampled


def _mean_primary(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped = _group(rows, ("block_id",))
    values = [primary_metrics(group_rows) for group_rows in grouped.values()]
    return {
        metric: statistics.fmean(float(value[metric]) for value in values)
        for metric in PRIMARY_METRICS
    }


def _paired_primary(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_method = {
        method: _mean_primary([row for row in rows if row["method"] == method])
        for method in ("sequential", "batch")
    }
    return {
        metric: by_method["sequential"][metric] - by_method["batch"][metric]
        for metric in PRIMARY_METRICS
    }


def _interval(estimate: float | None, draws: list[Any], confidence: float) -> dict[str, Any]:
    numeric = [float(value) for value in draws if value is not None and math.isfinite(float(value))]
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": estimate,
        "lower": _quantile(numeric, alpha) if numeric else None,
        "upper": _quantile(numeric, 1.0 - alpha) if numeric else None,
    }


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fields:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _group(
    rows: Iterable[dict[str, Any]], keys: tuple[str, ...]
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def _deduplicate_static(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use each static example once; logical order copies are reporting materialization only."""
    indexed: dict[str, dict[str, Any]] = {}
    for row in sorted(
        rows,
        key=lambda item: (
            int(item.get("block_id", 0)),
            str(item.get("order", "")),
            str(item["example_id"]),
        ),
    ):
        indexed.setdefault(row["example_id"], row)
    return list(indexed.values())


def _ranks(values: list[int]) -> list[float]:
    sorted_indices = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(values):
        end = start
        while (
            end + 1 < len(values)
            and values[sorted_indices[end + 1]] == values[sorted_indices[start]]
        ):
            end += 1
        average_rank = (start + end) / 2.0 + 1.0
        for offset in range(start, end + 1):
            ranks[sorted_indices[offset]] = average_rank
        start = end + 1
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    left_sum = sum((value - left_mean) ** 2 for value in left)
    right_sum = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_sum * right_sum)
    return numerator / denominator if denominator else None


def _tokens(text: str) -> list[str]:
    return re.findall(r"(?u)\b\w\w+\b", text.casefold())


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _stable_offset(value: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(value))
