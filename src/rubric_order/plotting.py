from __future__ import annotations

import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any


def create_figures(
    summary: dict[str, Any],
    agreement_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    output_dir: Path,
) -> list[str]:
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    primary = summary["primary"]
    methods = [method for method in ("batch", "sequential") if method in primary]
    if methods:
        values = [
            primary[method]["calibration_order_instability"]["estimate"] for method in methods
        ]
        lower = [primary[method]["calibration_order_instability"]["lower"] for method in methods]
        upper = [primary[method]["calibration_order_instability"]["upper"] for method in methods]
        errors = [
            [
                value - low if low is not None else 0
                for value, low in zip(values, lower, strict=True)
            ],
            [
                high - value if high is not None else 0
                for value, high in zip(values, upper, strict=True)
            ],
        ]
        figure, axis = plt.subplots(figsize=(4.5, 3.2))
        axis.bar(
            methods, values, color=["#9AA5B1", "#315E7D"][: len(methods)], yerr=errors, capsize=4
        )
        axis.set_ylabel("Calibration-order instability")
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        path = output_dir / "instability_sequential_vs_batch.png"
        figure.savefig(path, dpi=300)
        plt.close(figure)
        paths.append(path.name)

    if agreement_rows:
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in agreement_rows:
            value = row["quadratic_weighted_kappa"]
            if value is not None:
                grouped[(row["method"], row["order"])].append(float(value))
        labels = sorted(grouped)
        means = [sum(grouped[label]) / len(grouped[label]) for label in labels]
        figure, axis = plt.subplots(figsize=(7.0, 3.4))
        axis.bar(
            range(len(labels)),
            means,
            color=["#607D8B", "#8D9F87", "#315E7D"] * ((len(labels) + 2) // 3),
        )
        axis.set_xticks(
            range(len(labels)),
            [f"{method}\n{order}" for method, order in labels],
            rotation=35,
            ha="right",
        )
        axis.set_ylabel("Quadratic weighted kappa")
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        path = output_dir / "agreement_by_method_order.png"
        figure.savefig(path, dpi=300)
        plt.close(figure)
        paths.append(path.name)

    sequential = [row for row in predictions if row["method"] == "sequential"]
    orders = sorted({row["order"] for row in sequential})
    if len(orders) >= 2:
        lookup = {
            (row["block_id"], row["example_id"], row["order"]): row["predicted_score"]
            for row in sequential
        }
        matrix = [[0.0 for _ in orders] for _ in orders]
        keys = sorted({(row["block_id"], row["example_id"]) for row in sequential})
        for left_index, right_index in itertools.combinations(range(len(orders)), 2):
            valid = [
                key
                for key in keys
                if (*key, orders[left_index]) in lookup and (*key, orders[right_index]) in lookup
            ]
            rate = (
                sum(
                    lookup[(*key, orders[left_index])] != lookup[(*key, orders[right_index])]
                    for key in valid
                )
                / len(valid)
                if valid
                else 0.0
            )
            matrix[left_index][right_index] = rate
            matrix[right_index][left_index] = rate
        figure, axis = plt.subplots(figsize=(4.5, 3.8))
        image = axis.imshow(matrix, vmin=0, vmax=max(max(row) for row in matrix) or 1, cmap="Blues")
        axis.set_xticks(range(len(orders)), orders, rotation=35, ha="right")
        axis.set_yticks(range(len(orders)), orders)
        figure.colorbar(image, ax=axis, label="Score disagreement rate", shrink=0.8)
        figure.tight_layout()
        path = output_dir / "sequential_order_disagreement_heatmap.png"
        figure.savefig(path, dpi=300)
        plt.close(figure)
        paths.append(path.name)
    return paths
