from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any

ORDER_NAMES = ("ascending", "descending", "random", "reverse_random")


def select_partitions(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]] | None,
    heldout_prompts: int,
    responses_per_prompt: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    heldout_source = validation if validation is not None else train
    heldout = select_heldout(
        heldout_source,
        prompt_count=heldout_prompts,
        responses_per_prompt=responses_per_prompt,
        seed=seed,
    )
    heldout_prompt_ids = {row["prompt_id"] for row in heldout}
    calibration_pool = [row for row in train if row["prompt_id"] not in heldout_prompt_ids]
    if not calibration_pool:
        raise ValueError("no calibration rows remain after prompt-grouped held-out selection")
    leakage = {row["prompt_id"] for row in calibration_pool} & heldout_prompt_ids
    if leakage:
        raise AssertionError(f"prompt leakage detected: {sorted(leakage)}")
    metadata = {
        "official_validation_used": validation is not None,
        "heldout_prompt_count": len(heldout_prompt_ids),
        "heldout_response_count": len(heldout),
        "calibration_pool_count": len(calibration_pool),
        "prompt_leakage_count": 0,
    }
    return calibration_pool, heldout, metadata


def select_heldout(
    rows: list[dict[str, Any]], prompt_count: int, responses_per_prompt: int, seed: int
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["prompt_id"]].append(row)
    if len(groups) < prompt_count:
        raise ValueError(
            f"requested {prompt_count} held-out prompts but only {len(groups)} are available"
        )
    rng = random.Random(seed)
    prompt_ids = sorted(groups)
    rng.shuffle(prompt_ids)
    prompt_ids.sort(key=lambda prompt_id: len(groups[prompt_id]) < responses_per_prompt)
    selected: list[dict[str, Any]] = []
    for prompt_id in prompt_ids[:prompt_count]:
        candidates = sorted(groups[prompt_id], key=lambda row: row["example_id"])
        group_rng = random.Random(f"{seed}:{prompt_id}")
        group_rng.shuffle(candidates)
        selected.extend(candidates[:responses_per_prompt])
    return sorted(selected, key=lambda row: (row["prompt_id"], row["example_id"]))


def build_calibration_blocks(
    rows: list[dict[str, Any]], block_count: int, examples_per_score: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_score: dict[int, list[dict[str, Any]]] = {score: [] for score in range(5)}
    for row in rows:
        by_score[row["human_score"]].append(row)
    for score, candidates in by_score.items():
        unique_ids = {row["example_id"] for row in candidates}
        if len(unique_ids) < examples_per_score:
            raise ValueError(
                f"score {score} has {len(unique_ids)} examples; need at least {examples_per_score}"
            )
        candidates.sort(key=lambda row: row["example_id"])
        random.Random(seed + score * 104729).shuffle(candidates)

    used_ids: set[str] = set()
    overlap_events: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    for block_index in range(block_count):
        block_rows: list[dict[str, Any]] = []
        block_prompt_ids: set[str] = set()
        for score in range(5):
            selected = _choose_for_score(
                by_score[score], examples_per_score, used_ids, block_prompt_ids
            )
            for row in selected:
                if row["example_id"] in used_ids:
                    overlap_events.append(
                        {"block": block_index, "score": score, "example_id": row["example_id"]}
                    )
                used_ids.add(row["example_id"])
                block_prompt_ids.add(row["prompt_id"])
                block_rows.append(row)
        orders = make_orders(block_rows, seed=seed, block_index=block_index)
        blocks.append(
            {
                "block_id": block_index,
                "examples": block_rows,
                "orders": orders,
                "distinct_prompt_count": len(block_prompt_ids),
            }
        )
    counts = Counter(row["example_id"] for block in blocks for row in block["examples"])
    metadata = {
        "overlap_event_count": len(overlap_events),
        "overlap_events": overlap_events,
        "reused_example_ids": sorted(key for key, count in counts.items() if count > 1),
        "all_blocks_disjoint": all(count == 1 for count in counts.values()),
    }
    return blocks, metadata


def _choose_for_score(
    candidates: list[dict[str, Any]],
    count: int,
    used_ids: set[str],
    block_prompt_ids: set[str],
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    chosen_ids: set[str] = set()
    priorities = (
        lambda row: row["example_id"] not in used_ids and row["prompt_id"] not in block_prompt_ids,
        lambda row: row["example_id"] not in used_ids,
        lambda row: row["prompt_id"] not in block_prompt_ids,
        lambda row: True,
    )
    for predicate in priorities:
        for row in candidates:
            if row["example_id"] in chosen_ids or not predicate(row):
                continue
            chosen.append(row)
            chosen_ids.add(row["example_id"])
            block_prompt_ids.add(row["prompt_id"])
            if len(chosen) == count:
                return chosen
    raise ValueError("could not construct a balanced calibration block")


def make_orders(
    examples: list[dict[str, Any]], seed: int, block_index: int
) -> dict[str, list[str]]:
    ascending = [
        row["example_id"]
        for row in sorted(examples, key=lambda row: (row["human_score"], row["example_id"]))
    ]
    descending = [
        row["example_id"]
        for row in sorted(examples, key=lambda row: (-row["human_score"], row["example_id"]))
    ]
    random_order = sorted(ascending)
    random.Random(seed + (block_index + 1) * 1_000_003).shuffle(random_order)
    orders = {
        "ascending": ascending,
        "descending": descending,
        "random": random_order,
        "reverse_random": list(reversed(random_order)),
    }
    assert_order_invariants(orders)
    return orders


def assert_order_invariants(orders: dict[str, list[str]]) -> None:
    if set(orders) != set(ORDER_NAMES):
        raise ValueError(f"orders must contain exactly {ORDER_NAMES}")
    reference = orders["ascending"]
    if len(reference) != len(set(reference)):
        raise ValueError("calibration order contains duplicate example IDs")
    for name, order in orders.items():
        if len(order) != len(reference) or set(order) != set(reference):
            raise ValueError(f"order {name} does not contain the same evidence")
    if orders["reverse_random"] != list(reversed(orders["random"])):
        raise ValueError("reverse_random is not the exact reverse of random")


def ordered_examples(block: dict[str, Any], order: str) -> list[dict[str, Any]]:
    by_id = {row["example_id"]: row for row in block["examples"]}
    return [by_id[example_id] for example_id in block["orders"][order]]
