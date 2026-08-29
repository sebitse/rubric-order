from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Selection:
    blocks: frozenset[int] | None = None
    orders: frozenset[str] | None = None
    methods: frozenset[str] | None = None
    replicates: frozenset[int] | None = None

    def allows_block(self, block_id: int) -> bool:
        return self.blocks is None or block_id in self.blocks

    def allows_order(self, order: str) -> bool:
        return self.orders is None or order in self.orders

    def allows_method(self, method: str) -> bool:
        return self.methods is None or method in self.methods

    def allows_replicate(self, replicate: int) -> bool:
        return self.replicates is None or replicate in self.replicates


@dataclass(frozen=True)
class RubricJob:
    method: str
    block_id: int
    order: str
    replicate: int
    repeat_control: bool
    example_count: int

    @property
    def updater_calls(self) -> int:
        return self.example_count if self.method == "sequential" else 1


def plan_rubric_jobs(config: dict[str, Any], selection: Selection) -> list[RubricJob]:
    sampling = config["sampling"]
    block_count = sampling["calibration_blocks"]
    example_count = sampling["examples_per_score"] * 5
    jobs: list[RubricJob] = []
    for block_id in range(block_count):
        if not selection.allows_block(block_id):
            continue
        for order in config["orders"]:
            if not selection.allows_order(order):
                continue
            for method in ("batch", "sequential"):
                if selection.allows_method(method) and selection.allows_replicate(0):
                    jobs.append(RubricJob(method, block_id, order, 0, False, example_count))
    repeat = config["repeat_control"]
    if repeat["enabled"]:
        for block_id in repeat["blocks"]:
            if block_id >= block_count:
                raise ValueError(f"repeat-control block {block_id} is outside configured blocks")
            if not selection.allows_block(block_id):
                continue
            for order in repeat["orders"]:
                if not selection.allows_order(order):
                    continue
                for method in repeat["methods"]:
                    if method not in {"batch", "sequential"}:
                        raise ValueError(f"repeat control cannot generate method {method}")
                    if not selection.allows_method(method):
                        continue
                    for replicate in repeat["replicates"]:
                        if replicate < 1:
                            raise ValueError("repeat-control replicate IDs must be positive")
                        if selection.allows_replicate(replicate):
                            jobs.append(
                                RubricJob(
                                    method,
                                    block_id,
                                    order,
                                    replicate,
                                    True,
                                    example_count,
                                )
                            )
    return jobs


def estimate_calls(
    config: dict[str, Any],
    selection: Selection | None = None,
    heldout_response_count: int | None = None,
) -> dict[str, int]:
    selected = selection or Selection()
    jobs = plan_rubric_jobs(config, selected)
    normal_jobs = [job for job in jobs if not job.repeat_control]
    repeat_jobs = [job for job in jobs if job.repeat_control]
    sequential = sum(job.updater_calls for job in normal_jobs if job.method == "sequential")
    batch = sum(job.updater_calls for job in normal_jobs if job.method == "batch")
    if heldout_response_count is None:
        sampling = config["sampling"]
        heldout_response_count = sampling["heldout_prompts"] * sampling["responses_per_prompt"]
    normal_rubrics = len(normal_jobs)
    static_calls = heldout_response_count if selected.allows_method("static") else 0
    judge = heldout_response_count * normal_rubrics + static_calls
    repeat_updater = sum(job.updater_calls for job in repeat_jobs)
    repeat_judge = heldout_response_count * len(repeat_jobs)
    repeat_total = repeat_updater + repeat_judge
    updater_tokens = config["models"]["updater_max_tokens"]
    judge_tokens = config["models"]["judge_max_tokens"]
    approximate_tokens = (sequential + batch + repeat_updater) * updater_tokens + (
        judge + repeat_judge
    ) * judge_tokens
    return {
        "sequential_updater_calls": sequential,
        "batch_updater_calls": batch,
        "judge_calls": judge,
        "repeat_control_calls": repeat_total,
        "repeat_control_updater_calls": repeat_updater,
        "repeat_control_judge_calls": repeat_judge,
        "rubric_stage_calls": sequential + batch + repeat_updater,
        "judge_stage_calls": judge + repeat_judge,
        "total_inference_calls": sequential + batch + judge + repeat_total,
        "approximate_max_generated_tokens": approximate_tokens,
        "heldout_response_count": heldout_response_count,
    }
