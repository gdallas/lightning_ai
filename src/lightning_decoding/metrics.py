from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

import torch


def distinct_valid_per_prompt(records: list[dict[str, Any]]) -> float:
    by_prompt: dict[str, set[str]] = defaultdict(set)
    for row in records:
        if row["valid"]:
            by_prompt[row["prompt_id"]].add(row["normalized"])
    return mean(len(values) for values in by_prompt.values()) if by_prompt else 0.0


def mode_share(records: list[dict[str, Any]]) -> float:
    valid = [row for row in records if row["valid"]]
    if not valid:
        return 0.0
    by_prompt: dict[str, Counter[str]] = defaultdict(Counter)
    for row in valid:
        by_prompt[row["prompt_id"]][row["normalized"]] += 1
    modal_hits = sum(counter.most_common(1)[0][1] for counter in by_prompt.values())
    return modal_hits / len(valid)


def bootstrap_ci_by_prompt(
    records: list[dict[str, Any]],
    *,
    iterations: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_prompt[row["prompt_id"]].append(row)
    prompt_ids = list(by_prompt)
    if not prompt_ids:
        return (0.0, 0.0)

    generator = torch.Generator().manual_seed(seed)
    samples: list[float] = []
    for _ in range(iterations):
        idxs = torch.randint(len(prompt_ids), (len(prompt_ids),), generator=generator)
        sampled_rows: list[dict[str, Any]] = []
        for idx in idxs.tolist():
            sampled_rows.extend(by_prompt[prompt_ids[idx]])
        samples.append(distinct_valid_per_prompt(sampled_rows))
    samples.sort()
    lo = samples[int(0.025 * len(samples))]
    hi = samples[min(len(samples) - 1, int(0.975 * len(samples)))]
    return lo, hi


def summarize(records: list[dict[str, Any]], answer_space_sizes: dict[str, int]) -> dict[str, float]:
    total = len(records)
    valid = sum(1 for row in records if row["valid"])
    distinct = distinct_valid_per_prompt(records)
    avg_space = mean(answer_space_sizes.values()) if answer_space_sizes else 0.0
    ci_lo, ci_hi = bootstrap_ci_by_prompt(records)
    return {
        "trials": float(total),
        "validity_rate": valid / total if total else 0.0,
        "distinct_valid_per_prompt": distinct,
        "coverage": distinct / avg_space if avg_space else 0.0,
        "mode_share": mode_share(records),
        "distinct_valid_ci_low": ci_lo,
        "distinct_valid_ci_high": ci_hi,
    }

