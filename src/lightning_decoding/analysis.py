from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from typing import Any


def load_run(run_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Read a capture run's trial rows and per-prompt lens records.

    Returns ``(trials, lens_by_prompt)``. ``lens_by_prompt`` is keyed by ``prompt_id``.
    """
    run_dir = Path(run_dir)
    trials: list[dict[str, Any]] = []
    with (run_dir / "trials.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if "__meta__" in row:
                continue
            trials.append(row)

    lens_by_prompt: dict[str, dict[str, Any]] = {}
    lens_path = run_dir / "lens_per_prompt.jsonl"
    if lens_path.exists():
        with lens_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                lens_by_prompt[record["prompt_id"]] = record
    return trials, lens_by_prompt


def modal_token_id(lens_record: dict[str, Any]) -> int:
    """The token the model commits to under the lens: the final-layer lens argmax.

    Because the logit lens on the last block equals the model's own head, this is the
    clean-pass greedy (modal) token for the prompt.
    """
    return int(lens_record["lens_argmax_per_layer"][-1])


def classify_valid_trials(
    trials: list[dict[str, Any]], lens_by_prompt: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Label each valid trial that has a commitment depth as ``modal`` or ``novel``.

    ``modal``: the produced token equals the prompt's clean-pass greedy token.
    ``novel``: a valid produced token that differs from the greedy token.
    """
    classified: list[dict[str, Any]] = []
    for row in trials:
        if not row.get("valid"):
            continue
        if "commitment_depth" not in row:
            continue
        lens_record = lens_by_prompt.get(row["prompt_id"])
        if lens_record is None:
            continue
        label = "modal" if row["token_id"] == modal_token_id(lens_record) else "novel"
        classified.append(
            {
                "prompt_id": row["prompt_id"],
                "token_id": row["token_id"],
                "token_str": row.get("token_str"),
                "commitment_depth": row["commitment_depth"],
                "label": label,
            }
        )
    return classified


def depth_groups(classified: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    modal = [row["commitment_depth"] for row in classified if row["label"] == "modal"]
    novel = [row["commitment_depth"] for row in classified if row["label"] == "novel"]
    return modal, novel


def _summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None}
    return {"n": len(values), "mean": mean(values), "median": median(values)}


def compare_depths(modal: list[int], novel: list[int]) -> dict[str, Any]:
    """Mann-Whitney U comparison of commitment depth for modal vs novel tokens.

    Uses a two-sided alternative; ``p_value`` is ``None`` when either group is empty.
    ``mean_depth_gap`` is ``novel_mean - modal_mean`` (positive => novel commits later).
    """
    result: dict[str, Any] = {
        "modal": _summary(modal),
        "novel": _summary(novel),
        "u_statistic": None,
        "p_value": None,
        "mean_depth_gap": None,
    }
    if modal and novel:
        from scipy.stats import mannwhitneyu

        u_stat, p_value = mannwhitneyu(novel, modal, alternative="two-sided")
        result["u_statistic"] = float(u_stat)
        result["p_value"] = float(p_value)
        result["mean_depth_gap"] = mean(novel) - mean(modal)
    return result


def save_commitment_histogram(
    modal: list[int],
    novel: list[int],
    output_path: str | Path,
    *,
    num_layers: int | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    max_depth = max([*modal, *novel, num_layers or 0], default=0)
    bins = range(0, max_depth + 2)
    fig, ax = plt.subplots()
    ax.hist(
        [modal, novel],
        bins=bins,
        label=[f"modal (n={len(modal)})", f"novel (n={len(novel)})"],
        align="left",
        rwidth=0.9,
    )
    ax.set_xlabel("commitment depth (layer)")
    ax.set_ylabel("valid trials")
    ax.set_title("Layerwise commitment depth: modal vs novel")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def analyze_depth(run_dir: str | Path) -> dict[str, Any]:
    """Full pipeline: classify a capture run and write comparison + histogram artifacts."""
    run_dir = Path(run_dir)
    trials, lens_by_prompt = load_run(run_dir)
    if not lens_by_prompt:
        raise SystemExit(
            f"{run_dir} has no lens_per_prompt.jsonl; rerun with experiment.capture_hidden_states"
        )

    classified = classify_valid_trials(trials, lens_by_prompt)
    modal, novel = depth_groups(classified)
    comparison = compare_depths(modal, novel)

    num_layers = next(iter(lens_by_prompt.values()))["num_layers"]
    comparison["num_layers"] = num_layers
    comparison["run_dir"] = str(run_dir)

    with (run_dir / "depth_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, sort_keys=True)
        handle.write("\n")

    histogram_path = run_dir / "commitment_histogram.png"
    save_commitment_histogram(modal, novel, histogram_path, num_layers=num_layers)
    comparison["histogram_path"] = str(histogram_path)
    return comparison
