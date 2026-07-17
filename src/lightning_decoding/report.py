from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lightning_decoding.config import config_hash, load_config
from lightning_decoding.metrics import summarize
from lightning_decoding.model_io import load_model
from lightning_decoding.noise import NoiseState
from lightning_decoding.runner import run_trials
from lightning_decoding.tasks import load_task

TABLE_METRICS = [
    "validity_rate",
    "distinct_valid_per_prompt",
    "distinct_valid_ci_low",
    "distinct_valid_ci_high",
    "coverage",
    "mode_share",
    "trials",
]


def compare_baselines(
    config_path: str | Path,
    *,
    trials: int | None = None,
    max_prompts: int | None = None,
    output_root: str | Path = "results",
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Run every decoder in a config's ``baselines`` list over one shared task.

    All methods see the same prompts, seed, and trial count, so their metrics are
    directly comparable. Writes ``comparison.csv`` plus per-metric bar charts.
    """
    cfg = load_config(config_path)
    baselines = cfg.get("baselines")
    if not baselines:
        raise SystemExit(f"{config_path} has no baselines section")

    model_name = cfg["model"]["name"]
    model, tokenizer = load_model(
        model_name,
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
        local_files_only=local_files_only,
    )
    task = load_task(cfg["task"])
    prompts = task.prompts()
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    answer_space_sizes = {p["prompt_id"]: task.answer_space_size(p["key"]) for p in prompts}

    seed = int(cfg.get("seed", 0))
    r = int(trials or cfg["experiment"].get("trials_per_prompt", 20))

    rows: list[dict[str, Any]] = []
    for spec in baselines:
        method_cfg = {k: v for k, v in spec.items() if k != "label"}
        label = spec.get("label", method_cfg["method"])
        method_cfg["global_seed"] = seed
        if method_cfg["method"] == "ensemble_minority":
            method_cfg["_noise_state"] = NoiseState(model)
        print(f"running baseline {label} ({method_cfg['method']})")
        records, _ = run_trials(
            model,
            tokenizer,
            model_name,
            task,
            prompts,
            method_cfg,
            seed=seed,
            trials=r,
            progress=False,
        )
        summary = summarize(records, answer_space_sizes)
        rows.append({"label": label, "method": method_cfg["method"], **summary})

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_root) / f"{stamp}_baselines_{config_hash(cfg)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = ["label", "method", *TABLE_METRICS]
    with (out_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    save_comparison_bar(
        rows,
        "distinct_valid_per_prompt",
        out_dir / "comparison_distinct_valid.png",
        ci_keys=("distinct_valid_ci_low", "distinct_valid_ci_high"),
    )
    save_comparison_bar(rows, "validity_rate", out_dir / "comparison_validity.png")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "model": model_name,
        "task": task.name,
        "trials_per_prompt": r,
        "num_prompts": len(prompts),
        "rows": rows,
        "output_dir": str(out_dir),
    }
    with (out_dir / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return report


def save_comparison_bar(
    rows: list[dict[str, Any]],
    metric: str,
    output_path: str | Path,
    *,
    ci_keys: tuple[str, str] | None = None,
) -> None:
    """Bar chart of ``metric`` per baseline, with optional asymmetric CI error bars."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    labels = [row["label"] for row in rows]
    values = [row[metric] for row in rows]

    yerr = None
    if ci_keys is not None and all(k in rows[0] for k in ci_keys):
        lo_key, hi_key = ci_keys
        lower = [max(0.0, row[metric] - row[lo_key]) for row in rows]
        upper = [max(0.0, row[hi_key] - row[metric]) for row in rows]
        yerr = [lower, upper]

    fig, ax = plt.subplots()
    ax.bar(labels, values, yerr=yerr, capsize=4)
    ax.set_ylabel(metric)
    ax.set_title(f"Baselines: {metric}")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
