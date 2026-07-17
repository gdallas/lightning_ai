from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from lightning_decoding.config import config_hash, load_config
from lightning_decoding.metrics import summarize
from lightning_decoding.model_io import load_model
from lightning_decoding.noise import NoiseState
from lightning_decoding.runner import run_trials
from lightning_decoding.tasks import load_task


def knob_combinations(axis: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand a per-method knob grid into a list of concrete knob dicts."""
    names = list(axis)
    return [dict(zip(names, combo, strict=True)) for combo in itertools.product(*(axis[n] for n in names))]


def select_best(rows: list[dict[str, Any]], validity_floor: float) -> dict[str, Any]:
    """Pick the knob setting with the most distinct valid answers per prompt.

    Prefers settings whose validity clears ``validity_floor``; if none do, falls
    back to the whole pool. Ties break toward higher validity.
    """
    eligible = [row for row in rows if row["validity_rate"] >= validity_floor]
    pool = eligible or rows
    return max(pool, key=lambda row: (row["distinct_valid_per_prompt"], row["validity_rate"]))


def calibrate_experiment(
    config_path: str | Path,
    *,
    validity_floor: float = 0.9,
    trials_per_prompt: int | None = None,
    max_prompts: int | None = None,
    output_root: str | Path = "results",
    write_config: bool = False,
    local_files_only: bool = False,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    grid = cfg.get("calibration_grid")
    if not grid:
        raise SystemExit(f"{config_path} has no calibration_grid section")

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
    trials = int(trials_per_prompt or cfg["experiment"].get("trials_per_prompt", 20))

    all_rows: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}

    for method, axis in grid.items():
        method_rows: list[dict[str, Any]] = []
        for knobs in knob_combinations(axis):
            method_cfg = {"method": method, "global_seed": seed, **knobs}
            print(f"calibrating {method} {knobs}")
            records, _ = run_trials(
                model,
                tokenizer,
                model_name,
                task,
                prompts,
                method_cfg,
                seed=seed,
                trials=trials,
                progress=False,
            )
            summary = summarize(records, answer_space_sizes)
            row = {"method": method, "knobs": knobs, **summary}
            method_rows.append(row)
            all_rows.append(row)
        best = select_best(method_rows, validity_floor)
        selected[method] = best["knobs"]
        print(
            f"  best {method}: {best['knobs']} "
            f"validity={best['validity_rate']:.3f} distinct={best['distinct_valid_per_prompt']:.3f}"
        )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_hash": config_hash(cfg),
        "model": model_name,
        "task": task.name,
        "validity_floor": validity_floor,
        "trials_per_prompt": trials,
        "num_prompts": len(prompts),
        "selected": selected,
        "rows": all_rows,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_root) / f"{stamp}_calibrate_{config_hash(cfg)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "calibration.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    report["output_dir"] = str(out_dir)

    if write_config:
        write_calibrated_knobs(config_path, selected)

    return report


def sigma_settings(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a config's ``sigma_grid`` into ensemble method cfgs with fixed N/k.

    N and k come from the config's ``decoder`` block (defaults 10 and 3), so the sweep
    varies only sigma.
    """
    grid = cfg.get("sigma_grid")
    if not grid:
        raise SystemExit("config has no sigma_grid section")
    decoder = cfg.get("decoder", {})
    n = int(decoder.get("N", 10))
    k = int(decoder.get("k", 3))
    return [{"method": "ensemble_minority", "sigma": float(sigma), "N": n, "k": k} for sigma in grid]


def calibrate_sigma(
    config_path: str | Path,
    *,
    validity_floor: float = 0.9,
    trials_per_prompt: int | None = None,
    max_prompts: int | None = None,
    output_root: str | Path = "results",
    write_config: bool = False,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Sweep the ensemble ``sigma_grid`` and pick the sigma with the most distinct
    valid answers subject to a validity floor (default 0.90, per 2.B)."""
    cfg = load_config(config_path)
    settings = sigma_settings(cfg)

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
    trials = int(trials_per_prompt or cfg["experiment"].get("trials_per_prompt", 20))
    noise_state = NoiseState(model)

    rows: list[dict[str, Any]] = []
    for setting in settings:
        method_cfg = {**setting, "global_seed": seed, "_noise_state": noise_state}
        print(f"calibrating sigma={setting['sigma']}")
        records, _ = run_trials(
            model,
            tokenizer,
            model_name,
            task,
            prompts,
            method_cfg,
            seed=seed,
            trials=trials,
            progress=False,
        )
        summary = summarize(records, answer_space_sizes)
        rows.append({"knobs": {k: v for k, v in setting.items() if k != "method"}, **summary})

    best = select_best(rows, validity_floor)
    selected = {"ensemble_minority": best["knobs"]}
    print(
        f"  best sigma: {best['knobs']} "
        f"validity={best['validity_rate']:.3f} distinct={best['distinct_valid_per_prompt']:.3f}"
    )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_hash": config_hash(cfg),
        "model": model_name,
        "task": task.name,
        "validity_floor": validity_floor,
        "trials_per_prompt": trials,
        "num_prompts": len(prompts),
        "selected": selected,
        "rows": rows,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_root) / f"{stamp}_calibrate_sigma_{config_hash(cfg)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "sigma_calibration.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    report["output_dir"] = str(out_dir)

    if write_config:
        write_calibrated_knobs(config_path, selected)

    return report


def write_calibrated_knobs(config_path: str | Path, selected: dict[str, Any]) -> None:
    """Store selected knobs under a ``calibrated`` key in the raw config file.

    Reads the un-resolved YAML so ``extends`` and other authored keys survive.
    """
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw["calibrated"] = selected
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)
