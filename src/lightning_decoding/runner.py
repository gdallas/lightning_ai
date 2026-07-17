from __future__ import annotations

import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

from lightning_decoding.config import config_hash, load_config, write_yaml
from lightning_decoding.decoders import decode, last_logits
from lightning_decoding.lens import (
    CaptureHiddenStates,
    commitment_depth_from_argmaxes,
    lens_argmax_per_layer,
)
from lightning_decoding.metrics import summarize
from lightning_decoding.model_io import format_prompt, load_model
from lightning_decoding.noise import NoiseState
from lightning_decoding.tasks import Task, load_task


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def run_dir_for(cfg: dict[str, Any], root: str | Path = "results") -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = cfg.get("name", "experiment")
    return Path(root) / f"{stamp}_{name}_{config_hash(cfg)}"


def build_method_cfg(cfg: dict[str, Any], model: Any) -> dict[str, Any]:
    method_cfg = dict(cfg["decoder"])
    method_cfg["global_seed"] = int(cfg.get("seed", 0))
    if method_cfg["method"] == "ensemble_minority":
        method_cfg["_noise_state"] = NoiseState(model)
    return method_cfg


def run_trials(
    model: Any,
    tokenizer: Any,
    model_name: str,
    task: Task,
    prompts: list[dict[str, str]],
    method_cfg: dict[str, Any],
    *,
    seed: int,
    trials: int,
    capture_hidden_states: bool = False,
    row_sink: Callable[[dict[str, Any]], None] | None = None,
    progress: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run ``trials`` decodes per prompt and return (trial rows, per-prompt lens rows).

    Seeding matches a single global counter so a given (config, prompt, trial)
    is reproducible regardless of how the loop is sliced.
    """
    records: list[dict[str, Any]] = []
    lens_records: list[dict[str, Any]] = []
    trial_idx = 0
    for prompt in prompts:
        prompt_text = format_prompt(tokenizer, model_name, prompt["prompt_text"])

        lens_argmaxes: list[int] | None = None
        if capture_hidden_states:
            with CaptureHiddenStates(model) as capture:
                last_logits(model, tokenizer, prompt_text)
            lens_argmaxes = lens_argmax_per_layer(model, capture.hidden_states)
            lens_records.append(
                {
                    "prompt_id": prompt["prompt_id"],
                    "key": prompt["key"],
                    "num_layers": len(lens_argmaxes),
                    "lens_argmax_per_layer": lens_argmaxes,
                }
            )

        for _ in range(trials):
            run_seed = seed * 1000 + trial_idx
            rng = torch.Generator().manual_seed(run_seed)
            start = time.perf_counter()
            result = decode(model, tokenizer, prompt_text, method_cfg, rng)
            latency_ms = (time.perf_counter() - start) * 1000
            row = {
                "prompt_id": prompt["prompt_id"],
                "key": prompt["key"],
                "method": method_cfg["method"],
                "knobs": {k: v for k, v in method_cfg.items() if not k.startswith("_")},
                "run_seed": run_seed,
                "token_id": result.token_id,
                "token_str": result.token_str,
                "normalized": task.normalize(result.token_str),
                "valid": task.is_valid(prompt["key"], result.token_str),
                "latency_ms": latency_ms,
                "decoder_meta": result.meta,
            }
            if lens_argmaxes is not None:
                row["commitment_depth"] = commitment_depth_from_argmaxes(
                    lens_argmaxes, result.token_id
                )
            records.append(row)
            if row_sink is not None:
                row_sink(row)
            trial_idx += 1
            if progress and trial_idx % 50 == 0:
                print(f"completed {trial_idx} trials")

    return records, lens_records


def run_experiment(config_path: str | Path) -> Path:
    cfg = load_config(config_path)
    output_dir = run_dir_for(cfg, cfg.get("output_root", "results"))
    output_dir.mkdir(parents=True, exist_ok=False)

    write_yaml(output_dir / "resolved_config.yaml", cfg)

    model_name = cfg["model"]["name"]
    model, tokenizer = load_model(
        model_name,
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
        local_files_only=cfg["model"].get("local_files_only", False),
    )
    task = load_task(cfg["task"])
    prompts = task.prompts()
    method_cfg = build_method_cfg(cfg, model)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash(cfg),
        "git_commit": current_git_commit(),
        "model": model_name,
        "task": task.name,
        "decoder": method_cfg["method"],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    trials_path = output_dir / "trials.jsonl"
    r = int(cfg["experiment"].get("trials_per_prompt", 20))
    seed = int(cfg.get("seed", 0))
    capture = bool(cfg["experiment"].get("capture_hidden_states", False))

    with trials_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"__meta__": manifest}, sort_keys=True) + "\n")
        records, lens_records = run_trials(
            model,
            tokenizer,
            model_name,
            task,
            prompts,
            method_cfg,
            seed=seed,
            trials=r,
            capture_hidden_states=capture,
            row_sink=lambda row: handle.write(json.dumps(row, sort_keys=True) + "\n"),
        )

    if lens_records:
        with (output_dir / "lens_per_prompt.jsonl").open("w", encoding="utf-8") as handle:
            for record in lens_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    answer_space_sizes = {prompt["prompt_id"]: task.answer_space_size(prompt["key"]) for prompt in prompts}
    summary = summarize(records, answer_space_sizes)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    return output_dir
