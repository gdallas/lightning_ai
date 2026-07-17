from __future__ import annotations

import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from lightning_decoding.config import config_hash, load_config, write_yaml
from lightning_decoding.decoders import decode
from lightning_decoding.metrics import summarize
from lightning_decoding.model_io import format_prompt, load_model
from lightning_decoding.noise import NoiseState
from lightning_decoding.tasks import load_task


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


def run_experiment(config_path: str | Path) -> Path:
    cfg = load_config(config_path)
    output_dir = run_dir_for(cfg, cfg.get("output_root", "results"))
    output_dir.mkdir(parents=True, exist_ok=False)

    write_yaml(output_dir / "resolved_config.yaml", cfg)

    model_name = cfg["model"]["name"]
    model, tokenizer = load_model(model_name, trust_remote_code=cfg["model"].get("trust_remote_code", False))
    task = load_task(cfg["task"])
    prompts = task.prompts()
    method_cfg = dict(cfg["decoder"])
    method_cfg["global_seed"] = int(cfg.get("seed", 0))
    if method_cfg["method"] == "ensemble_minority":
        method_cfg["_noise_state"] = NoiseState(model)

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

    records: list[dict[str, Any]] = []
    trials_path = output_dir / "trials.jsonl"
    r = int(cfg["experiment"].get("trials_per_prompt", 20))
    seed = int(cfg.get("seed", 0))

    with trials_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"__meta__": manifest}, sort_keys=True) + "\n")
        trial_idx = 0
        for prompt in prompts:
            prompt_text = format_prompt(tokenizer, model_name, prompt["prompt_text"])
            for run_idx in range(r):
                run_seed = seed * 1000 + trial_idx
                rng = torch.Generator().manual_seed(run_seed)
                start = time.perf_counter()
                result = decode(model, tokenizer, prompt_text, method_cfg, rng)
                latency_ms = (time.perf_counter() - start) * 1000
                normalized = task.normalize(result.token_str)
                row = {
                    "prompt_id": prompt["prompt_id"],
                    "key": prompt["key"],
                    "method": method_cfg["method"],
                    "knobs": {k: v for k, v in method_cfg.items() if not k.startswith("_")},
                    "run_seed": run_seed,
                    "token_id": result.token_id,
                    "token_str": result.token_str,
                    "normalized": normalized,
                    "valid": task.is_valid(prompt["key"], result.token_str),
                    "latency_ms": latency_ms,
                    "decoder_meta": result.meta,
                }
                records.append(row)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                trial_idx += 1
                if trial_idx % 50 == 0:
                    print(f"completed {trial_idx} trials")

    answer_space_sizes = {prompt["prompt_id"]: task.answer_space_size(prompt["key"]) for prompt in prompts}
    summary = summarize(records, answer_space_sizes)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    return output_dir

