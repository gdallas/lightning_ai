from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from lightning_decoding.calibrate import calibrate_experiment
from lightning_decoding.config import load_config
from lightning_decoding.model_io import load_model, load_tokenizer
from lightning_decoding.runner import run_experiment
from lightning_decoding.token_spaces import filter_category_file


def cmd_smoke(args: argparse.Namespace) -> None:
    model, tokenizer = load_model(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    prompt = args.prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    elapsed = time.perf_counter() - start
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
    print(f"elapsed_s={elapsed:.3f}")
    print(f"torch_threads={torch.get_num_threads()}")


def cmd_benchmark_forward(args: argparse.Namespace) -> None:
    model, tokenizer = load_model(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    inputs = tokenizer(args.prompt, return_tensors="pt")

    with torch.no_grad():
        model(**inputs)

    timings_ms: list[float] = []
    for _ in range(args.runs):
        start = time.perf_counter()
        with torch.no_grad():
            model(**inputs)
        timings_ms.append((time.perf_counter() - start) * 1000)

    avg_ms = sum(timings_ms) / len(timings_ms)
    print(f"model={args.model}")
    print(f"prompt_tokens={int(inputs['input_ids'].numel())}")
    print(f"runs={args.runs}")
    print(f"avg_forward_ms={avg_ms:.2f}")
    print(f"min_forward_ms={min(timings_ms):.2f}")
    print(f"max_forward_ms={max(timings_ms):.2f}")
    print(f"torch_threads={torch.get_num_threads()}")


def cmd_filter_token_space(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    model_name = cfg["model"]["name"]
    tokenizer = load_tokenizer(
        model_name,
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
        local_files_only=args.local_files_only,
    )
    task = cfg["task"]
    if task["type"] != "category":
        raise SystemExit("filter-token-space currently supports category tasks")

    output = args.output or f"data/processed/categories.{model_name.replace('/', '__')}.json"
    report = filter_category_file(task["path"], output, tokenizer)
    print(json.dumps({"output": output, "report": report}, indent=2, sort_keys=True))


def cmd_run(args: argparse.Namespace) -> None:
    output_dir = run_experiment(args.config)
    print(f"wrote {output_dir}")


def cmd_calibrate(args: argparse.Namespace) -> None:
    report = calibrate_experiment(
        args.config,
        validity_floor=args.validity_floor,
        trials_per_prompt=args.trials,
        max_prompts=args.max_prompts,
        write_config=args.write_config,
        local_files_only=args.local_files_only,
    )
    print(f"wrote {report['output_dir']}")
    print(json.dumps({"selected": report["selected"]}, indent=2, sort_keys=True))


def cmd_summarize(args: argparse.Namespace) -> None:
    path = Path(args.run_dir) / "summary.csv"
    print(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lightning-decoding")
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="Load a model and greedily generate a short completion.")
    smoke.add_argument("--model", default="EleutherAI/pythia-160m")
    smoke.add_argument("--prompt", default="The capital of France is")
    smoke.add_argument("--max-new-tokens", type=int, default=20)
    smoke.add_argument("--trust-remote-code", action="store_true")
    smoke.add_argument("--local-files-only", action="store_true")
    smoke.set_defaults(func=cmd_smoke)

    benchmark = sub.add_parser("benchmark-forward", help="Benchmark single forward-pass latency.")
    benchmark.add_argument("--model", default="EleutherAI/pythia-160m")
    benchmark.add_argument(
        "--prompt",
        default="Q: Name one animal.\nA: One animal is the",
    )
    benchmark.add_argument("--runs", type=int, default=10)
    benchmark.add_argument("--trust-remote-code", action="store_true")
    benchmark.add_argument("--local-files-only", action="store_true")
    benchmark.set_defaults(func=cmd_benchmark_forward)

    filter_space = sub.add_parser("filter-token-space", help="Build single-token category spaces.")
    filter_space.add_argument("config")
    filter_space.add_argument("--output")
    filter_space.add_argument("--local-files-only", action="store_true")
    filter_space.set_defaults(func=cmd_filter_token_space)

    run = sub.add_parser("run", help="Run one YAML-configured experiment.")
    run.add_argument("config")
    run.set_defaults(func=cmd_run)

    calibrate = sub.add_parser("calibrate", help="Sweep decoder knobs from a config's calibration_grid.")
    calibrate.add_argument("config")
    calibrate.add_argument("--validity-floor", type=float, default=0.9)
    calibrate.add_argument("--trials", type=int, default=None, help="Override trials_per_prompt.")
    calibrate.add_argument("--max-prompts", type=int, default=None, help="Limit prompts for a fast sweep.")
    calibrate.add_argument("--write-config", action="store_true", help="Store selected knobs back into the config.")
    calibrate.add_argument("--local-files-only", action="store_true")
    calibrate.set_defaults(func=cmd_calibrate)

    summarize = sub.add_parser("summarize", help="Print a run summary.csv.")
    summarize.add_argument("run_dir")
    summarize.set_defaults(func=cmd_summarize)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
