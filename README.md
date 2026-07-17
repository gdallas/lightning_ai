# Lightning Decoding

Robust-minority decoding under weight perturbation, plus a logit-lens commitment-depth study.

The project is designed to be cloned and run locally as a small Python research package.
Fast tests do not download model weights; model-backed commands use Hugging Face models on CPU.

## Quick Start

Install Python 3.11 or 3.12 from [python.org](https://www.python.org/downloads/) first.
On Windows, avoid the Microsoft Store shim for this project because virtual
environments created from that shim can fail later when running `pip` or `pytest`.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
lightning-decoding smoke --model EleutherAI/pythia-160m
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## First Experiment

Filter category wordlists down to answers that are a single token for the configured tokenizer:

```bash
lightning-decoding filter-token-space configs/base.yaml
```

Run a small baseline experiment:

```bash
lightning-decoding run configs/phase1_baselines.yaml
```

Results are written under `results/` as reproducible run folders containing:

- `resolved_config.yaml`
- `manifest.json`
- `trials.jsonl`
- `summary.csv`

## Repository Layout

```text
configs/                  YAML experiment configs
data/                     Offline task data and generated token spaces
docs/                     Project notes
src/lightning_decoding/   Installable Python package
tests/                    Fast unit tests
results/                  Gitignored experiment outputs
```

## Development Notes

- Keep all experiment behavior config-driven.
- Run `pytest` before trusting an experiment.
- Never perturb model weights without restoring them in a `finally` block.
- Generated results, model caches, and local virtual environments are intentionally not committed.
