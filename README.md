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

On Windows, this repository also includes a setup helper:

```powershell
.\scripts\setup.ps1 -RecreateVenv
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
- `lens_per_prompt.jsonl` (only when `experiment.capture_hidden_states` is set)

## Baseline Comparison

Run every decoder listed in a config's `baselines` section over the same task, seed, and
trial count, then emit a comparison table and CI bar charts:

```bash
lightning-decoding compare-baselines configs/phase1_baselines.yaml
```

The run folder gets `comparison.csv` (validity, distinct-valid with bootstrap CIs,
coverage, mode-share per method) plus `comparison_distinct_valid.png` and
`comparison_validity.png`. Use `--max-prompts` / `--trials` for a fast reduced-scale run.

## Calibrating Decoder Knobs

Sweep the `calibration_grid` in a config to pick per-method knobs (temperature `T`,
nucleus `p`, gap-sampler `delta`). Each setting is scored by distinct valid answers per
prompt, keeping only settings whose validity clears `--validity-floor`:

```bash
lightning-decoding calibrate configs/phase1_baselines.yaml --write-config
```

The command writes a `calibration.json` report under `results/` and, with
`--write-config`, stores the selected knobs back under a `calibrated:` key in the config.
Use `--max-prompts` and `--trials` to run a fast reduced-scale sweep first.

## Commitment-Depth Study

Sanity-check the logit lens (per-layer predictions should sharpen to the model's own
next-token prediction):

```bash
lightning-decoding lens-check
```

Run a decoder with `experiment.capture_hidden_states: true` (see `configs/phase4_lens.yaml`),
then analyze how deep in the network valid tokens are decided. Answers matching the clean
greedy token are labelled `modal`; valid alternatives are `novel`:

```bash
lightning-decoding run configs/phase4_lens.yaml
lightning-decoding analyze-depth results/<run-folder>
```

`analyze-depth` writes `depth_comparison.json` (a Mann-Whitney U test of modal vs novel
commitment depth) and `commitment_histogram.png` into the run folder.

For a perturbation-ensemble run, plot how far below the clean top logit the
minority-selected tokens sat:

```bash
lightning-decoding run configs/phase2_ensemble.yaml
lightning-decoding gap-histogram results/<run-folder>
```

This writes `clean_gap_histogram.png` + `gap_stats.json` for the non-fallback ensemble
selections.

## Prompt Calibration

The category prompt template is tuned by greedy validation. On Pythia-160m the original
`"Q: Name one {category}.\nA: One {category} is the"` scores **0.00** greedy validity
(the model continues "...is the *same*"). The default is now
`"The most common {category} is the"`, which reaches **0.50** greedy validity over the
20 base categories. Note that 0.50 is below the 0.80 Phase 1 gate: greedy validity at
this bar is a model-capacity limitation of the 160m model, addressed by scaling up in
later phases.

## Phase 0 Calibration

Initial Pythia-160m smoke test on this machine:

```powershell
.\.venv\Scripts\python.exe -m lightning_decoding.cli smoke --model EleutherAI/pythia-160m --local-files-only --prompt "The capital of France is" --max-new-tokens 20
```

Output contained:

```text
The capital of France is located in the city of Paris.
```

Single forward-pass benchmark:

```powershell
.\.venv\Scripts\python.exe -m lightning_decoding.cli benchmark-forward --model EleutherAI/pythia-160m --local-files-only --runs 20 --prompt "Q: Name one animal that commonly appears in children's books. A: One animal is the"
```

Measured result:

```text
prompt_tokens=19
runs=20
avg_forward_ms=841.88
min_forward_ms=300.88
max_forward_ms=1446.42
torch_threads=2
```

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
