# Progress

Last updated: after Phase 1 harness work (prompt calibration, hidden-state capture, knob calibration).

## Current State

The repository has been scaffolded, committed, and pushed to the public GitHub repository:

- Repository: `gdallas/lightning_ai`
- Branch: `main`
- Latest pushed commit at the time of this note: `51a96fe`

Phase 0 is verified. Phase 1 harness is now complete enough to run baselines and
calibrate knobs; the remaining Phase 1 gaps are the full baseline table (1.B/1.C),
wordlist expansion (1.3/1.4), and the greedy validity gate (1.A), which does not
pass at the Pythia-160m scale (see below).

## This Session

- Implemented `CaptureHiddenStates(model)`, a forward-hook context manager, plus a
  `transformer_layers()` resolver (GPTNeoX / Llama-Qwen / GPT-2 style) in `lens.py`. [1.6]
- Added a pure `commitment_depth_from_argmaxes()` helper and `lens_argmax_per_layer()`. [4.3]
- Refactored the runner into a reusable `run_trials()` helper. When
  `experiment.capture_hidden_states` is set, each run writes `lens_per_prompt.jsonl`
  (raw per-layer lens argmax) and adds `commitment_depth` to every trial row. [4.3]
- Implemented `calibrate.py` + a `lightning-decoding calibrate` CLI command that
  sweeps `calibration_grid` for temperature/nucleus/gap_sampler, selects the best knob
  by `distinct_valid_per_prompt` subject to a validity floor, and writes
  `calibration.json`. `--write-config` stores selected knobs back under a `calibrated`
  key. [1.21, mechanism for 1.22]
- Added `model.local_files_only` support to the experiment runner (config-driven).
- Added fast unit tests: `test_lens.py`, `test_calibrate.py`. Suite is now 18 passing.
- Tuned the category prompt template via a 24-template greedy sweep on Pythia-160m. [1.5]

## Key Findings

- **Prompt template.** The original template `"Q: Name one {category}.\nA: One {category}
  is the"` yields **0.00** greedy validity on Pythia-160m (the model continues "...is
  the *same*"). The tuned template `"The most common {category} is the"` reaches
  **0.50** greedy validity over the 20 base categories and is now the default in
  `configs/base.yaml` and `CategoryTask`. Zero-shot alternatives topped out at 0.15 and
  5-shot at 0.20; the "most common ... is the" stem was the clear winner.
- **Greedy validity gate (1.A) is not met at 160m.** Best achievable greedy validity is
  ~0.50, well under the 0.80 bar. This is a model-capacity limit, not a harness bug;
  reaching 0.80 likely requires a larger model (Phase 6 scales to 410m/1b) or a relaxed
  acceptance criterion. Recorded honestly rather than checked off.
- Verified the lens pipeline end-to-end: 12 layers captured for Pythia-160m, per-layer
  lens argmax recorded, `commitment_depth` computed per trial.
- Verified the calibrate pipeline end-to-end at a small scale (3 prompts x 2 trials);
  full-scale calibration remains a heavier run to be kicked off by the user.

No experiment folders are committed; `results/` is gitignored (contains only `.gitkeep`).

## Completed

- Created installable Python package under `src/lightning_decoding/`.
- Added public repo files: `README.md`, `LICENSE`, `.gitignore`, `.gitattributes`, `pyproject.toml`.
- Added starter configs:
  - `configs/base.yaml`
  - `configs/phase1_baselines.yaml`
  - `configs/phase2_ensemble.yaml`
  - `configs/phase4_lens.yaml`
- Added offline task data:
  - `data/categories.json`
  - `data/rhymes.json`
- Added core modules:
  - `config.py`
  - `model_io.py`
  - `tasks.py`
  - `token_spaces.py`
  - `decoders.py`
  - `noise.py`
  - `metrics.py`
  - `runner.py`
  - `lens.py`
  - `plotting.py`
  - `cli.py`
- Added CLI commands:
  - `smoke`
  - `benchmark-forward`
  - `filter-token-space`
  - `run`
  - `summarize`
- Added fast unit tests for tasks, decoders, noise, and metrics.
- Added `scripts/check.ps1`.
- Added `scripts/setup.ps1` to bootstrap a real Python venv, install dependencies, run tests, and optionally run model smoke/experiment commands.
- Added `BACKLOG.md` with project-plan tickets.
- Added this progress log.
- Connected local git repo to `https://github.com/gdallas/lightning_ai.git`.
- Preserved the GitHub-created GPLv3 license and updated package metadata to match `GPL-3.0-only`.
- Pushed the scaffold to `main`.
- Verified Python 3.12.10 venv exists and can run the project.
- Ran fast tests successfully: `7 passed in 21.24s`.
- Fixed normalization so singular words ending in `-is` or `-us`, such as `Paris`, are not over-stripped.
- Updated Torch thread setup to happen once at `model_io` import.
- Added cached/local model loading flags to avoid unnecessary Hugging Face network checks.
- Verified Pythia-160m smoke output includes `Paris`.
- Benchmarked a 19-token Pythia-160m forward pass: average 841.88 ms over 20 runs, with 2 Torch threads.

## Blocked Or Not Yet Verified

- Full Phase 1 baseline experiments (1.B/1.C) have not been run at full scale yet.
- Greedy Task A validity is measured (0.50) but does **not** meet the 0.80 gate (1.A) at 160m.
- Full-scale knob calibration has not been run; only a 3-prompt x 2-trial verification pass.
- Ensemble sigma calibration (2.10) is still not implemented (temperature/nucleus/gap are).
- Category wordlists are still ~20 entries each (1.3/1.4 not started).

## Next Recommended Steps

1. Run a full knob calibration and store the results:

   ```powershell
   .\.venv\Scripts\python.exe -m lightning_decoding.cli calibrate configs\phase1_baselines.yaml --write-config --local-files-only
   ```

   Then confirm the `calibrated:` block written into `configs/phase1_baselines.yaml` (1.22).
2. Produce the full baseline table (1.B) and check the gap-sampler-beats-greedy claim (1.C)
   by running each method config at full trial counts and comparing `summary.csv` files.
3. Decide how to handle the 1.A gate: either scale up the model (Phase 6, 410m/1b) or
   relax the acceptance criterion, since 0.80 greedy validity is unreachable at 160m.
4. Expand category wordlists to 50-300 accepted answers and spot-check them (1.3/1.4).
5. Add ensemble sigma calibration (2.10) to the calibration grid/runner.
