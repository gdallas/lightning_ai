# Progress

Last updated: after the decoder parameter study and the H0 verdict (Phase 3.1).

## Verdict (Phase 3): H0 holds at 160M

The perturbation ensemble — the project's core hypothesis — **does not beat greedy** and is strictly
dominated by a trivial gap sampler, across every noise strength (σ 0.02–0.10) and every layer/component
scope (all, attention, mlp, early, mid, late, attention∩late), including a decisive best-shot test. It
surfaces generic high-prior tokens ("common", "most"), not valid alternatives. This is a **negative
result at 160M, not a refutation** — the fair next test is scale (Phase 6: 410M/1B). Full analysis,
examples, and reasoning: `docs/parameter_study.md` (§7, §8, §8b, Verdict). Calibrated decoder knobs
(gap δ=2, temperature T=0.7, nucleus p=0.95) are now stored in `configs/phase1_baselines.yaml`.

## Decoder knobs — parameter study (docs/parameter_study.md)

Swept every decoder's coefficient with per-word example outputs (category task, Pythia-160m):

- **Gap sampler δ=2** is the best decoder — 3.09 distinct valid answers/prompt (the most of anything
  tested), cleanest outputs; δ=1 barely leaves greedy, δ≥6 collapses to noise.
- **Temperature T=0.7**, not the 1.0 default — beats 1.0 on both validity (0.29 vs 0.15) and diversity
  (2.57 vs 1.92); T≥1.3 degrades into sub-word fragments.
- **Nucleus `p` is a near no-op** — outputs barely change from p=0.85 to 0.995 (the raw distribution is
  sharp, so widening the nucleus only admits tail tokens that are never sampled).
- Added a per-word `answers.md` side-by-side output to `compare-baselines`, and a configurable
  perturbation `noise_scope` (attention/MLP components + layer bands) that drove the §8 scope study.
- Test suite is now 38 passing.

## Current State

The repository has been scaffolded, committed, and pushed to the public GitHub repository:

- Repository: `gdallas/lightning_ai`
- Branch: `main` (scaffold was `51a96fe`; main has advanced well past it — see git log)

Phases 0-2 and 4 are done; the H0 verdict (Phase 3) is in. Remaining Phase 1 gaps are the
rhyme-task baseline (1.B), wordlist expansion (1.3/1.4), and the greedy validity gate (1.A),
which does not pass at the Pythia-160m scale (model-capacity limit).

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

## Full Commitment-Depth Study (4.4) — first real experiment result

Ran the real Phase 4 study: nucleus p=0.95, **R=50** over the 20 Task-A categories with
hidden-state capture (`configs/phase4_lens.yaml`), 1,000 trials total.

- **Nucleus diversity:** 136 valid trials (13.6% per-sample validity at p=0.95),
  **3.82 distinct valid answers/prompt** [2.92, 4.70].
- **Depth comparison:** modal (n=36) mean depth 10.6; novel (n=100) depth 12;
  Mann-Whitney U=3600, **p=1.5e-30**.
- **The substantive slice (added `visibility` to `analyze-depth`):** modal answers
  commit *only* at layers 10-11 of 12 (13 at L10, 23 at L11) — the model settles on its
  obvious answer in the last one or two layers. **Only 3 of 100 novel valid answers ever
  appear as the top lens prediction at any layer** (and those flash at L7 before being
  overwritten).

Interpretation: the plain commitment-depth gap is partly definitional (a novel token can
never be the sustained top prediction, so it always lands at `num_layers`). The real,
non-tautological finding is the visibility rate: **valid alternative answers are almost
invisible to the greedy/lens view (~3%)** — which is precisely the gap the perturbation
ensemble is meant to exploit. A richer per-layer metric (lens rank/probability trajectory,
not just argmax) is the natural follow-up if we want to grade *how close* novel tokens get.

## Ensemble Sigma Calibration (2.10)

- Added `calibrate_sigma` + `calibrate-sigma` CLI: sweeps the ensemble `sigma_grid`
  (fixed N/k from the config), selects sigma by distinct-valid subject to a validity
  floor (default 0.90, per 2.B), writes `sigma_calibration.json`, and can `--write-config`.
- Added a pure `sigma_settings` helper with unit tests; suite is now 31 passing.
- Verified end-to-end at reduced scale (all 5 sigmas swept, selection + report + CLI).
  Full-scale sweep for the real calibrated sigma remains a heavier run for the user.

## Phase 2 Ensemble Reporting (2.12/2.13)

- Added `gap-histogram` CLI + `analysis.minority_clean_gaps` / `save_gap_histogram`:
  reads an ensemble run, plots the clean-pass logit gap for minority (non-fallback)
  selections, and writes `clean_gap_histogram.png` + `gap_stats.json`. [2.13]
- Added a `baselines:` section (including the ensemble) to `configs/phase2_ensemble.yaml`
  so `compare-baselines` runs ensemble head-to-head with the baselines. [2.12]
- Extended `test_analysis.py`; suite is now 28 passing.
- Verified end-to-end: a sigma=0.08, N=10, k=2 run over 20 categories produced 16
  minority selections out of 20 (4 fallbacks), clean-pass gaps 0.04-3.93 (mean 1.33,
  median 1.16). `compare-baselines` on the phase2 config emits the ensemble-vs-baselines
  table + CI bar charts. Calibrated sigma (2.10) and full-scale runs (2.11/2.B) remain open.

## Baseline Comparison Harness (1.B/1.C)

- Implemented `report.py` + `compare-baselines` CLI: runs every decoder in a config's
  `baselines` list over one shared task (same prompts/seed/trials), writing
  `comparison.csv`, `comparison.json`, and CI bar charts. Added a `baselines:` section to
  `configs/phase1_baselines.yaml`.
- Forced the non-interactive matplotlib `Agg` backend in all plotting helpers so charts
  render headless (the machine defaulted to Tk, which failed on save).
- Added `test_report.py`; suite is now 26 passing.
- Reduced-scale run (category task, 8 prompts x 6 trials): validity / distinct-valid were
  greedy 0.63 / 1.00, temperature 0.10 / 1.25, nucleus 0.19 / 2.00, gap_sampler 0.33 /
  2.40 [1.67, 3.50]. Gap sampler beats greedy on distinct-valid with non-overlapping CIs
  (1.C). Full 4x2 table at full scale is still pending (1.B).

## Phase 4 Session (commitment-depth study)

- Added `lens-check` CLI: prints per-layer logit-lens predictions and asserts the
  final-layer lens argmax equals the model head. [4.A]
- Implemented `analysis.py` + `analyze-depth` CLI: labels valid trials modal vs novel,
  runs a Mann-Whitney U test on commitment depth, and writes `depth_comparison.json`
  plus `commitment_histogram.png`. [4.5, 4.6, 4.7]
- Added `test_analysis.py` (labeling, comparison, empty-group handling, histogram). Suite
  is now 24 passing.
- Ran the study end-to-end at reduced scale (nucleus p=0.95, R=15 over 20 categories with
  hidden-state capture) and produced real artifacts. [4.4 partial, 4.B]

### Phase 4 results (reduced scale — not yet the headline run)

- Lens sanity check: on `"The capital of France is the city of"` the lens sharpens to
  `" Paris"` by layer 8 and the final layer matches the model's greedy token.
- Commitment depth: modal tokens (n=11) mean depth 10.5; novel valid tokens (n=31) mean
  depth 12.0 (mostly never a sustained lens argmax). Mann-Whitney U=341, p=2.8e-10;
  novel tokens commit ~1.5 layers later. Consistent with H1, but at R=15 with an
  uncalibrated nucleus knob — rerun at R=50 with calibrated `p` for the headline claim.

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

- 1.B: the category baseline table is done; the **rhyme task (Task B)** baseline has not been run.
- 1.A: greedy validity 0.50 does **not** meet the 0.80 gate at 160M (model-capacity limit, not a bug).
- 2.B: ensemble validity 0.17-0.28 does **not** meet the 0.90 bar — part of the H0 verdict.
- Category wordlists are still ~20 entries each (1.3/1.4 not started).
- The ensemble numbers rest on reduced-scale runs (R=2-4, to fit a 10-min-per-run cap); the
  *directions* are firm and consistent across configs, but exact values are indicative.

## Next Recommended Steps

The core hypothesis is answered (H0) at 160M. Two live directions:

1. **Scale up (Phase 6):** repeat greedy / gap δ=2 / ensemble on Pythia-410M then 1B to test whether
   the perturbation effect appears *at all* at larger scale (needs model downloads + more compute);
   plot effect size vs model size.
2. **Write up the negative result (Phase 5):** center it on the commitment-depth study (valid
   alternatives ~3% visible to the late layers) + the gap histogram, with the decoder-knob
   recommendations as the practical takeaway.

Smaller loose ends: run the rhyme (Task B) baseline to finish the 4x2 table (1.B); expand category
wordlists (1.3/1.4); re-run the ensemble sweeps at higher R if a firmer negative is wanted.
