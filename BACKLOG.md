# Backlog

This backlog is derived from `lightning_decoding_project_plan.md`. Checkboxes here are implementation tickets; acceptance criteria remain binding.

## Phase 0 - Environment Setup

- [x] 0.1 Create a virtual environment with Python 3.11+.
- [x] 0.2 Install CPU PyTorch, `transformers`, `pyyaml`, `pandas`, `matplotlib`, `pytest`, and `pronouncing`.
- [x] 0.3 Implement `load_model(name)` with float32 CPU loading and `model.eval()`.
- [x] 0.3a Set torch thread count once using physical-core-oriented defaults.
- [x] 0.4 Smoke test Pythia-160m on `"The capital of France is"` for 20 generated tokens.
- [x] 0.A Verify smoke output contains `"Paris"`.
- [x] 0.B Benchmark a single forward pass on a ~20-token prompt under ~1 second and record timing in `README.md`.

## Phase 1 - Harness, Tasks, Baselines, Null Hypothesis

- [x] 1.1 Implement category task loading and validity checking.
- [x] 1.2 Implement rhyme task loading and `pronouncing` validity checking.
- [ ] 1.3 Expand category wordlists to 50-300 accepted answers per category.
- [ ] 1.4 Manually spot-check about 10 entries per category wordlist.
- [x] 1.5 Tune and record the final category prompt template after greedy validation.
- [x] 1.6 Implement `CaptureHiddenStates(model)` hook context manager.
- [x] 1.7 Implement greedy decoder.
- [x] 1.8 Implement temperature decoder.
- [x] 1.9 Implement nucleus decoder.
- [x] 1.10 Implement logit-gap sampler.
- [x] 1.11 Implement single-token category answer-space filtering.
- [x] 1.12 Implement `validity_rate`.
- [x] 1.13 Implement `distinct_valid_per_prompt`.
- [x] 1.14 Implement `coverage`.
- [x] 1.15 Implement `mode_share`.
- [x] 1.16 Implement bootstrap 95% CI over prompts.
- [x] 1.17 Implement YAML-driven experiment runner.
- [x] 1.18 Write one JSONL row per trial.
- [x] 1.19 Write summary CSV per experiment.
- [x] 1.20 Print console progress every 50 trials.
- [x] 1.21 Implement knob calibration runner for temperature, nucleus, and gap sampler.
- [ ] 1.22 Store calibrated knobs in `configs/phase1_baselines.yaml`. (Mechanism ready: `calibrate --write-config`; awaiting a full-scale sweep to fill real values.)
- [ ] 1.A Verify greedy reaches at least 80% validity on Task A. (Measured 0.50 on Pythia-160m with the tuned template; the 80% bar is not reachable at this scale — see PROGRESS.md.)
- [ ] 1.B Produce full baseline table for 4 methods x 2 tasks x metrics with CIs.
- [ ] 1.C Verify gap sampler beats greedy on `distinct_valid_per_prompt`.
- [x] 1.D Run and pass unit tests.

## Phase 2 - Perturbation Ensemble

- [x] 2.1 Implement attention/MLP-only default noise filter for Pythia and Qwen-style names.
- [x] 2.2 Implement deterministic `stable_hash()` with `zlib.crc32`.
- [x] 2.3 Cache clean per-tensor standard deviations once.
- [x] 2.4 Apply perturbation ops under `torch.no_grad()`.
- [x] 2.5 Implement `perturb_()`.
- [x] 2.6 Implement `unperturb_()`.
- [x] 2.7 Add noise round-trip unit test.
- [x] 2.8 Implement `ensemble_minority`.
- [x] 2.9 Record ensemble counts, fallback flag, per-run latency, and clean logit gap.
- [ ] 2.10 Calibrate ensemble sigma.
- [ ] 2.11 Run ensemble vs calibrated baselines with same tasks, seeds, and trials.
- [ ] 2.12 Produce comparison table and bar chart.
- [ ] 2.13 Plot clean-pass logit-gap histogram for minority-selected tokens.
- [x] 2.A Verify noise round-trip test passes in the real venv.
- [ ] 2.B Verify ensemble validity is at least 0.90 at calibrated sigma.
- [ ] 2.C Verify comparison table and gap histogram exist.
- [ ] 2.D Verify full Phase 2 runtime is within budget or reduce R and note it.

## Phase 3 - Verdict and Ablations

- [ ] 3.1 Apply H0/H1 decision rule after Phase 2.
- [ ] 3.2 If H0 wins, write the negative result and proceed to Phase 4.
- [ ] 3.3 If H1 survives, run sigma sweep and plot novelty/validity vs sigma.
- [ ] 3.4 If H1 survives, test layer restrictions `[0-3]`, `[4-7]`, `[8-11]`, and all.
- [ ] 3.5 If H1 survives, ablate `N` in `{5, 10, 20}` with `k` in `{2, 3, 6}`.
- [ ] 3.6 If H1 survives, compare attention-only vs MLP-only perturbation.
- [ ] 3.7 If H1 survives, replicate best configuration on Qwen2.5-0.5B-Instruct.

## Phase 4 - Commitment-Depth Study

- [x] 4.1 Implement Pythia/Qwen logit-lens projection helper.
- [x] 4.2 Implement commitment-depth helper.
- [x] 4.3 Add raw per-layer lens argmax recording.
- [ ] 4.4 Run calibrated nucleus decoder for R=50 trials per Task A prompt with hidden-state capture. (Pipeline verified at reduced scale: nucleus p=0.95, R=15; full calibrated R=50 run pending.)
- [x] 4.5 Label valid produced tokens as modal or novel.
- [x] 4.6 Compare mean commitment depth with Mann-Whitney U test.
- [x] 4.7 Plot layerwise commitment histograms for modal vs novel tokens.
- [x] 4.A Verify lens sanity check on `"The capital of France is"`. (Lens sharpens to " Paris" with the "...is the city of" continuation and the final-layer lens argmax matches the model head.)
- [x] 4.B Verify depth comparison plot and test statistic exist. (Reduced-scale run: U=341, p=2.8e-10, novel commits ~1.5 layers later; regenerate for headline numbers after full calibration.)

## Phase 5 - Writeup

- [ ] 5.1 Draft motivation, H0/H1 framing, methods, calibration protocol, results, histogram, ablations or negative result, commitment-depth study, and limitations.
- [ ] 5.2 If H1 survives, frame abstract around rollout/candidate diversity at matched validity.
- [ ] 5.3 If H0 wins, publish negative result centered on the gap histogram.

## Phase 6 - Scale Replication

- [ ] 6.1 Repeat best configuration on Pythia 410m.
- [ ] 6.2 Repeat best configuration on Pythia 1b.
- [ ] 6.3 Plot effect size vs model size.
- [ ] 6.A Verify effect does not shrink toward zero.

## Phase 7 - Distill Ensemble Into A Probe

- [ ] 7.1 Build dataset for about 5k diverse prompts.
- [ ] 7.2 Record clean-pass hidden states around two-thirds depth.
- [ ] 7.3 Label top-20 candidates as robust or not under perturbation ensemble.
- [ ] 7.4 Train logistic-regression probe.
- [ ] 7.5 Evaluate AUROC.
- [ ] 7.A Verify AUROC is at least 0.8 before treating distillation as a headline result.

## Phase 8 - Branching Sequence Decoder

- [ ] 8.1 Implement multi-token branching decoder with max width 4 and max 3 branch points.
- [ ] 8.2 Evaluate on multi-token task variants with 2-4 token answers.
- [ ] 8.3 Compare distinct valid sequences against diverse beam search and nucleus sampling.

## Global Conventions

- [x] G1 Record config hash and git commit in experiment manifests/JSONL metadata.
- [ ] G2 Require `pytest` to pass before treating any experiment as valid.
- [x] G3 Restore model weights in `finally` after perturbation.
- [x] G4 Log wall-clock latency per trial.
- [ ] G5 Abort and re-plan if projected experiment time exceeds 8 hours.
