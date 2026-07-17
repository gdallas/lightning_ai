# Architecture

Lightning Decoding is split into three layers:

1. Core library modules under `src/lightning_decoding/`.
2. Experiment orchestration through config files and the CLI.
3. Reproducible result artifacts under `results/`.

The core modules are intentionally small and testable:

- `tasks.py`: offline task definitions and validity checks.
- `token_spaces.py`: model-specific single-token answer filtering.
- `decoders.py`: greedy, sampling, gap, and perturbation-ensemble decoders.
- `noise.py`: deterministic in-place weight perturbation and restoration.
- `metrics.py`: validity, diversity, coverage, mode-share, and bootstrap CIs.
- `lens.py`: logit-lens projection and commitment-depth helpers.

The runner writes a complete run folder for every experiment so results can be
audited without reconstructing command-line state from memory.

