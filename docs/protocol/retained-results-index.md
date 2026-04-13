# Retained Results Index

This document lists the experiment output folders intentionally retained in the repository after cleanup.

## Phase 1 discriminative results

### `outputs/phase1_midrun/`

Primary subject-held-out EEGNet result set for the current phase-1 CPP-like classification pipeline.

Contents include:

- fold-wise metrics
- summary metrics
- per-subject metrics
- summary plots
- fold-level ROI ERP comparison plots

### `outputs/phase1_baseline_midrun/`

Unscaled logistic baseline outputs for the hand-crafted CPP feature baseline.

### `outputs/phase1_baseline_midrun_scaled/`

Scaled logistic baseline outputs. These are the retained baseline outputs most appropriate for direct comparison with EEGNet.

## Phase 2 exploratory results

### `outputs/phase2_quickcheck/`

Retained quick-check outputs for the `cue_dimensionality` external-label experiment. These provide the current near-chance evidence discussed in the stage summary.

## Conditional EEG generation results

### `outputs/phase1_conditional_vae_auxquick/`

Retained quick-run conditional VAE training outputs, including fold-level generated sample arrays and loss summaries.

### `outputs/phase1_conditional_vae_auxquick_analysis/`

Retained analysis outputs comparing generated and real ROI ERP waveforms for the conditional VAE.

## Notes

- Additional smoke-test folders were removed because they were redundant once stronger retained runs existed.
- The retained folders above are the current evidence base for the stage summary in `docs/protocol/current-stage-summary.md`.
