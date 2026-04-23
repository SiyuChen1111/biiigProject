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

## ROI-only conditional generator (canonical contract, implemented)

ROI-only conditional generation is now the canonical next-step workflow for improving waveform-level plausibility in the CPP ROI.

Dedicated entry points:

- `scripts/train_roi_conditional_generator.py`
- `scripts/analyze_roi_conditional_generator.py`

This workflow uses a single, stable output contract so that smoke runs and retained evidence runs produce identical directory structures and predictable filenames.

Canonical output roots (do not scatter ROI artifacts into unrelated `phase1_*` folders):

- `outputs/roi_conditional_generator_smoke/` (smoke root, currently present)
- `outputs/roi_conditional_generator_evidence/` (evidence root for retained runs)

Retention policy for these canonical roots:

- retain manifests, JSON/CSV summaries, publication-style figures, fold ERP comparison plots, and the small per-fold `conditional_samples.npy` arrays that are part of the canonical contract
- do **not** retain intermediate morphology-loss verification roots such as `outputs/roi_conditional_generator_cpp_loss_*`; these are local scratch verification runs and are ignored by `.gitignore`
- do **not** retain ad-hoc checkpoints or cache directories inside `outputs/`

The canonical layout and filenames are specified in:

- `docs/protocol/roi-conditional-generator-output-contract.md`

## Notes

- Additional smoke-test folders were removed because they were redundant once stronger retained runs existed.
- The retained folders above are the current evidence base for the stage summary in `docs/protocol/current-stage-summary.md`.
