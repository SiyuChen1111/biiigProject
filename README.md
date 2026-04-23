# biiigProject

This repository now follows a **TIER-inspired transitional layout** for reproducible EEG analysis work. The current focus is a CPP-related EEG workflow on the Kosciessa `task-dynamic` dataset, with a completed phase-1 discriminative pipeline, exploratory phase-2 label tests, and an early conditional EEG generation branch.

## Project layout

- `README.md` — project entry point and navigation
- `docs/`
  - `notes/` — idea notes and early project framing
  - `protocol/` — implementation blueprints, architectural notes, and proposal-style documents
- `src/`
  - `data/` — dataset loading utilities
  - `preprocessing/` — epoch extraction and preprocessing
  - `features/` — CPP-related feature construction
  - `evaluation/` — metrics and plotting helpers
  - `models/` — reusable EEG model definitions
- `scripts/`
  - executable experiment entry points
- `2_Analysis/`
  - legacy notebook folder retained during the transition
- `outputs/`
  - retained experiment outputs and analysis artifacts
- `data/raw/`
  - optional preferred local raw dataset location if created by the user (ignored by git)

## Raw data

If present, the code prefers the Kosciessa dataset at:

- `data/raw/CPP_low-level-2_Kosciessa_et_al_2021/`

For backward compatibility during the current reorganization, the loaders also fall back to:

- `1_Data/CPP_low-level-2_Kosciessa_et_al_2021/`

The raw dataset is intentionally not versioned in git.

## Reproducibility metadata

The current Python dependency manifest is:

- `requirements.txt`

The expected working directory for all scripts is the **repository root**.

## Current project status

### Phase 1

Implemented and validated:

- response-locked epoch extraction
- CPP-like scoring in `CPz / CP1 / CP2`
- subject-held-out EEGNet training
- logistic sanity baseline

Interpretation:

- the hand-crafted CPP-like score is highly stable across subjects
- EEGNet can recover that distinction, but does not outperform simple hand-crafted features

### Phase 2

Implemented exploratory scripts for external task labels:

- `cue_dimensionality` extremes
- RT fast vs slow

Current quick-check results suggest neither of these labels yet provides a strong cross-subject EEG decoding target under the present setup.

### Conditional generation

Implemented and retained:

- `scripts/train_phase1_conditional_vae.py`
- `scripts/analyze_conditional_vae_samples.py`
- `src/models/conditional_eeg_vae.py`

### ROI-only conditional generation (current canonical generator path)

Implemented and retained:

- `scripts/train_roi_conditional_generator.py`
- `scripts/analyze_roi_conditional_generator.py`
- `scripts/plot_cpp_style_roi_figure.py`
- `src/models/cpp_losses.py`

Current interpretation:

- this is now the **canonical** conditional-generation workflow for CPP-related follow-up work
- it keeps the generator on the ROI (`CPz / CP1 / CP2`) rather than trying to scale immediately to all channels
- it includes morphology-aware CPP losses, smoke/evidence output roots, and publication-style plotting utilities
- the generated waveforms are still not yet physiologically convincing enough to claim successful CPP generation

Current interpretation:

- the conditional generator can train and write **simulated EEG epoch arrays**
- the model does **not** directly output a CPP value or a standalone CPP waveform; instead, it generates EEG conditioned on a CPP-like class label
- CPP relevance is evaluated afterward by checking whether the generated EEG shows plausible structure in `CPz / CP1 / CP2`
- the generated waveforms are not yet physiologically convincing
- the strongest next step is ROI-focused conditional generation rather than immediate scaling to all channels

## Main scripts

- `scripts/train_phase1_eegnet.py`
- `scripts/baseline_phase1_cpp_features.py`
- `scripts/train_phase2_cue_dimensionality.py`
- `scripts/train_phase2_rt_fastslow.py`
- `scripts/train_phase1_conditional_vae.py`
- `scripts/analyze_conditional_vae_samples.py`
- `scripts/train_roi_conditional_generator.py`
- `scripts/analyze_roi_conditional_generator.py`
- `scripts/plot_cpp_style_roi_figure.py`

## One-click-style master script

Use the master script below from the repository root:

- `scripts/run_master.py`

It provides a single entry point for the currently supported experiment commands.

## Examples

From the repository root:

```bash
python3 scripts/train_phase1_eegnet.py --max-subjects 15 --epochs 40 --output-dir outputs/phase1_midrun
python3 scripts/baseline_phase1_cpp_features.py --max-subjects 15 --output-dir outputs/phase1_baseline_midrun
python3 scripts/train_phase2_cue_dimensionality.py --max-subjects 10 --epochs 10 --output-dir outputs/phase2_quickcheck
python3 scripts/train_phase2_rt_fastslow.py --max-subjects 5 --epochs 2 --output-dir outputs/phase2_rt_smoketest

# Canonical ROI-only conditional generator workflow
python3 scripts/train_roi_conditional_generator.py --output-dir outputs/roi_conditional_generator_smoke/train --max-subjects 10 --epochs 3
python3 scripts/analyze_roi_conditional_generator.py --output-root outputs/roi_conditional_generator_smoke
python3 scripts/plot_cpp_style_roi_figure.py --output-root outputs/roi_conditional_generator_smoke
```

## TIER adaptation note

This repository is aligned to the spirit of TIER Protocol 4.0 rather than copying every directory name literally. At the moment, it is a **transitional adaptation**, not a perfect literal TIER mirror. In particular:

- raw data are kept local and ignored by git
- scripts are organized as reproducible entry points
- reusable code is separated from executable scripts
- project notes and protocol documents are kept under `docs/`
- generated outputs are kept separate from source materials and current key results are retained with the repository

Current transitional caveats:

- legacy folders such as `1_Data/` and `2_Analysis/` are still present
- the loader supports both the preferred `data/raw/` path and the legacy `1_Data/` path
- legacy notebooks are still stored in `2_Analysis/` rather than a renamed canonical analysis directory

The current next scientific step is not more phase-1 tuning, but moving from broad full-head generation toward a more defensible ROI-focused conditional generator.
