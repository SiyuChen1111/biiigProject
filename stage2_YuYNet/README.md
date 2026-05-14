# Stage 2 YuYNet Modeling

This folder contains the first CPP latent-dynamics baseline for the stage2_YuYNet project.

## What this model currently does

- validates the Stage 2 EEG contract
- trains a **causal forward GRU** on `CP1`, `CP2`, `CPz`
- learns trial-wise latent states `z_t`
- reconstructs the current EEG and predicts short future windows
- exports latent tensors for PCA / response-locked analysis
- runs minimal controls for time-index and response-hand confounds
- reports train / validation / **test** loss

## What the current model can achieve

It can already serve as a scientifically constrained baseline for asking:

- whether the CPP-region signal has a stable low-dimensional latent structure
- whether latent trajectories move toward a response-like state before RT
- whether reconstruction and short-horizon prediction are possible from a causal state
- whether time-index or response-hand effects are obvious confounds

What it **cannot** do yet:

- it does not prove evidence accumulation
- it does not handle multimodal EEG channels
- it does not include semi-supervised behavioral heads by default
- it does not yet provide publication-ready scientific conclusions

## File map

### Core pipeline

- `modeling/config.py` — config objects and defaults
- `modeling/data_contract.py` — Stage 1 dataset validation and intake report
- `modeling/dataset.py` — loading, subject-aware normalization, masks, splits
- `modeling/model.py` — causal GRU encoder, heads, and self-supervised loss
- `modeling/train.py` — training loop, checkpointing, latent export, test metrics
- `modeling/analysis.py` — PCA and latent-dynamics analysis
- `modeling/controls.py` — time-index / response-hand controls
- `modeling/cli.py` — command-line entry point
- `modeling/utils.py` — shared helpers

### Validation and notes

- `tests/test_stage2_modeling.py` — synthetic end-to-end checks
- `ROADMAP_EXECUTION_DECISIONS.md` — first-pass modeling decision record

## Expected data layout

```text
dataset/
├── eeg_cpp_trials.npy
├── metadata.csv
├── times_ms.npy
├── channel_names.txt
└── preprocessing_notes.md
```

For repository-only preparation of a preliminary package:

```bash
python -m modeling.cli prepare --dataset-dir dataset --output-dir evidence
```

This writes a preliminary package plus a blocking audit. It does not mark the dataset as formally training-ready.

## How to run

From `stage2_YuYNet/`:

```bash
python -m modeling.cli validate --dataset-dir <dataset> --output-dir evidence
python -m modeling.cli train --dataset-dir <dataset> --output-dir evidence
python -m modeling.cli analyze --dataset-dir <dataset> --latent-path <latents.npz> --output-dir evidence
python -m modeling.cli controls --dataset-dir <dataset> --latent-path <latents.npz> --output-dir evidence
```

If you prefer to run from the repo root, use:

```bash
PYTHONPATH=stage2_YuYNet python -m modeling.cli ...
```

## Outputs

- `stage1_data_contract_report.json`
- `stage2_training_report.json`
- `stage2_validation_loss_history.png`
- `stage2_reconstruction_sanity.png`
- `latents_train.npz`, `latents_val.npz`, `latents_test.npz`
- `stage3_analysis_report.json`
- `stage4_controls_report.json`

## Evaluation note

The current implementation has been verified on synthetic data with a full train → test → analysis → controls pass. The repository is now ready for real EEG contract input.
