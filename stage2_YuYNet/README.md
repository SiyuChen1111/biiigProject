# Stage 2 YuYNet Modeling

This folder contains the response-locked CPP modeling pipeline for the stage2_YuYNet project.

## Current status

The active model is a **causal forward GRU** trained on response-locked EEG from `CP1`, `CP2`, and `CPz`.

The retained best result is stored in:

`evidence/best_cpp_model/`

That folder contains the final checkpoint, summary tables, latent exports, and comparison figures.

## What this setup does now

- keeps the response-locked EEG data as the only required input
- trains a self-learning causal GRU baseline
- reconstructs the CPP-like average waveform
- exports latent tensors for downstream PCA-style checks
- keeps a single retained best model for interpretation
- stores sweep results only in the retained model folder

## Active folder layout

```text
stage2_YuYNet/
├── CPP_latent_dynamics_scientific_proposal.md
├── EEG_preprocessing_request_for_partner.md
├── README.md
├── ROADMAP_EXECUTION_DECISIONS.md
├── dataset/
│   ├── eeg_cpp_trials.npy
│   ├── metadata.csv
│   ├── times_ms.npy
│   ├── channel_names.txt
│   ├── preprocessing_notes.md
│   └── README.md
├── evidence/
│   ├── stage0/
│   │   ├── stage0_preliminary_package_report.json
│   │   └── stage0_blocking_audit_report.json
│   └── best_cpp_model/
│       ├── best_model.pt
│       ├── best_run_summary.json
│       ├── best_training_loss.png
│       ├── best_cpp_overlay.png
│       ├── best_cpp_slope_overlay.png
│       ├── latents_train.npz
│       ├── latents_val.npz
│       ├── latents_test.npz
│       ├── stage2_cpp_average_sanity.npz
│       ├── stage2_training_report.json
│       ├── stage2_average_waveform_comparison.png
│       ├── sweep_results.csv
│       └── README.md
├── modeling/
│   ├── analysis.py
│   ├── cli.py
│   ├── config.py
│   ├── controls.py
│   ├── data_contract.py
│   ├── dataset.py
│   ├── model.py
│   ├── preparation.py
│   ├── prepare_contract.py
│   ├── sweep.py
│   ├── train.py
│   └── utils.py
├── script_pre_EEG/
└── tests/
    └── test_stage2_modeling.py
```

## File map

### Core pipeline

- `modeling/config.py` — config objects and defaults
- `modeling/data_contract.py` — dataset validation and intake report
- `modeling/dataset.py` — loading, normalization, masks, splits
- `modeling/model.py` — causal GRU encoder, heads, and losses
- `modeling/train.py` — training loop, checkpointing, latent export, test metrics
- `modeling/analysis.py` — CPP waveform and latent-dynamics analysis
- `modeling/controls.py` — time-index / response-hand controls
- `modeling/cli.py` — command-line entry point
- `modeling/sweep.py` — parameter sweep and best-run selection
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

This writes a preliminary package plus a blocking audit.

## How to run

From `stage2_YuYNet/`:

```bash
python -m modeling.cli prepare --dataset-dir dataset --output-dir evidence
python -m modeling.cli train --dataset-dir <dataset> --output-dir evidence
python -m modeling.cli analyze --dataset-dir <dataset> --latent-path <latents.npz> --output-dir evidence
python -m modeling.cli controls --dataset-dir <dataset> --latent-path <latents.npz> --output-dir evidence
python -m modeling.cli sweep --dataset-dir dataset --output-dir evidence
```

If you prefer to run from the repo root, use:

```bash
PYTHONPATH=stage2_YuYNet python -m modeling.cli ...
```

## Outputs

- `evidence/stage0/`
- `evidence/best_cpp_model/`
- `best_model.pt`
- `best_run_summary.json`
- `stage2_training_report.json`
- `stage2_cpp_average_sanity.npz`
- `stage2_average_waveform_comparison.png`
- `best_cpp_overlay.png`
- `best_cpp_slope_overlay.png`
- `best_training_loss.png`
- `latents_train.npz`, `latents_val.npz`, `latents_test.npz`

## Evaluation note

The current implementation has been verified on synthetic data and on the retained real-data best run. The repository now keeps one canonical result folder for interpretation and follow-up analysis.
