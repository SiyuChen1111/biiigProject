# Pipeline Overview
## CPP Latent-Dynamics Project — TIER-Compliant Analysis Walkthrough

This document is the **written Master Script** companion to `Scripts/master_pipeline.ipynb`.
It provides a narrative description of every pipeline step for reproducibility and audit.

---

## Folder Structure

```
biiigProject/
├── conftest.py                     ← pytest sys.path bootstrap (project root)
├── pytest.ini                      ← pytest config (testpaths = tests)
├── README.md
├── AGENTS.md
├── logs.md                         ← chronological experiment log
│
├── Data/
│   ├── InputData/
│   │   ├── raw/                    ← original MATLAB exports (read-only source)
│   │   └── Metadata/
│   │       ├── DataSourcesGuide.md
│   │       └── channel_names.txt
│   ├── ProcessedData/              ← output of S0; input to S1–S4
│   │   ├── eeg_cpp_trials.npy      shape: (7297, 308, 3)
│   │   ├── times_ms.npy            shape: (308,)
│   │   ├── metadata.csv            7297 rows
│   │   ├── channel_names.txt
│   │   └── preprocessing_notes.md
│   └── IntermediateData/
│       └── latents_full/
│           └── latents_full.npz    shape: (7297, 308, 32)
│
├── Scripts/
│   ├── master_pipeline.ipynb       ← interactive master script (run me!)
│   ├── pipeline_overview.md        ← THIS FILE
│   │
│   ├── s0_preprocessing/
│   │   └── build_stage2_dataset_from_export.py   raw MATLAB → ProcessedData
│   │
│   ├── s1_modeling/                importable as "modeling"
│   │   ├── __init__.py
│   │   ├── config.py               ModelConfig · LossWeights · TrainingConfig
│   │   ├── data_contract.py        validate_stage2_dataset
│   │   ├── dataset.py              EEGWindowDataset · make_dataloaders
│   │   ├── model.py                CPPForwardGRU · loss sub-functions
│   │   ├── preparation.py          dataset package helpers
│   │   └── utils.py                seed · device utilities
│   │
│   ├── s2_training/                importable as "training"
│   │   ├── __init__.py
│   │   ├── train.py                train_model · export_full_latents_from_checkpoint
│   │   ├── sweep.py                run_cpp_prior_sweep
│   │   ├── controls.py             run_minimal_controls
│   │   └── cli.py                  command-line entry point
│   │
│   ├── s3_validation/
│   │   └── README.md               validation entrypoints now live in Results/validation/
│   │
│   └── s4_analysis/                importable as "analysis"
│       ├── __init__.py
│       ├── rt_ridge.py             run_ridge_rt_analysis
│       ├── analysis.py             PCA · latent readout
│       ├── make_publication_figures.py
│       └── notebooks/
│           └── 2_beh_z_reg.ipynb
│
├── Results/
│   ├── model_checkpoints/
│   │   └── best_model.pt           ← frozen production checkpoint
│   ├── validation/
│   │   ├── neural_goodness_of_fit.csv
│   │   ├── hidden_state_classification_decoding.csv
│   │   ├── hidden_state_neural_regression_decoding.csv
│   │   ├── validation_summary.md
│   │   └── hidden_cpp_audit/
│   ├── regression/
│   │   └── ridge_rt_hidden_rt_rerun/
│   └── figures/
│       ├── publication/            ← Figure 2, Supplementary S1
│       └── diagnostic/
│
└── tests/
    ├── conftest.py                 ← per-tests/ sys.path mirror
    └── test_stage2_modeling.py     ← unit tests
```

---

## Import Namespaces

| Directory | Importable as | Example import |
|-----------|---------------|---------------|
| `Scripts/s1_modeling/` | `modeling` | `from modeling.config import TrainingConfig` |
| `Scripts/s2_training/` | `training` | `from training.train import train_model` |
| `Scripts/s4_analysis/` | `analysis` | `from analysis.rt_ridge import run_ridge_rt_analysis` |

`sys.path` is injected automatically by `conftest.py` (root) and `tests/conftest.py`.

---

## Step-by-Step Description

### S0 · Pre-processing
**Script:** `Scripts/s0_preprocessing/build_stage2_dataset_from_export.py`
**Input:**  `Data/InputData/raw/` (MATLAB `.mat` files, 41 subjects)
**Output:** `Data/ProcessedData/`

Extracts response-locked epochs (±600 ms around button press), selects three CPP
channels (CP1, CP2, CPz), and concatenates all subjects into a single array.
Saves `eeg_cpp_trials.npy` (7297 × 308 × 3), `times_ms.npy`, `metadata.csv`,
`channel_names.txt`, and `preprocessing_notes.md`.

**Key design decision:** Response-locking aligns the CPP build-up to the moment
of evidence commitment, the scientifically meaningful anchor for this analysis.

---

### S1 · Data Contract Validation
**Module:** `modeling.data_contract.validate_stage2_dataset`
**Input:**  `Data/ProcessedData/`
**Output:** Console report + `Results/validation/validation_report.json`

Checks all five required files exist, channel order is exactly `(CP1, CP2, CPz)`,
and metadata contains `trial_id` and `alignment` columns.
Raises `ValueError` on failure — this is the **gate** before any model code runs.

---

### S2a · Model Training
**Module:** `training.train.train_model`
**Input:**  `Data/ProcessedData/`
**Output:** `Results/model_checkpoints/best_model.pt`

Trains `CPPForwardGRU` — a causal forward GRU encoder:

```
Input (batch, T, 3)  →  Linear(3→16) + LayerNorm
  →  Causal GRU (hidden=32, layers=1)
  →  Reconstruction head  (hidden → 3)         [current frame]
  →  Future-prediction head  (hidden → 3×H)    [next H=50 ms frames]
```

**Loss:** Composite self-supervised loss (`LossWeights`):
- `lambda_recon=1.0`       MSE: reconstruct current EEG frame
- `lambda_future=0.2`      MSE: predict next 50 ms
- `lambda_derivative=0.5`  Match first-order EEG slope
- `lambda_variance=0.5`    Align channel-level variance
- `lambda_cpp_mean=0.5`    CPP proxy (3-channel mean) alignment
- `lambda_cpp_prior=0.1`   Shape-prior group scale
- `lambda_smooth=0.001`    Latent temporal smoothness

No behaviour labels are used during training.
Early stopping: patience=15 epochs (max 100).

**Achieved:** test total_loss = 0.3953, CPP average waveform correlation = 0.9899.

---

### S2b · Hyperparameter Sweep (optional)
**Module:** `training.sweep.run_cpp_prior_sweep`
**Input:**  `Data/ProcessedData/`
**Output:** `Results/sweep/`

Scans `lambda_cpp_prior` ∈ {0.0, 0.05, 0.1, 0.2, 0.5} with 3 random seeds.
Best config (`long_002`, λ=0.1) selected by validation reconstruction loss.
Production model uses this best config.

---

### S2c · Latent Export
**Module:** `training.train.export_full_latents_from_checkpoint`
**Input:**  `Results/model_checkpoints/best_model.pt` + `Data/ProcessedData/`
**Output:** `Data/IntermediateData/latents_full/latents_full.npz`

Runs the frozen model over all 7 297 trials in `model.eval()` mode.
Saves the complete hidden-state tensor:
- `latents` : shape `(7297, 308, 32)` — hidden state at every time point
- `times_ms` : shape `(308,)`
- `trial_ids` : shape `(7297,)`

This file is the **input to all S3 and S4 analyses**.

---

### S3 · Neural Validation Audit
**Scripts:** `Results/validation/run_neural_validation_audit.py`
**Input:**  latents + `Data/ProcessedData/`
**Output:** `Results/validation/`

| Decoding target | Result | Interpretation |
|----------------|--------|----------------|
| CPP amplitude (R²) | 0.970 | Hidden states strongly encode CPP |
| CPP amplitude (Δ R² vs baseline) | 0.953 | Increment is genuine |
| RT bin (fast/slow) | marginal > chance | Weak real RT signal |
| Choice direction | ~chance | Not encoding stimulus identity ✓ |
| Condition | ~chance | Not encoding task context ✓ |

Generates evidence for **Figure 2** and **Supplementary S1**.

---

### S4a · Ridge Regression — Hidden States → RT
**Module:** `analysis.rt_ridge.run_ridge_rt_analysis`
**Input:**  `Data/IntermediateData/latents_full/latents_full.npz`
**Output:** `Results/regression/ridge_rt_hidden_rt_rerun/`

Predicts `log(RT_ms)` from window-averaged hidden states.
Nested CV: outer 5-fold KFold + inner held-out validation for α selection.

| Window | Baseline R² | +Hidden R² | Δ R² |
|--------|-------------|------------|------|
| −600 to −300 ms | 0.197 | **0.300** | **+0.103** |

Shuffled-hidden control R² ≈ 0.195 confirms the increment is not overfitting.

---

### S4b · Minimal Sanity Controls
**Module:** `training.controls.run_minimal_controls`
**Input:**  `Data/ProcessedData/`
**Output:** `Results/validation/`

Compares trained model against:
1. Untrained (random-weight) baseline — substantially worse → training matters
2. Time-shuffled latent control — R² drops sharply → temporal structure is real

---

### S4c · Publication Figures
**Script:** `Results/validation/make_publication_figures.py`
**Input:**  `Results/validation/` + `Results/regression/`
**Output:** `Results/validation/figures/publication_style/`

- **Figure 2:** Hidden-state vs CPP relationship
- **Supplementary S1:** Behavioural external validation

---

## Quick-Reference CLI Commands

```bash
# From the project root:

# Validate data contract
python Scripts/s2_training/cli.py validate --dataset-dir Data/ProcessedData

# Train model
python Scripts/s2_training/cli.py train --dataset-dir Data/ProcessedData \
    --output-dir Results/model_checkpoints

# Export latents
python Scripts/s2_training/cli.py extract-latents \
    --dataset-dir Data/ProcessedData \
    --checkpoint-path Results/model_checkpoints/best_model.pt \
    --output-dir Data/IntermediateData/latents_full

# Ridge RT analysis
python Scripts/s2_training/cli.py ridge-rt \
    --dataset-dir Data/ProcessedData \
    --latent-path Data/IntermediateData/latents_full/latents_full.npz \
    --output-dir Results/regression

# Run full test suite
pytest

# Open interactive master script
jupyter notebook Scripts/master_pipeline.ipynb
```

---

*Last updated: 2026-06-14*
