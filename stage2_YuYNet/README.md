# Stage 2 YuYNet Modeling

This folder contains the current Stage 2 CPP latent modeling work. The folder has been cleaned so the retained outputs point to one current model and one full-trial latent export.

## Current model and latent output

Current recommended model:

```text
evidence/dataset_fixed_forward_gru_clean/stage2/best_model.pt
```

Current recommended full latent output:

```text
dataset_fixed/latents_full/latents_full.npz
```

This file is a local generated artifact and is not committed because it is larger than the normal GitHub file limit. Regenerate it with the command below if it is missing.

The full latent file contains:

- `Z`: encoder latent array with shape `(7297, 308, 32)`
- `metadata`: trial metadata aligned row-by-row with `Z`
- `times_ms`: the 308 EEG time points in milliseconds

Use this full latent export for downstream analyses such as selecting a time window, averaging within that window, building CPP-like latent scores, and linking those scores to DDM drift rate or other behavioral measures.

## Input to output process

The active input dataset is:

```text
dataset_fixed/
├── eeg_cpp_trials.npy
├── metadata.csv
├── times_ms.npy
├── channel_names.txt
└── preprocessing_notes.md
```

The model process is:

1. Load `dataset_fixed/eeg_cpp_trials.npy`, with trial-by-time-by-channel EEG from `CP1`, `CP2`, and `CPz`.
2. Use `dataset_fixed/metadata.csv` and `dataset_fixed/times_ms.npy` to keep trial identity and time alignment.
3. Run the trained Stage 2 forward GRU model from `evidence/dataset_fixed_forward_gru_clean/stage2/best_model.pt`.
4. Extract only the encoder latent `z` at every time point for every trial.
5. Save the full latent tensor and aligned metadata to `dataset_fixed/latents_full/latents_full.npz`.

No time-window averaging, PCA, or trial compression is applied in `latents_full.npz`.

## Current folder layout

```text
stage2_YuYNet/
├── README.md
├── CPP_latent_dynamics_scientific_proposal.md
├── EEG_preprocessing_request_for_partner.md
├── ROADMAP_EXECUTION_DECISIONS.md
├── build_stage2_dataset_from_export.py
├── run_dataset_fixed_stage2_stage3.py
├── dataset_fixed/
│   ├── eeg_cpp_trials.npy
│   ├── metadata.csv
│   ├── times_ms.npy
│   ├── channel_names.txt
│   ├── preprocessing_notes.md
│   ├── validation_report.json
│   ├── subject_trial_count_summary.csv
│   ├── diagnostic plots
│   └── latents_full/
│       ├── latents_full.npz
│       └── latent_extraction_report.json
├── evidence/
│   └── dataset_fixed_forward_gru_clean/
│       ├── dataset_fixed_stage2_stage3_report.json
│       └── stage2/
│           ├── best_model.pt
│           ├── stage2_completion_report.json
│           ├── latents_train.npz
│           ├── latents_val.npz
│           ├── latents_test.npz
│           └── reconstruction and waveform figures
├── modeling/
│   ├── config.py
│   ├── data_contract.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── analysis.py
│   ├── controls.py
│   ├── cli.py
│   ├── sweep.py
│   └── utils.py
├── script_pre_EEG/
└── tests/
    └── test_stage2_modeling.py
```

## File roles

- `dataset_fixed/`: current formal data input and full latent output.
- `evidence/dataset_fixed_forward_gru_clean/stage2/`: current retained model and model-quality figures.
- `modeling/`: code for data loading, model definition, training, latent export, and analyses.
- `tests/`: synthetic checks for the modeling pipeline and latent export alignment.
- `script_pre_EEG/`: original preprocessing source materials kept for reference.

Older sweep outputs, old preliminary datasets, test-only Stage 3 analyses, and cache files have been removed to avoid confusing them with the current retained result.

## Useful commands

Run from `stage2_YuYNet/`.

Extract full-trial latents from the current model:

```bash
python -m modeling.cli extract-latents \
  --dataset-dir dataset_fixed \
  --checkpoint-path evidence/dataset_fixed_forward_gru_clean/stage2/best_model.pt \
  --output-dir dataset_fixed/latents_full \
  --device auto \
  --output-filename latents_full.npz
```

Validate the active dataset:

```bash
python -m modeling.cli validate --dataset-dir dataset_fixed --output-dir evidence
```

Run the relevant latent-export test:

```bash
python -m unittest tests.test_stage2_modeling.Stage2ModelingTests.test_full_latent_extraction_preserves_metadata_order
```
