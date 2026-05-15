# Best CPP Reconstruction Model

## Status

This directory contains the retained best model from the Stage 2 response-locked CPP reconstruction sweep.

All non-best sweep runs and temporary training outputs should be treated as disposable. This directory is the canonical retained result for the current modeling stage.
The retained checkpoint is `long_002`.

## Data

The model was trained on:

```text
stage2_YuYNet/dataset/eeg_cpp_trials.npy
```

Data shape:

```text
255 trials x 308 time points x 3 channels
```

Channels:

```text
CP1, CP2, CPz
```

The data are response-locked EEG trials. This stage does not use behavior labels.

## Model

The retained model uses the current forward GRU architecture:

```text
CP1/CP2/CPz EEG -> linear projection -> LayerNorm -> forward GRU -> latent state
```

The model has two output heads:

```text
current EEG reconstruction
short-horizon future prediction
```

No BiGRU, VAE, DDM joint model, or semi-supervised behavioral head is used in this retained model.

## Best Parameters

The best run is:

```text
long_002
```

### Parameters that mainly shaped the result

```text
lambda_cpp_prior = 0.05
lambda_late_amplitude = 4.0
lambda_cpp_mean_alignment = 0.05
lambda_slope_floor = 0.5
slope_floor_ratio = 0.5
max_epochs = 50
```

These settings had the clearest effect on whether the model stayed too flat or recovered the CPP shape well.

### Mostly fixed settings

```text
batch_size = 32
lambda_future = 0.2
lambda_monotonic = 1.0
lambda_recon = 1.0
lambda_derivative = 0.5
lambda_variance = 0.5
lambda_smooth = 0.001
learning_rate = 0.001
weight_decay = 0.0001
hidden_dim = 32
projection_dim = 16
analysis_window_ms = [-600, -50]
early_window_ms = [-600, -300]
mid_window_ms = [-300, -120]
late_window_ms = [-120, -50]
```

## Performance

Best model metrics:

```text
mean CPP waveform correlation = 0.9899
slope correlation = 0.9719
amplitude ratio = 0.9911
late-window amplitude error = 0.0168
MSE = 0.00044
```

Interpretation:

The retained model reproduces the average CPP waveform and slope pattern well under the current response-locked EEG reconstruction objective. This supports using the model for the next latent-space analysis step, but it does not by itself prove evidence accumulation or behavioral relevance.

## Files

```text
best_model.pt
best_run_summary.json
stage2_training_report.json
sweep_results.csv
stage2_cpp_average_sanity.npz
latents_train.npz
latents_val.npz
latents_test.npz
best_cpp_overlay.png
best_cpp_slope_overlay.png
best_training_loss.png
stage2_average_waveform_comparison.png
```

## Analysis Outputs

The retained latent exports support two analysis folders:

```text
stage3/
stage3_pooled/
```

### `stage3/`

- Conservative test-split analysis using `latents_test.npz`
- Response-locked PCA, participation ratio, CPP proxy mapping, and control plots
- Use this for the independent sanity check on the retained model

### `stage3_pooled/`

- Pooled train/val/test analysis using all three latent exports
- Adds bootstrap confidence intervals and permutation p-values for the PCA window comparisons
- Use this to assess whether the pre-response changes are stable when trial count increases

## Log Notes

- The pooled analysis currently reports `255` trials and `4/6` statistically supported primary tests after FDR correction.
- The pooled result is useful for sample-size stability, but it is not an independent test-set inference because `train` latents were seen during fitting.
- The test-only and pooled folders are both retained so the sample-size effect can be compared directly.

The current scientific goal remains:

```text
build a response-locked CPP latent dynamics model that can reconstruct CPP shape and support interpretable latent-space analysis of response-proximal dynamics
```
