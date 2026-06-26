# Full-Data Rank-5 Low-Rank RNN Latent Training Report

## Executive Summary

- Trained rank-5 low-rank RNN models on the complete processed dataset for seeds: [0, 1, 2, 3, 4].
- Reference model for latent alignment: seed 2.
- Mean subject-aware CV R2, CPP-only: 0.0980.
- Mean subject-aware CV R2, CPP + selected latents: 0.1070.
- Delta R2 CPP+z minus CPP-only: 0.0090.
- Conclusion category: Category A: Strong CPP-like but no behavioral improvement.

## Data Summary

- n_subjects: 41
- n_total_trials_available: 7297
- n_valid_trials_after_exclusion: 7297
- n_excluded_trials: 0
- n_correct_trials: 5830
- n_error_trials: 1467
- RT_ms_mean: 823.7262135124023
- RT_ms_sd: 255.97952248649176
- RT_ms_min: 498.21
- RT_ms_max: 1799.1
- alignment: response_locked
- response_time_zero_ms: 0
- time_unit: milliseconds
- sampling_rate_hz: 255.8333282470703
- input_window_start_ms: -1000.0
- input_window_end_ms: 200.0
- n_timepoints: 308
- model_input_channels: CP1,CP2,CPz
- cpp_channels_in_model_input: True

The data are response-locked, with response at 0 ms. The model input channels are CP1, CP2, and CPz, so high CPP-latent correlation can partly reflect compression of CPP-related input activity.

## Training Setup

The training objective was the existing composite self-supervised loss: EEG reconstruction, one-step future prediction, derivative matching, variance alignment, CPP mean alignment, CPP shape prior terms, and latent smoothness.
Best validation losses ranged from 0.1734 to 0.6657.

## Validation Strategy

The final models were trained on the complete valid dataset for descriptive latent interpretation. Behavioral claims use downstream GroupKFold validation by subject, using full-data model latents as extracted features. This tests downstream behavioral generalization, not full RNN retraining on held-out subjects.

## Latent Alignment and CPP Mapping

- seed 0 amplitude_like_latent: aligned_z2 (aligned_z2_late_mean, r=0.944)
- seed 0 slope_like_latent: aligned_z2 (aligned_z2_slope, r=0.949)
- seed 1 amplitude_like_latent: aligned_z2 (aligned_z2_late_mean, r=0.942)
- seed 1 slope_like_latent: aligned_z2 (aligned_z2_slope, r=0.943)
- seed 2 amplitude_like_latent: aligned_z2 (aligned_z2_late_mean, r=0.917)
- seed 2 slope_like_latent: aligned_z2 (aligned_z2_slope, r=0.915)
- seed 3 amplitude_like_latent: aligned_z3 (aligned_z3_late_mean, r=0.928)
- seed 3 slope_like_latent: aligned_z3 (aligned_z3_slope, r=0.934)
- seed 4 amplitude_like_latent: aligned_z2 (aligned_z2_late_mean, r=0.950)
- seed 4 slope_like_latent: aligned_z2 (aligned_z2_slope, r=0.949)

Raw z indices should not be interpreted before alignment. The report therefore uses labels such as amplitude-like and slope-like latent rather than assuming the smoke-model z1/z4 identities.

## Behavioral Prediction

The key comparison is CPP + z versus CPP-only. Here the average delta R2 was 0.0090. A positive but tiny value should be treated cautiously; a negative or near-zero value means the latents mostly recapitulate CPP information for RT prediction.

## Accumulation-Like Diagnostics

The generated figures compare CPP and selected latent trajectories across fast, medium, and slow RT groups, averaging within subject before grand averaging. These plots are descriptive support for response-proximal build-up dynamics.

## Limitations

- The available processed data are response-locked, not stimulus-locked.
- CPP channels are model inputs, so CPP-like latents may reflect low-dimensional compression of input CPP activity.
- Subject-aware behavioral validation is downstream validation using extracted latents; full RNN retraining inside each subject fold was not run by default.
- These analyses support interpretability, not causal claims.

## Final Interpretation

After full-data training, the model reliably recovers CPP-like coordinates, but they do not clearly improve RT prediction beyond conventional CPP features. The safest wording is that the low-rank RNN learns CPP-like response-proximal latent dynamics, which are candidate low-dimensional accumulation-like coordinates rather than proven evidence-accumulation variables.

## Recommended Next Step

Run a smaller number of full subject-held-out RNN retraining folds for the selected configuration, then compare whether the same aligned CPP-like latent subspace appears on held-out subjects.
