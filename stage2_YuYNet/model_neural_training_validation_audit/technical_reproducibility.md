# Hidden-CPP technical reproducibility record

## 1. Exact input files

- Model checkpoint:
  - `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/evidence/dataset_fixed_forward_gru_clean/stage2/best_model.pt`
- Latent file:
  - `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/dataset_fixed/latents_full/latents_full.npz`
- Latent extraction report:
  - `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/dataset_fixed/latents_full/latent_extraction_report.json`
- EEG file:
  - `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/dataset_fixed/eeg_cpp_trials.npy`
- Metadata file:
  - `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/dataset_fixed/metadata.csv`
- Time axis:
  - `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/dataset_fixed/times_ms.npy`
- Channel names:
  - `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/dataset_fixed/channel_names.txt`
- Primary audit script:
  - `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/model_neural_training_validation_audit/run_hidden_cpp_audit.py`
- Earlier audit script:
  - `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/model_neural_training_validation_audit/run_neural_validation_audit.py`
- Figure script:
  - `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/model_neural_training_validation_audit/make_publication_figures.py`
- Output directory:
  - `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/model_neural_training_validation_audit/hidden_cpp_audit/`
- Figure directory:
  - `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/model_neural_training_validation_audit/figures/publication_style/`

## 2. Data dimensions

- Number of trials: `7297`
- Number of time points per trial: `308`
- Hidden dimensions: `32`
- CPP channels: `CP1`, `CP2`, `CPz`
- Response-locked time axis: `-1000 ms` to `200 ms`
- Main audit windows:
  - `-600 to -300 ms`
  - `-300 to -120 ms`
  - `-120 to -50 ms`
  - `-600 to -50 ms`

## 3. Hidden feature construction

```python
X_hidden = Z[:, window_mask, :].mean(axis=1)
```

- `Z` is shaped `trial x time x hidden_dim`
- `window_mask` selects a response-locked time window
- The output feature matrix is shaped `trial x hidden_dim`

This construction was implemented in `run_hidden_cpp_audit.py` via `make_hidden_features(...)`.

## 4. CPP feature construction

```python
CPP = mean(CP1, CP2, CPz)
```

- Amplitude: mean CPP value in the selected window
- Slope: `np.polyfit(time, CPP, 1)[0]`
- AUC: `np.trapezoid(CPP, time)` with `np.trapz` fallback if needed

Important distinction:

- The older figure pipeline in `run_neural_validation_audit.py` derived CPP targets from the channel-normalized EEG returned by `load_stage2_dataset(...)`
- The strict hidden-CPP audit in `run_hidden_cpp_audit.py` used raw empirical EEG from `eeg_cpp_trials.npy`

## 5. Ridge regression model

The regression model can be written as:

`y_i = beta_0 + beta_1 h_i1 + beta_2 h_i2 + ... + beta_p h_ip + epsilon_i`

- `y_i`: CPP feature for trial `i`
- `h_ij`: hidden-state feature `j` for trial `i`
- `beta_0`: intercept
- `beta_j`: regression coefficient for hidden dimension `j`
- `epsilon_i`: residual error

Ridge objective:

`minimize sum_i (y_i - yhat_i)^2 + lambda sum_j beta_j^2`

Implementation:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", RidgeCV(alphas=np.logspace(-3, 5, 25), cv=5))
])
```

This was implemented in `build_ridge_pipeline()` inside `run_hidden_cpp_audit.py`.

## 6. Cross-validation

- Trial-level CV:
  - 5-fold `KFold(shuffle=True, random_state=2026)`
- Within-subject CV:
  - 5-fold subject-balanced splits created by `within_subject_folds(...)`
- Subject-demeaned checks:
  - Features and targets were demeaned within subject before the same outer CV

Reported metrics:

- Fold-wise `R^2`
- Mean CV `R^2`
- Standard deviation across folds
- Pooled prediction-target correlation
- RMSE
- MAE
- Selected ridge alpha
- Control-corrected `delta R^2`

## 7. Controls

- Trial-shuffled hidden:
  - Shuffles hidden-feature rows across trials
  - Tests whether trial identity matters beyond marginal feature distribution
- Within-subject shuffled hidden:
  - Shuffles hidden features within each subject
  - Preserves subject-level distribution while breaking trial-level correspondence
- Within-subject x condition shuffled hidden:
  - Shuffles within subject-condition cells
  - Preserves both subject and condition structure
- Time-window mismatch:
  - Uses hidden features from one time window to predict CPP targets from a different window
  - Tests whether the relationship is time-specific or dominated by global trial-level factors
- Subject-demeaned check:
  - Removes each subject's mean feature level and mean target level
  - Tests whether prediction survives beyond subject offsets
- Task-decoding shuffled-label control:
  - Randomizes training labels while keeping features fixed
- Task-decoding dummy-majority control:
  - Uses the majority class as a conservative baseline

## 8. Output files

| File name | Path | Description | Generated by | Used in |
|---|---|---|---|---|
| `hidden_to_cpp_cv_performance.csv` | `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/model_neural_training_validation_audit/hidden_cpp_audit/hidden_to_cpp_cv_performance.csv` | Raw hidden-to-CPP CV performance summary | `run_hidden_cpp_audit.py` / `run_hidden_cpp_audit()` | README, collaborator summary, Main Figure 2a |
| `hidden_to_cpp_control_performance.csv` | `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/model_neural_training_validation_audit/hidden_cpp_audit/hidden_to_cpp_control_performance.csv` | Control-model performance summary | `run_hidden_cpp_audit.py` / `run_hidden_cpp_audit()` | Technical audit, control comparisons |
| `hidden_to_cpp_control_deltas.csv` | `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/model_neural_training_validation_audit/hidden_cpp_audit/hidden_to_cpp_control_deltas.csv` | Raw minus best-control `delta R^2` | `run_hidden_cpp_audit.py` / `run_hidden_cpp_audit()` | README, collaborator summary, Main Figure 2b |
| `hidden_to_cpp_fold_predictions.csv` | `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/model_neural_training_validation_audit/hidden_cpp_audit/hidden_to_cpp_fold_predictions.csv` | Fold-wise predictions and targets | `run_hidden_cpp_audit.py` / `run_hidden_cpp_audit()` | Reproducibility |
| `hidden_to_cpp_coefficients_by_fold.csv` | `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/model_neural_training_validation_audit/hidden_cpp_audit/hidden_to_cpp_coefficients_by_fold.csv` | Ridge coefficients per fold and hidden dimension | `run_hidden_cpp_audit.py` / `run_hidden_cpp_audit()` | Reproducibility |
| `hidden_to_cpp_report.json` | `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/model_neural_training_validation_audit/hidden_cpp_audit/hidden_to_cpp_report.json` | Summary report and leakage audit | `run_hidden_cpp_audit.py` / `run_hidden_cpp_audit()` | README, analysis log |
| `hidden_to_cpp_subject_demeaned_performance.csv` | `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/model_neural_training_validation_audit/hidden_cpp_audit/hidden_to_cpp_subject_demeaned_performance.csv` | Subject-demeaned performance | `run_hidden_cpp_audit.py` / `run_hidden_cpp_audit()` | Leakage audit |
| `hidden_task_decoding_performance.csv` | `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/model_neural_training_validation_audit/hidden_cpp_audit/hidden_task_decoding_performance.csv` | Observed decoding performance and margin above best control | `run_hidden_cpp_audit.py` / `run_task_decoding_audit()` | README, Main Figure 2c |
| `hidden_task_decoding_controls.csv` | `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/model_neural_training_validation_audit/hidden_cpp_audit/hidden_task_decoding_controls.csv` | Decoding control results | `run_hidden_cpp_audit.py` / `run_task_decoding_audit()` | Reproducibility |
| `hidden_task_decoding_class_counts.csv` | `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/model_neural_training_validation_audit/hidden_cpp_audit/hidden_task_decoding_class_counts.csv` | Class counts and chance expectations | `run_hidden_cpp_audit.py` / `run_task_decoding_audit()` | Reproducibility |
| `methods_trace.md` | `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/model_neural_training_validation_audit/hidden_cpp_audit/methods_trace.md` | Trace of earlier figure generation pipeline | `run_hidden_cpp_audit.py` / `write_methods_trace()` | Reproducibility |
| `main_figure_2_hidden_state_relations.png/.pdf` | `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/model_neural_training_validation_audit/figures/publication_style/main_figure_2_hidden_state_relations.png` and `.pdf` | Final publication-style figure | `run_hidden_cpp_audit.py` / `make_updated_figure_2()` | Main figure |
| `supplementary_figure_behavioral_external_validation.png/.pdf` | `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/model_neural_training_validation_audit/figures/publication_style/supplementary_figure_behavioral_external_validation.png` and `.pdf` | Secondary external RT validation figure | `run_hidden_cpp_audit.py` / `make_behavior_figure()` | Supplementary figure |
| `supplementary_shared_scale_waveforms_from_minus600.png/.pdf` | `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/model_neural_training_validation_audit/figures/publication_style/supplementary_shared_scale_waveforms_from_minus600.png` and `.pdf` | Shared-scale waveform figure | `make_publication_figures.py` / `make_shared_scale_windowed_waveform_figure()` | Supplementary figure |

## 9. Reproducibility commands

Minimal commands:

```bash
python /Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/model_neural_training_validation_audit/run_hidden_cpp_audit.py
python /Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/model_neural_training_validation_audit/make_publication_figures.py
```

What these do:

- `run_hidden_cpp_audit.py`
  - writes the methods trace
  - reruns the strict hidden-to-CPP audit
  - reruns task decoding audit
  - refreshes Main Figure 2 and the behavioral supplementary figure
- `make_publication_figures.py`
  - refreshes the broader publication-style figure set, including the waveform figure

## 10. Known limitations

- Hidden states and CPP targets come from the same EEG trial segments.
- Very high raw `R^2` should therefore be interpreted conservatively.
- Amplitude and AUC effects are stronger than slope effects.
- Task decoding remains weak outside RT bin, and only mildly positive for correctness.
- Behavioral validation is secondary and should not be used as the main mechanistic claim.
- The strict audit supports neural information preservation, not a full decision-mechanism account.

## 11. Git and provenance notes

- Repository root:
  - `/Users/siyu/Documents/GitHub/biiigProject`
- Active project area:
  - `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet`
- Documentation commit created for this audit:
  - see local git commit with message `Document hidden-CPP audit and validation outputs`
- The exact commit hash is reported in the final run summary after commit.
