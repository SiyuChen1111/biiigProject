# Analysis log

## 2026-06-10 Initial neural validation audit

- What was done:
  - Reviewed the existing neural validation and publication-style figure pipeline
  - Confirmed that the earlier hidden-state figure and behavioral figure were generated from saved CSV tables rather than manually edited graphics
- Scripts used:
  - `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/model_neural_training_validation_audit/run_neural_validation_audit.py`
  - `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/model_neural_training_validation_audit/make_publication_figures.py`
- Files generated or confirmed:
  - `hidden_state_neural_regression_decoding.csv`
  - `hidden_state_classification_decoding.csv`
  - `behavioral_external_validation_rt.csv`
  - publication-style figure outputs
- Checks performed:
  - Traced which saved tables fed Main Figure 2
  - Confirmed that the older CPP target pipeline used EEG returned by `load_stage2_dataset(...)`
- Result summary:
  - The older figure pipeline was reproducible, but it mixed neural validation with a less explicit target-definition layer
- Interpretation:
  - A stricter audit was needed before making any hidden-state interpretation
- Next step:
  - Rebuild the hidden-to-CPP analysis using raw empirical CPP targets and stronger controls

## 2026-06-10 Strict hidden-CPP audit implementation

- What was done:
  - Implemented a dedicated strict audit script
  - Added raw empirical CPP feature extraction for amplitude, slope, and AUC
  - Added trial-level and within-subject ridge regression
  - Added shuffled and time-mismatch controls
  - Added subject-demeaned checks
- Script used:
  - `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/model_neural_training_validation_audit/run_hidden_cpp_audit.py`
- Files generated:
  - `hidden_to_cpp_cv_performance.csv`
  - `hidden_to_cpp_control_performance.csv`
  - `hidden_to_cpp_control_deltas.csv`
  - `hidden_to_cpp_fold_predictions.csv`
  - `hidden_to_cpp_coefficients_by_fold.csv`
  - `hidden_to_cpp_subject_demeaned_performance.csv`
  - `hidden_to_cpp_report.json`
- Checks performed:
  - Verified that latent trial order matched `metadata.csv`
  - Verified that the strict audit used raw empirical EEG from `eeg_cpp_trials.npy`
- Result summary:
  - Hidden states strongly predicted empirical CPP amplitude and AUC
  - Raw matched-window `R^2` values were extremely high
  - Control-corrected results were most convincing for late pre-response amplitude and AUC
  - Slope prediction was notably weaker
- Interpretation:
  - High raw scores reflect strong preservation of information from the same EEG trials
  - Control-corrected values are more informative than raw `R^2`
- Next step:
  - Audit task decoding under the same conservative logic

## 2026-06-10 Task-decoding audit

- What was done:
  - Reran task decoding across all main windows
  - Compared observed decoding against shuffled-label, shuffled-hidden, within-subject shuffled-hidden, within-subject x condition shuffled-hidden, and dummy-majority controls
- Script used:
  - `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/model_neural_training_validation_audit/run_hidden_cpp_audit.py`
- Files generated:
  - `hidden_task_decoding_performance.csv`
  - `hidden_task_decoding_controls.csv`
  - `hidden_task_decoding_class_counts.csv`
- Checks performed:
  - Confirmed class counts and chance expectations for each target
  - Compared observed balanced accuracy to the strongest available control
- Result summary:
  - RT bin showed the clearest stable margin above controls
  - Correctness was weakly positive
  - Condition and difficulty were near chance
  - Choice and arrangement were not reliable
- Interpretation:
  - Hidden states preserve some broad RT-related structure
  - Task coding beyond RT bin should not be overinterpreted
- Next step:
  - Regenerate the final publication-style hidden-state figure using the strict audit outputs

## 2026-06-10 Final figure and documentation refresh

- What was done:
  - Rebuilt Main Figure 2 using the strict audit outputs
  - Kept behavioral validation as a secondary supplementary figure
  - Added collaborator-facing and reproducibility-oriented documentation
- Scripts used:
  - `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/model_neural_training_validation_audit/run_hidden_cpp_audit.py`
  - `/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/model_neural_training_validation_audit/make_publication_figures.py`
- Files generated:
  - `figures/publication_style/main_figure_2_hidden_state_relations.png`
  - `figures/publication_style/main_figure_2_hidden_state_relations.pdf`
  - `figures/publication_style/supplementary_figure_behavioral_external_validation.png`
  - `figures/publication_style/supplementary_figure_behavioral_external_validation.pdf`
  - `hidden_cpp_audit/methods_trace.md`
  - `README.md`
  - `technical_reproducibility.md`
  - `analysis_log.md`
  - `hidden_cpp_audit/collaborator_summary.md`
- Checks performed:
  - Confirmed all referenced files exist
  - Confirmed that the README distinguishes raw `R^2` from control-corrected `delta R^2`
  - Confirmed that old normalized-target and new raw-target pipelines are clearly separated
- Result summary:
  - Final figure set now matches the stricter audit logic
  - Documentation is split into collaborator-facing and technical layers
- Interpretation:
  - The project now has a clearer basis for saying the hidden states preserve CPP-related neural information
  - The same documentation also clearly limits what can currently be claimed
- Next step:
  - Freeze the current model and define explicit CPP-related latent axes for stability testing

## Current status

The current model passes as suitable for exploratory hidden-CPP mapping, but only as neural validation. It does not yet pass as evidence for a strong behavioral decision mechanism.

## Next planned step

1. Freeze the current model.
2. Define CPP-related latent axes.
3. Test axis stability across folds and subjects.
4. Test whether CPP-related latent axes provide incremental RT prediction beyond empirical CPP features.
