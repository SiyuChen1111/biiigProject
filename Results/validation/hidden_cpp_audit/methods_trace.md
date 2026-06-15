# Methods Trace

This file traces the actual code and data dependencies used to generate the current publication-style figures related to hidden states, behavioral validation, and shared-scale waveform plotting.

## Figure 1

### `main_figure_2_hidden_state_relations.pdf`

1. **Output figure file**
   - `/Users/siyu/Documents/GitHub/biiigProject/Results/validation/figures/publication_style/main_figure_2_hidden_state_relations.pdf`
2. **Figure drawing code**
   - Script: `/Users/siyu/Documents/GitHub/biiigProject/Results/validation/make_publication_figures.py`
   - Function: `make_hidden_state_figure()`
3. **Immediate input tables used by the figure**
   - `/Users/siyu/Documents/GitHub/biiigProject/Results/validation/hidden_state_neural_regression_decoding.csv`
   - `/Users/siyu/Documents/GitHub/biiigProject/Results/validation/hidden_state_classification_decoding.csv`
4. **Upstream code that generated those tables**
   - Script: `/Users/siyu/Documents/GitHub/biiigProject/Results/validation/run_neural_validation_audit.py`
   - Function: `hidden_state_validation(...)`
5. **Model checkpoint used upstream**
   - `/Users/siyu/Documents/GitHub/biiigProject/Results/model_checkpoints/best_model.pt`
6. **Latent file used upstream**
   - `/Users/siyu/Documents/GitHub/biiigProject/Data/IntermediateData/latents_full/latents_full.npz`
7. **EEG/CPP file used upstream**
   - `/Users/siyu/Documents/GitHub/biiigProject/Data/ProcessedData/eeg_cpp_trials.npy`
   - Important detail: the upstream hidden-state validation used `load_stage2_dataset(...)`, which returns channel-normalized EEG for the regression targets in that script.
8. **Metadata file used upstream**
   - `/Users/siyu/Documents/GitHub/biiigProject/Data/ProcessedData/metadata.csv`
9. **Train/test or CV split used upstream**
   - Latent and metadata rows covered all 7297 trials from `latents_full.npz`.
   - Hidden-to-CPP regression in the old audit used **within-subject 5-fold CV only**.
   - Task decoding in the old audit used both `trial_level` and `within_subject` 5-fold CV, but the publication figure selected the `within_subject` rows.
10. **Time windows used upstream**
   - `-600 to -300 ms`
   - `-300 to -120 ms`
   - `-120 to -50 ms`
   - `-600 to -50 ms`
11. **Model type used upstream**
   - Regression: `Pipeline(StandardScaler(), Ridge(alpha=10.0))`
   - Classification: `Pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced", solver="liblinear"))`
12. **Scoring metric used upstream**
   - Regression heatmap: mean CV `R^2`
   - Task coding heatmap: balanced accuracy above the best control
13. **Control or shuffled baselines used upstream**
   - Regression controls: `shuffled_hidden_within_subject`, `shuffled_label`
   - Classification controls: `shuffled_hidden_within_subject`, `shuffled_hidden_within_subject_condition`, `shuffled_label`, `dummy_majority`

## Figure 2

### `supplementary_figure_behavioral_external_validation.pdf`

1. **Output figure file**
   - `/Users/siyu/Documents/GitHub/biiigProject/Results/validation/figures/publication_style/supplementary_figure_behavioral_external_validation.pdf`
2. **Figure drawing code**
   - Script: `/Users/siyu/Documents/GitHub/biiigProject/Results/validation/make_publication_figures.py`
   - Function: `make_behavior_figure()`
3. **Immediate input table used by the figure**
   - `/Users/siyu/Documents/GitHub/biiigProject/Results/validation/behavioral_external_validation_rt.csv`
4. **Upstream code that generated the table**
   - Script: `/Users/siyu/Documents/GitHub/biiigProject/Results/validation/run_neural_validation_audit.py`
   - Function: `behavioral_external_validation(...)`
5. **Model checkpoint used upstream**
   - `/Users/siyu/Documents/GitHub/biiigProject/Results/model_checkpoints/best_model.pt`
6. **Latent file used upstream**
   - `/Users/siyu/Documents/GitHub/biiigProject/Data/IntermediateData/latents_full/latents_full.npz`
7. **EEG/CPP file used upstream**
   - `/Users/siyu/Documents/GitHub/biiigProject/Data/ProcessedData/eeg_cpp_trials.npy`
   - Important detail: the CPP features in the old behavior figure were derived from the EEG array returned by `load_stage2_dataset(...)`, so they were based on channel-normalized EEG units rather than raw EEG units.
8. **Metadata file used upstream**
   - `/Users/siyu/Documents/GitHub/biiigProject/Data/ProcessedData/metadata.csv`
9. **Train/test or CV split used upstream**
   - 5-fold `within_subject` cross-validation over all 7297 trials
10. **Time windows used upstream**
   - Hidden states: `-600 to -50 ms`
   - CPP features: the same four windows used in `cpp_features(...)`
11. **Model type used upstream**
   - `Pipeline(StandardScaler(), Ridge(alpha=10.0))`
12. **Scoring metric used upstream**
   - Mean cross-validated `R^2` for `log(RT_ms)`
13. **Control or shuffled baselines used upstream**
   - `behavior_only`
   - `cpp_features_only`
   - `hidden_states_only`
   - `behavior_plus_cpp`
   - `behavior_plus_hidden`
   - `behavior_plus_cpp_plus_hidden`
   - `behavior_plus_shuffled_hidden`

## Figure 3

### `supplementary_shared_scale_waveforms_from_minus600.pdf`

1. **Output figure file**
   - `/Users/siyu/Documents/GitHub/biiigProject/Results/validation/figures/publication_style/supplementary_shared_scale_waveforms_from_minus600.pdf`
2. **Figure drawing code**
   - Script: `/Users/siyu/Documents/GitHub/biiigProject/Results/validation/make_publication_figures.py`
   - Function: `make_shared_scale_windowed_waveform_figure()`
3. **Immediate computation used by the figure**
   - Calls `load_model_predictions()` from `/Users/siyu/Documents/GitHub/biiigProject/Results/validation/run_neural_validation_audit.py`
4. **Model checkpoint used**
   - `/Users/siyu/Documents/GitHub/biiigProject/Results/model_checkpoints/best_model.pt`
5. **Latent file used**
   - None for this figure
6. **EEG/CPP file used**
   - `/Users/siyu/Documents/GitHub/biiigProject/Data/ProcessedData/eeg_cpp_trials.npy`
   - Important detail: the plotted "real" traces come from the channel-normalized EEG returned by `load_stage2_dataset(...)`, not from raw microvolt values.
7. **Metadata file used**
   - `/Users/siyu/Documents/GitHub/biiigProject/Data/ProcessedData/metadata.csv`
8. **Train/test split used**
   - Only `artifacts.test_indices` from the random trial split reproduced by `load_stage2_dataset(...)`
   - Split sizes from the current checkpoint config: train `5108`, val `1095`, test `1094`
9. **Time range shown**
   - Plot displays `-600 to 200 ms`
   - Shared y-axis scale is estimated from the `-600 to 0 ms` analysis window
10. **Model type used**
   - The trained `CPPForwardGRU` checkpoint is used to produce the reconstruction traces
11. **Scoring metric used**
   - None; this figure is descriptive rather than a scored cross-validation panel
12. **Control or shuffled baseline used**
   - None

## Split provenance

- The stage-2 dataset split is defined in `/Users/siyu/Documents/GitHub/biiigProject/Scripts/s1_modeling/dataset.py` by `_random_trial_split(...)`.
- The split is a reproducible random trial split with `TrainingConfig.seed = 42`.
- This means the old publication figures mix two validation layers:
  - a model-fitting split used by the GRU checkpoint itself;
  - a later readout CV analysis over the full latent file for hidden-state regressions and decoders.
