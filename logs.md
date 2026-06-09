# Project Reset Log

## 2026-05-12 — Repository reset around `stage2_YuYNet`

### What changed
- Moved the previous root-level project materials into `archive/`.
- Kept `stage2_YuYNet/` at the repository root as the only active project area.
- Replaced the root `AGENTS.md` so future sessions default to `stage2_YuYNet/` and do not read `archive/` unless explicitly requested.

### What was archived
Archived former root-level project content including:
- `1_Data/`
- `2_Analysis/`
- previous `AGENTS.md`
- `docs/`
- `low-level_1.pdf`
- `outputs/`
- `README.md`
- `requirements.txt`
- `scripts/`
- `src/`
- `初步介绍.md`

### Why this reset was made
The project focus has changed.

The old repository structure represented an earlier CPP/EEG workflow with broader analysis code, legacy plans, and retained outputs. That material is still preserved for historical reference, but it is no longer the active working surface.

The active direction is now the stage-2 latent-dynamics program centered on:
- `stage2_YuYNet/EEG_preprocessing_request_for_partner.md`
- `stage2_YuYNet/CPP_latent_dynamics_scientific_proposal.md`

The main goal is to build a neural-network model for CPP-related single-trial EEG latent dynamics.

### Prior-plan status
- Prior repository-wide active plans are considered **retired / superseded**.
- Older planning documents, workflows, and assumptions should be treated as historical context only.
- Any future implementation planning should be based on the stage-2 documents, not on the archived project structure.

### Operational rule going forward
- Default all new conversations to `stage2_YuYNet/`.
- Avoid reading `archive/` during normal work to prevent unnecessary context and memory usage.
- Use archived material only when explicitly needed for comparison, recovery, or provenance.

## 2026-05-12 — Stage 2 CPP latent-dynamics baseline implemented

### What was added
- Created a new modeling package under `stage2_YuYNet/modeling/`.
- Added a Stage 1 data-contract validator for the expected CPP EEG intake files.
- Added a deterministic pooled-subject loading / split / normalization pipeline.
- Added a causal forward GRU baseline with reconstruction and short-horizon future prediction heads.
- Added latent export, PCA-based analysis, and minimal control analyses.
- Added a repository-local README describing the model and file locations.

### Current architecture
- `modeling/data_contract.py` — validates `eeg_cpp_trials.npy`, `metadata.csv`, `times_ms.npy`, `channel_names.txt`, and `preprocessing_notes.md`.
- `modeling/dataset.py` — subject-aware normalization, split logic, and pre-response masking.
- `modeling/model.py` — `3 -> 16 -> LayerNorm -> GRU(hidden=32)` with reconstruction and future-prediction heads.
- `modeling/train.py` — training loop, checkpointing, latent export, and test evaluation.
- `modeling/analysis.py` — PCA / dimensionality / response-locked convergence analysis.
- `modeling/controls.py` — time-index and response-hand diagnostics.
- `modeling/cli.py` — command-line entry point.

### Training / evaluation status
- The pipeline runs end-to-end on synthetic Stage 2 data.
- This synthetic dataset is only a placeholder for pipeline verification and will be replaced by the real CPP EEG training dataset once the partner-preprocessed files are available.
- Static diagnostics are clean.
- Unit tests pass.
- A demo run produced test metrics and CPP-average preview artifacts.

### Representative demo metrics
- `test_total_loss = 1.1474`
- `test_future_loss = 0.8979`
- `test_recon_loss = 0.8317`

### CPP-average capability
- The model can currently produce a **conditional** CPP-average reconstruction / prediction trace from real CP1+CP2+CPz input.
- It does **not** yet generate an unconditional CPP signal from noise.

### Assessment
- The current model is a valid baseline, but training quality is still insufficient for strong scientific claims.
- The loss is still relatively high, so the model should be treated as a proof-of-pipeline rather than a final model.

---
- 回应：
  - 当前训练接口要的是单试次、刺激锁定的 EEG 输入，形状是 `trial × time × channel`，不是已经平均好的 CPP waveform。
  - 目前仓库里的可运行示例数据仍是 synthetic Stage 2 data，只用于验证 pipeline；真实训练数据还要等 partner 预处理后的 EEG 文件。
  - 训练配置当前使用的是默认 baseline：`max_epochs=100`、`early_stopping_patience=15`、自监督损失由 reconstruction + future prediction + smoothness 组成。
  - 如果后面训练效果仍然弱，优先怀疑数据规模 / 数据质量 / 真实 EEG 与 synthetic 差异，而不是先认为结构一定有问题。

## 2026-05-15 — Retained best response-locked CPP reconstruction model

### What changed
- The active Stage 2 modeling work shifted from stimulus-locked training assumptions to response-locked CPP reconstruction.
- The retained training data are `stage2_YuYNet/dataset/eeg_cpp_trials.npy`.
- The data shape is `255 trials x 308 time points x 3 channels`.
- The retained channels are `CP1`, `CP2`, and `CPz`.
- Behavior labels are not used in the current retained model.

### Current method
- The model remains a deterministic forward GRU.
- The model reconstructs current EEG and predicts a short future window.
- A soft CPP shape prior is used as an inductive bias, not as a hard DDM constraint.
- A parameter sweep selected the best model by CPP reconstruction quality rather than by total loss alone.

### Retained model
- Canonical retained output directory:
  - `stage2_YuYNet/evidence/best_cpp_model/`
- Retained checkpoint:
  - `stage2_YuYNet/evidence/best_cpp_model/best_model.pt`
- Best sweep run:
  - `long_002`

### Best parameters
- `max_epochs = 50`
- `batch_size = 32`
- `lambda_cpp_prior = 0.05`
- `lambda_late_amplitude = 4.0`
- `lambda_cpp_mean_alignment = 0.05`
- `lambda_slope_floor = 0.5`
- `slope_floor_ratio = 0.5`

### Best performance
- Mean CPP waveform correlation: `0.9899`
- Slope correlation: `0.9719`
- Amplitude ratio: `0.9911`
- Late-window amplitude error: `0.0168`
- MSE: `0.00044`

### Interpretation
- The retained model can reproduce the average response-locked CPP waveform under the current self-supervised reconstruction objective.
- This is a modeling milestone, not a final scientific claim.
- The result supports moving to latent-space analysis, but it does not yet establish evidence accumulation or behavioral relevance.

### Cleanup policy
- Non-best sweep runs and temporary smoke-training outputs are not part of the retained active result.
- The active retained modeling artifact is `stage2_YuYNet/evidence/best_cpp_model/`.

### Next plan
- Analyze latent states from the retained model.
- Inspect PC explained / dimensionality changes near response.
- Check whether latent dimensions align with CPP amplitude, slope, and cumulative CPP proxies.
- Use those analyses to decide whether the current forward GRU latent space supports interpretable response-proximal CPP dynamics.

### Final goal
- Build a response-locked CPP latent-dynamics model that reconstructs CPP shape and supports interpretable latent-space analysis of response-proximal neural dynamics.

## 2026-05-15 — Stage 3 PCA formalization and pooled significance check

### What changed
- Kept the retained best model artifacts under `stage2_YuYNet/evidence/best_cpp_model/`.
- Split latent analysis outputs into two retained folders:
  - `stage3/` for the conservative test-split PCA / CPP check
  - `stage3_pooled/` for pooled train/val/test significance analysis
- Added pooled-trial bootstrap and permutation testing for window-level PCA differences.
- Updated the best-model README to document the formal PCA outputs and the retained-model analysis layout.

### What the formal outputs now mean
- `stage3/` is the independent sanity check on `latents_test.npz`.
- `stage3_pooled/` is the sample-size stability analysis across all 255 trials.
- The pooled report currently shows 4 of 6 primary window-comparison tests statistically supported after FDR correction.

### Cleanup policy for this stage
- Keep the best model checkpoint, sweep summary, latent exports, and finalized PCA figures/tables.
- Keep both `stage3/` and `stage3_pooled/` because they support different interpretations.
- Treat temporary runner artifacts, hidden OS files, and any non-retained sweep outputs as disposable.

### Operational note
- Future PCA reads should default to `stage3_pooled/` when the question is sample-size stability.
- Use `stage3/` when the question is conservative test-split behavior or model sanity checking.

## 2026-05-27 — Fixed trial-level EEG-behavior dataset and reran Stage 2/3

### What changed
- Confirmed the old `resp_locked_erp.mat` export problem: EEG appending was previously outside the subject loop, so the old export retained only the final subject.
- Rebuilt the response-locked trial-level dataset from the corrected subject-block export.
- Added `stage2_YuYNet/build_stage2_dataset_from_export.py` to convert the corrected MATLAB export into the Stage 2/3 dataset contract.
- Added `stage2_YuYNet/run_dataset_fixed_stage2_stage3.py` to finish Stage 2 outputs from the best checkpoint and run test-split Stage 3 latent analyses.

### Fixed dataset
- Canonical fixed dataset:
  - `stage2_YuYNet/dataset_fixed/`
- EEG file:
  - `eeg_cpp_trials.npy`
- Final EEG shape:
  - `7297 trials x 308 time points x 3 channels`
- Metadata shape:
  - `7297 rows x 28 columns`
- Subjects:
  - `41`
- Channel order:
  - `CP1`, `CP2`, `CPz`
- Validation:
  - `validation_report.json` has `final_passed = true`

### Trial handling
- The corrected MATLAB export contains `10258` EEG-behavior trial pairs across 41 subjects before final filtering.
- Trials with non-finite response-locked EEG windows were removed together with their matching behavior rows.
- The retained dataset keeps EEG and metadata row order aligned by `trial_id`.
- The dataset is response-locked, not stimulus-locked.

### Stage 2 rerun
- Canonical retained result directory:
  - `stage2_YuYNet/evidence/dataset_fixed_forward_gru_clean/`
- Model:
  - Existing forward GRU reconstruction model, unchanged.
- No DDM model was trained.
- No VAE was trained.
- Split sizes:
  - train: `5108`
  - validation: `1095`
  - test: `1094`
- Total loss:
  - train: `0.3992`
  - validation: `0.3865`
  - test: `0.3953`
- Exported latents:
  - `stage2/latents_train.npz`
  - `stage2/latents_val.npz`
  - `stage2/latents_test.npz`

### Stage 2 figures
- `stage2/real_vs_recon_cpp_waveform.png`
- `stage2/real_vs_recon_cpp_slope.png`
- `stage2/channel_wise_reconstruction.png`
- `stage2/model_reconstructed_channel_average_waveforms_matched_axes.png`
- `stage2/model_real_vs_recon_channel_average_waveforms_matched_axes.png`

### Stage 3 test-split analysis
- Test latent shape:
  - `1094 trials x 308 time points x 32 latent dimensions`
- Main outputs:
  - `stage3_test/stage3_time_resolved_pca.csv`
  - `stage3_test/stage3_window_pca_summary.csv`
  - `stage3_test/stage3_global_pca_scores.npz`
  - `stage3_test/test_cpp_trial_features.csv`
  - `stage3_test/test_latent_cpp_linking_with_controls.csv`
  - `stage3_test/latent_score_group_behavior_summary.csv`
- Strongest observed latent-CPP link:
  - PC1 late score vs CPP late amplitude
  - Pearson `r = -0.9803`
  - linear `R2 = 0.9610`

### Controls and interpretation
- Trial-shuffled latent control was weak:
  - max absolute `r = 0.0825`
- Time-shuffled latent control remained high:
  - max absolute `r = 0.8652`
- Random latent direction control remained high:
  - max absolute `r = 0.9467`
- Interpretation:
  - Reconstruction evidence is now available on the full fixed dataset.
  - Latent-CPP evidence is strong descriptively, but not yet specific enough to claim a unique CPP latent axis.
  - Latent-behavior evidence is descriptive only: low/mid/high PC1 late score groups differ in RT and accuracy summaries.
  - DDM evidence is still not complete and should not be claimed from this run.

## 2026-05-28 — Cleaned current Stage 2 outputs and exported full latent z

### What changed
- Cleaned the active `stage2_YuYNet/` result structure so the project now points to one current Stage 2 model and one full-trial latent export.
- Removed older retained result folders that were based on earlier smaller datasets, older sweep selections, or split-only downstream analyses.
- Updated `stage2_YuYNet/README.md` so the current model location, current full latent output, input-to-output process, and active folder structure are explicit.
- Added a command-line latent extraction path for exporting all trials from an existing checkpoint without running reconstruction, future prediction analysis, PCA, or time-window averaging.
- Added a regression test to confirm that full latent export preserves trial order relative to `metadata.csv`.

### Current retained model
- Current retained checkpoint:
  - `stage2_YuYNet/evidence/dataset_fixed_forward_gru_clean/stage2/best_model.pt`
- Current retained dataset:
  - `stage2_YuYNet/dataset_fixed/`
- Current retained model result folder:
  - `stage2_YuYNet/evidence/dataset_fixed_forward_gru_clean/stage2/`

### Removed outputs
- Removed the old `stage2_YuYNet/evidence/best_cpp_model/` folder.
- Removed the old `stage2_YuYNet/evidence/dataset_fixed_forward_gru_clean/stage3_test/` folder.
- Removed the old `stage2_YuYNet/evidence/stage0/` folder.
- Removed the old preliminary `stage2_YuYNet/dataset/` folder.
- Removed transient `.DS_Store` and `__pycache__/` files.

### Full latent z export
- Exported full encoder latent states from the current model to:
  - `stage2_YuYNet/dataset_fixed/latents_full/latents_full.npz`
- This full latent file is retained locally and ignored by Git because it is larger than the normal GitHub file limit.
- Export report:
  - `stage2_YuYNet/dataset_fixed/latents_full/latent_extraction_report.json`
- Latent contents:
  - `Z`
  - `metadata`
  - `times_ms`
- Latent shape:
  - `7297 trials x 308 time points x 32 latent dimensions`
- The export preserves all time points.
- No time-window averaging was applied.
- No PCA was applied.
- No latent time compression was applied.
- The exported metadata remains row-aligned with `dataset_fixed/metadata.csv`.

### Verification
- Confirmed the current model checkpoint still exists after cleanup.
- Confirmed `latents_full.npz` contains `Z`, `metadata`, and `times_ms`.
- Confirmed `Z.shape = (7297, 308, 32)`.
- Confirmed `times_ms` matches `dataset_fixed/times_ms.npy`.
- Confirmed the latent metadata `trial_id` order exactly matches `dataset_fixed/metadata.csv`.
- Confirmed all latent values are finite.
- Ran the full-latent export alignment test successfully.

### Interpretation
- The active project state now keeps the current all-trial Stage 2 model result and its full latent-space output as the canonical materials for follow-up analyses.
- Older small-dataset and split-only result folders are no longer retained as active outputs because they can be mistaken for the current model results.
- Downstream researchers should use `latents_full.npz` when choosing response windows, deriving CPP-like latent scores, or preparing later DDM drift-rate regressions.

## 2026-06-09 — Ridge Regression 评估 hidden states 与 RT 的关系

### What changed
- Added a Ridge Regression analysis for testing whether time-averaged hidden states predict trial-level response time.
- Added a new command-line entry:
  - `python -m modeling.cli ridge-rt --dataset-dir dataset_fixed --latent-path dataset_fixed/latents_full/latents_full.npz --output-dir evidence/ridge_rt_hidden_rt`
- Added a regression test to confirm the Ridge RT analysis runs and produces the expected model-comparison outputs.

### Input data
- Latent source:
  - `stage2_YuYNet/dataset_fixed/latents_full/latents_full.npz`
- Analysis output:
  - `stage2_YuYNet/evidence/ridge_rt_hidden_rt/`
- Data quality checks:
  - `7297` trials
  - `308` response-locked time points
  - `32` hidden dimensions
  - RT has no missing values
  - latent values are finite
  - latent metadata trial order matches `dataset_fixed/metadata.csv`

### Method
- For each trial, hidden states were averaged across selected response-preceding time windows.
- Tested windows:
  - full pre-response: `-600` to `-50 ms`
  - early pre-response: `-600` to `-300 ms`
  - mid pre-response: `-300` to `-120 ms`
  - late pre-response: `-120` to `-50 ms`
- Ridge Regression was used instead of ordinary linear regression because hidden dimensions can be correlated.
- Two RT targets were evaluated:
  - `log(RT_ms)` as the main target
  - raw `RT_ms` as a supplementary target
- Four model types were compared:
  - baseline model: subject, difficulty, correctness
  - hidden-only model: averaged hidden-state features
  - baseline + hidden model
  - baseline + shuffled hidden control
- Ridge strength was selected inside the training portion of each cross-validation fold, then evaluated on held-out trials.

### Main results
- For `log(RT_ms)`, the baseline model reached about `R2 = 0.197`.
- The best result came from the early pre-response window, `-600` to `-300 ms`.
- In that window:
  - baseline model: `R2 = 0.1965`
  - hidden-only model: `R2 = 0.1490`
  - baseline + hidden model: `R2 = 0.2996`
  - baseline + shuffled hidden control: `R2 = 0.1946`
- Adding hidden states improved prediction over the baseline by about `0.103 R2`.
- The real hidden-state improvement was also stronger than the shuffled-hidden control by about `0.105 R2`.

### Window comparison
- `-600` to `-300 ms` was the strongest window for `log(RT_ms)`.
- `-600` to `-50 ms` also improved over baseline, but slightly less strongly.
- `-300` to `-120 ms` and `-120` to `-50 ms` showed weaker hidden-state contributions.
- This suggests the current averaged hidden-state RT signal is not limited to the immediate motor-contaminated response period.

### Outputs retained
- `ridge_rt_analysis_report.json`
- `ridge_rt_model_performance.csv`
- `ridge_rt_model_deltas.csv`
- `ridge_rt_predictions.csv`
- `ridge_rt_beta_stability.csv`
- `ridge_rt_hidden_coefficients_by_fold.csv`
- `ridge_rt_feature_quality.csv`
- `ridge_rt_window_deltas.png`

### Interpretation
- The current hidden states contain RT-relevant information beyond subject, difficulty, and correctness.
- The result supports using time-averaged hidden states for follow-up behavioral modeling.
- The evidence is predictive, not mechanistic: it shows hidden states help predict RT, but it does not by itself prove a specific evidence-accumulation mechanism.
- The strongest result being in the earlier response-preceding window is useful because it reduces concern that the result only reflects immediate response execution.

### Verification
- Ran the Ridge RT analysis on the full current latent export.
- Confirmed all expected output files were generated.
- Ran the Ridge RT unit test successfully:
  - `python -m unittest tests.test_stage2_modeling.Stage2ModelingTests.test_ridge_rt_analysis_runs`
