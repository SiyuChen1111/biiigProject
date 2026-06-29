# Project Reset Log

## Current status note — Updated 2026-06-29

This log preserves earlier project states as historical records. Some older
entries refer to `stage2_YuYNet/`, which was the active working structure at
that time. The current checkout now uses the root-level TIER-style structure:
`Data/`, `Scripts/`, `Results/`, and `tests/`.

The current main scientific and modeling line is the low-rank RNN workflow,
especially the Rank-5 model and its five learned `z` variables (`z1`-`z5`).
Earlier GRU hidden-state work remains useful background and comparison, but it
is no longer the default main line for new analysis.

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


## 2026-06-10 — Neural validation audit and fast external validation

### What changed
- Ran a strict neural-validation audit on the full latent export from the current retained model.
- Added a fast external behavioral validation to test whether hidden states provide incremental RT prediction over hand-crafted CPP features.
- Generated publication-ready figures for the hidden-state relationship analysis.

### Method
- Strict audit compared hidden-state decoding against three control conditions:
  - Shuffled trial order (trial-level control)
  - Time-shuffled latent states (temporal structure control)
  - Random latent direction (direction specificity control)
- External behavioral validation compared four predictor sets:
  - Baseline only: subject, difficulty, correctness
  - Baseline + hand-crafted CPP features (amplitude, slope)
  - Baseline + hidden states
  - Baseline + shuffled hidden states
- All comparisons used the same nested cross-validation scheme as the Ridge RT analysis.

### Main results

#### Hidden states → CPP amplitude
- `R2 = 0.970` for hidden states predicting CPP amplitude.
- Delta `R2 = 0.953` after controlling for baseline features.
- Time-shuffled control `R2 = 0.312`: temporal structure accounts for part, but not most, of the link.
- Interpretation: hidden states encode CPP amplitude strongly and specifically.

#### RT bin decoding
- Hidden states show a marginal but consistent edge over shuffled controls for fast vs slow RT tertile classification.
- Choice direction decoding: near chance — hidden states do not encode stimulus identity.
- Experimental condition decoding: near chance — hidden states do not encode task context.

#### Fast external behavioral validation
- `behavior + hand-crafted CPP features`: `R2 = 0.168`
- `behavior + hidden states`: `R2 = -0.115`
- Interpretation: in the fast external validation setting, hidden states do **not** provide reliable incremental RT prediction beyond hand-crafted CPP features.
- This is a direct challenge to the claim that the latent space adds behavioral value.

### Outputs retained
- `Results/validation/behavioral_goodness_of_fit_summary.csv`
- `Results/validation/hidden_state_decoding_results.csv`
- `Results/validation/validation_summary.md`
- `Results/figures/publication/` — Figure 2 (hidden-state relationship) and Supplementary S1 (behavioral external validation)

### Interpretation
- The model reconstructs CPP form well and the hidden states encode CPP amplitude.
- The behavioral case is weak: hidden states do not reliably outperform direct CPP measurements for predicting RT.
- The current result is descriptive, not mechanistic. No DDM drift-rate evidence is available yet.
- The next required step is to define a formal CPP-related latent axis from the hidden states and test whether that axis provides incremental behavioral prediction over hand-crafted CPP features.


## 2026-06-14 — Full repository restructure and code refactoring

### What changed
- Restructured the entire repository to follow TIER Protocol 4.0 conventions.
- Refactored all active modeling code for readability: separated concerns into distinct dataclasses, split large functions into sub-functions, added comprehensive docstrings and type hints throughout.
- Fixed the test suite import infrastructure.
- Generated two new master-script deliverables.

### Folder restructure

The old layout (`stage2_YuYNet/modeling/`) was replaced by a TIER-aligned structure:

```
biiigProject/
├── Data/
│   ├── InputData/raw/           ← original MATLAB exports (read-only)
│   ├── ProcessedData/           ← output of S0; input to all downstream steps
│   └── IntermediateData/
│       └── latents_full/        ← full latent tensor (7297 × 308 × 32)
├── Scripts/
│   ├── master_pipeline.ipynb    ← NEW: interactive master script (S0→S4)
│   ├── pipeline_overview.md     ← NEW: TIER-style written walkthrough
│   ├── s0_preprocessing/        ← raw → processed data
│   ├── s1_modeling/             ← model library, importable as "modeling"
│   ├── s2_training/             ← training executables, importable as "training"
│   ├── s3_validation/           ← post-training audit scripts
│   └── s4_analysis/             ← downstream analyses, importable as "analysis"
├── Results/
│   ├── model_checkpoints/       ← best_model.pt
│   ├── validation/
│   ├── regression/
│   └── figures/
└── tests/
```

- `stage2_YuYNet/` was deleted after all contents were migrated.
- `archive/` was not migrated (retained as historical reference only).
- `prepare_contract.py` (legacy thin wrapper) was deleted.

### Code refactoring

#### `config.py` — three-class separation
The original single `TrainingConfig` (53 mixed fields) was split into three dataclasses with distinct responsibilities:

- `ModelConfig` — architecture parameters only (`projection_dim`, `hidden_dim`, `num_layers`).
- `LossWeights` — all 11 `lambda_*` loss weights, shape-prior flags, and loss-related time windows.
- `TrainingConfig` — training loop and data-pipeline parameters, with `model: ModelConfig` and `loss: LossWeights` as nested fields. Backwards-compatible `@property` shims allow existing call sites to continue using `config.hidden_dim`, `config.lambda_recon`, etc. without modification.

#### `model.py` — loss function decomposition
The 17-parameter "god function" `masked_self_supervised_loss` was refactored into:

- `_compute_reconstruction_losses()` — reconstruction, future prediction, derivative matching, variance alignment.
- `_compute_cpp_shape_prior_losses()` — monotonicity, slope floor, late amplitude, mean alignment.
- `_compute_smoothness_loss()` — latent temporal smoothness.
- `masked_self_supervised_loss()` — orchestration only; accepts a single `LossWeights` argument.

#### `dataset.py` — mask logic deduplication
Extracted `_build_trial_mask()` to eliminate duplicate mask construction code that existed separately in `load_stage2_dataset()` and `make_dataloaders()`. Added inline comments explaining the physical meaning of `valid_time` (time points excluded because they lack complete future-prediction targets).

#### `rt_ridge.py` — English section headers
Added six English-language section headers to the 628-line file:

- `§ 1  Data Loading & Input Validation`
- `§ 2  Feature Engineering`
- `§ 3  Ridge Regression & Cross-Validation`
- `§ 4  Summary Statistics & Delta Metrics`
- `§ 5  Visualisation & Output`
- `§ 6  Entry Point`

#### `train.py`, `sweep.py`, `controls.py` — docstrings and type hints
Added complete docstrings and type hint annotations to all public and private functions. For `train.py`, added explicit notes on the implicit dependency between `_save_cpp_average_examples()` (which writes a `.npz` side-effect) and the two functions that depend on that file being present.

#### `cli.py` — cross-package import fix
Rewrote the command-line entry point with a path-bootstrap block so that `rt_ridge` (in `s4_analysis`) is imported correctly regardless of which directory the CLI is invoked from.

### Test infrastructure fixes
- Created `conftest.py` at the project root to inject `sys.path` before pytest collects any module. This was the root cause of the `ModuleNotFoundError: No module named 'modeling'` error.
- Created `pytest.ini` at the project root to pin `testpaths = tests` and `addopts = -v`.
- Fixed four incorrect import lines in `tests/test_stage2_modeling.py` that used `from modeling.xxx` for modules that now live in the `training` and `analysis` namespaces:
  - `from training.controls import run_minimal_controls`
  - `from training.sweep import run_small_cpp_prior_sweep`
  - `from training.train import train_model, export_full_latents_from_checkpoint`
  - `from analysis.rt_ridge import run_ridge_rt_analysis`
- Created `__init__.py` files for `s2_training/`, `s4_analysis/`, and `s3_validation/`.

### New deliverables

#### `Scripts/master_pipeline.ipynb`
An interactive Jupyter notebook acting as the TIER-compliant master script. Contains 22 cells covering the full S0→S4 pipeline. Each stage has a Markdown explanation cell followed by a self-contained executable code cell. Cells can be run individually or sequentially top-to-bottom to reproduce the full analysis.

#### `Scripts/pipeline_overview.md`
A 276-line written walkthrough in TIER 4.0 style. Includes: complete annotated folder tree, per-step narrative description, loss-term parameter table, Ridge RT result table, import namespace reference, and CLI quick-reference command block.

### Import namespace convention (active going forward)

| Directory | Import namespace | Example |
|---|---|---|
| `Scripts/s1_modeling/` | `modeling` | `from modeling.config import TrainingConfig` |
| `Scripts/s2_training/` | `training` | `from training.train import train_model` |
| `Scripts/s4_analysis/` | `analysis` | `from analysis.rt_ridge import run_ridge_rt_analysis` |

### Current project status
- Model training: complete and frozen (`Results/model_checkpoints/best_model.pt`)
- Full latent export: complete (`Data/IntermediateData/latents_full/latents_full.npz`, shape `7297 × 308 × 32`)
- Neural reconstruction validation: passed (`R2 = 0.88`, CPP waveform correlation `0.99`)
- Hidden-state → CPP amplitude: strong (`delta R2 = 0.953`)
- Hidden-state → RT (incremental): weak in external validation; descriptive only
- DDM drift-rate regression: not yet done
- CPP latent axis: not yet formally defined

### Next steps (priority order)
1. Define a formal CPP-related latent axis from the hidden states (PCA or regression-based direction in latent space). Test cross-fold and cross-subject stability. Entry point: `Scripts/s4_analysis/analysis.py`.
2. Test whether the CPP latent axis provides incremental RT prediction beyond hand-crafted CPP features (CPP amplitude + slope). This is the key unresolved scientific question. Entry point: modify `Scripts/s4_analysis/rt_ridge.py` to accept axis-projected features.
3. Complete DDM drift-rate regression once the latent axis is validated. Entry point: new script under `Scripts/s4_analysis/`.


## 2026-06-26 — Minimal low-rank RNN smoke test

### What changed
- Ran a minimal low-rank RNN smoke-test path across small candidate ranks.
- Retained smoke-test outputs under `Results/low_rank_rnn_smoke/`.
- Compared Rank 2, Rank 3, and Rank 5 on reconstruction and CPP feature checks.

### Main observation
- Rank 5 appeared most promising in the smoke test.
- It had the strongest full-signal test reconstruction among the tested ranks.
- It also showed the clearest CPP-average waveform reconstruction result in the smoke outputs.

### Representative outputs
- `Results/low_rank_rnn_smoke/low_rank_smoke_metrics.csv`
- `Results/low_rank_rnn_smoke/rank_2/`
- `Results/low_rank_rnn_smoke/rank_3/`
- `Results/low_rank_rnn_smoke/rank_5/`

### Interpretation
- The smoke test did not establish a final scientific result.
- It supported treating Rank 5 as the most useful next candidate for a fuller low-rank RNN workflow.


## 2026-06-28 — Rank-5 low-rank RNN notebook updated and executed

### What changed
- Updated the self-contained Rank-5 low-rank RNN notebook.
- Updated the executed notebook copy.
- Added a full Rank-5 workflow that trains the low-rank model, exports compact latent states, generates diagnostics, and runs response-time analyses.

### Current notebook files
- `Scripts/low_rank_rnn_rank5_pipeline.ipynb`
- `Scripts/low_rank_rnn_rank5_pipeline.executed.ipynb`

### Current Rank-5 latent representation
- The exported low-rank latent state has shape `7297 trials x 308 time points x 5 z variables`.
- The five variables are treated as `z1` through `z5`.
- These `z` variables are compact learned summaries of trial-level neural dynamics, not direct CPP components or direct DDM parameters.

### Representative latest run
- Run directory: `tmp/low_rank_r5_notebook_runs/20260628_154318/`
- Latent export: `Data/IntermediateData/latents_low_rank_r5/latents_low_rank_r5.npz` inside that run directory.
- Test metrics and diagnostic outputs were generated inside the same timestamped run directory.

### Interpretation
- The Rank-5 notebook moved the low-rank RNN path from a minimal smoke test to a fuller exploratory workflow.
- The results suggested that Rank-5 `z` variables may carry response-time-relevant information beyond baseline and conventional CPP summaries.
- This remains a cautious modeling result, not proof that the `z` variables are CPP or drift-rate parameters.


## 2026-06-29 — Project guidance updated to prioritize low-rank RNN

### What changed
- Updated `AGENTS.md` to remove obsolete default routing to the old `stage2_YuYNet/` structure.
- Updated `AGENTS.md` so new work defaults to the current root-level project structure.
- Made the low-rank RNN, especially the Rank-5 `z1`-`z5` workflow, the priority main line for normal analysis and follow-up planning.

### Current priority
- Start from `README.md`.
- Then prioritize `low_rank_rnn_drift_rate_followup_plan.md`.
- Use the Rank-5 notebook and low-rank model code as the primary active workflow.
- Treat the earlier GRU workflow as background or comparison unless a task explicitly asks for it.

### Next plan
- Use the Rank-5 `z` variables for follow-up tests of CPP dynamics, response time, and drift-rate-like evidence accumulation.
- Focus on whether `z` explains useful variation beyond subject/task baselines and conventional CPP amplitude/slope summaries.
