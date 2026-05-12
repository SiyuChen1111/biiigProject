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
