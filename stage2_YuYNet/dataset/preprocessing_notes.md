# Preliminary Stage 2 Dataset Notes

## Package level

- This directory is a **preliminary package**, not a formal training-ready package.
- The EEG array comes from `script_pre_EEG/Kosciessa_et_al_2021/temp_data/resp_locked_erp.mat`.
- The reference dataset `van_et_al_2019` is audited only as supporting context and is not used as the main EEG source.

## What is confirmed

- The main EEG tensor is available as three channels and can be arranged to `trial x time x channel`.
- The package channel order is fixed to `CP1`, `CP2`, `CPz` to match the active Stage 2 model contract.
- The available EEG source is response-locked, not stimulus-locked.

## What is inferred

- The exported time axis is inferred from the notebook plotting window `RESP_PRE=-1.0 s` and `RESP_POST=0.2 s`.
- The resulting `times_ms.npy` spans approximately `-1000 ms` to `200 ms`.

## Blocking issues for formal training

- Trial-level `subject_id` is not available in the current repository source files for the chosen EEG tensor.
- Required formal metadata fields are missing: `condition`, `evidence_strength`, `choice`, `correctness`, `RT_ms`, `response_hand`, `artifact_rejection_flag`.
- The current EEG tensor is response-locked, while the formal Stage 2 model contract expects stimulus-locked input.

## Reference-only audit

- `script_pre_EEG/van_et_al_2019/temp_data/data_beh_memory.csv` contains behavioral rows and RT-like values.
- `script_pre_EEG/van_et_al_2019/temp_data/data_resp_locked_memory.csv` is also response-locked and contains only one CPP waveform per trial rather than three channel-resolved signals.
