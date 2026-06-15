# Fixed Stage 2/3 Trial-Level Dataset

## Data sources

- EEG source: `script_pre_EEG/Kosciessa_et_al_2021/temp_data/resp_locked_erp.mat`.
- Behavior source: `script_pre_EEG/Kosciessa_et_al_2021/temp_data/behavior_data_all.csv`.
- The notebook export was fixed so each subject block is appended inside the subject loop.

## Alignment

- The exported EEG contains 41 subject blocks.
- Subject block trial counts match `behavior_data_all.csv` subject trial counts before invalid-trial removal.
- Metadata rows are built in the same subject and trial order as the EEG blocks.
- Trials containing non-finite EEG values are removed together with their metadata rows.

## Window and sampling

- This dataset is response-locked, not stimulus-locked.
- Response-locked window: approximately -1000 ms to 200 ms.
- Sampling rate: 256 Hz.
- Time points: 308.

## Channels

- Channel order: CP1, CP2, CPz.
- Saved EEG shape: trial x time x channel = (7297, 308, 3).

## Trial inclusion

- Input EEG/behavior trial pairs before non-finite EEG removal: 10258.
- Removed trials because the response-locked EEG window contained NaN or inf: 2961.
- `RT_ms = probe_rt * 1000`.
- `correctness = probe_accuracy`.
- `condition = probe_attribute`.
- `difficulty` and `evidence_strength` currently use `cue_dimensionality` as a condition proxy.
- `choice` and `response_hand` use `probe_leftrightwin`.
- `artifact_rejection_flag = 0` because no separate artifact flag is available in the exported behavior table.

## Interpretation boundary

- This data is suitable for response-proximal CPP latent analysis.
- It should not be treated as pure stimulus-locked evidence accumulation.
- A strict stimulus-locked accumulation analysis requires an additional stimulus-locked export.
