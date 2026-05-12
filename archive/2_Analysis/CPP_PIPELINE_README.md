# CPP Pipeline Documentation

This document explains the CPP-related code under `2_Analysis/` so that another researcher or engineer can quickly understand:

1. what each file does,
2. why the code is organized this way,
3. how CPP is currently judged,
4. what outputs are produced,
5. and what the important limitations are.

---

## 1. Overall goal

The current pipeline is designed to:

1. read EEG + behavior data from a BIDS-style dataset,
2. create **response-locked epochs**,
3. compute a **CPP-related score** for each epoch,
4. convert that score into a default binary label,
5. export `CSV + NPY` files for downstream modeling,
6. train a simple ANN on the exported epoch tensors.

At the moment, the real data path used by the new code is:

```text
/Users/hyijie/Desktop/biiigProject/1_Data/CPP_low-level-2_Kosciessa_et_al_2021
```

Even though the original plan started from the Manning workflow, the code currently runs on the Kosciessa dataset because that is the dataset that actually exists in the repository in usable form.

---

## 2. File-by-file explanation

### `1_eeg_data_preprocess.ipynb`

### What it does
- This is the **older preprocessing notebook**.
- It was originally intended to read raw EEG `.mat` files and behavior files, clean trials, and write a preprocessed BIDS dataset.

### Why it exists
- It preserves the original preprocessing logic from an earlier stage of the project.
- It is useful as a historical reference for how the first pipeline was supposed to work on the Manning-style data.

### Important note
- This notebook is **not the main entry point for the current working pipeline**.
- The new code does **not** depend on it, because the repository currently contains a ready BIDS dataset in `1_Data/CPP_low-level-2_Kosciessa_et_al_2021/`.

---

### `2_epoch_data.ipynb`

### What it does
- This is the **older epoching/feature extraction notebook**.
- It creates response-locked epochs and computes trial-level EEG features such as:
  - `ams`
  - `pams`
  - `slps`

### Why it exists
- It is the bridge between the original preprocessing workflow and the new modular CPP pipeline.
- It shows the earlier logic for response-locked analysis and helped define the new modular implementation.

### Important note
- It is now mainly a **reference notebook**.
- The new code moves this logic into reusable `.py` modules so that the pipeline is easier to rerun, debug, and explain.

---

### `cpp_paths.py`

### What it does
- Centralizes the important paths used by the CPP pipeline.
- Defines:
  - `DATASET_ROOT`
  - `OUTPUT_ROOT`
  - `CPP_OUTPUT_ROOT`
  - `ANN_OUTPUT_ROOT`
- Also provides `ensure_output_dirs()`.

### Why it exists
- To avoid hard-coding the same paths in multiple files.
- To make the pipeline easier to move or update later.

### Current path assumptions

```python
DATASET_ROOT = PROJECT_ROOT / "1_Data" / "CPP_low-level-2_Kosciessa_et_al_2021"
```

So if the dataset path changes later, this file is the first place to update.

---

### `cpp_epoch_pipeline.py`

This is the **core data-processing file**.

### What it does

It handles the full epoch export pipeline:

1. **discover subjects** from the BIDS dataset,
2. read each subject’s EEG and behavior files,
3. prepare behavior tables,
4. derive a response sample for each trial,
5. create **response-locked epochs**,
6. pick canonical CPP channels,
7. compute CPP-related features,
8. build a metadata table,
9. export all epochs and labels.

### Main functions

#### `discover_subject_records()`
- Finds all available subjects in the dataset.
- Requires each subject to have:
  - `eeg/*_eeg.vhdr`
  - `beh/*_beh.csv`

#### `_prepare_behavior()`
- Reads the behavior CSV.
- Filters to valid trials.
- Removes trials with missing responses.
- Converts columns like `probe_rt` and `stim_onset` into numeric form.
- Computes `response_sample`.

#### `process_subject()`
- Reads a subject’s BrainVision EEG.
- Builds response-locked epochs.
- Extracts epoch tensors for the CPP channel set.
- Computes trial-wise CPP features.

#### `export_cpp_epoch_dataset()`
- Runs the full export process across all subjects.
- Writes all outputs to disk.

### Why it exists
- This file isolates the entire epoch-building workflow in one place.
- That makes the pipeline easier to test than putting everything in a notebook.

---

### `cpp_labeling.py`

This file defines **how CPP score and label are currently computed**.

### What it does

It takes the trial-level metadata/features and adds:

- `cpp_score`
- `cpp_label`
- `cpp_threshold`
- `morphology_gate`
- `task_proxy_label`

### Why it exists
- The scoring and labeling logic is the part that people will most likely want to inspect or change.
- Putting it in a separate file makes the CPP definition explicit and visible.

---

### `cpp_ann.py`

This file handles **ANN training**.

### What it does

1. Loads exported CPP epoch data,
2. flattens epoch tensors,
3. splits data by subject,
4. trains a neural network,
5. evaluates it,
6. saves metrics and predictions.

### Main functions

#### `load_epoch_dataset()`
- Reads:
  - `epoch_metadata.csv`
  - `epoch_tensor.npy`
  - `epoch_label.npy`

#### `split_by_subject()`
- Uses `GroupShuffleSplit` to ensure train/val/test are separated by **subject**, not random trial.

#### `train_ann()`
- Builds a simple pipeline:
  - `StandardScaler`
  - `PCA`
  - `MLPClassifier`

#### `run_ann_training()`
- Runs the end-to-end ANN training and export.

### Why it exists
- To separate “signal definition” from “classifier training”.
- This is important because the classifier should remain replaceable without rewriting epoch extraction.

---

### `2_cpp_epoch_labeling_and_export.ipynb`

### What it does
- This is the notebook entry point for **running the CPP export pipeline**.
- It calls:
  - `ensure_output_dirs()`
  - `export_cpp_epoch_dataset()`

### Why it exists
- Some users prefer notebooks for running and inspecting results.
- This notebook is a light wrapper around the modular code.

### Use it when
- You want to generate fresh CPP epoch outputs.
- You want to inspect label counts or tensor shapes.

---

### `3_ann_training.ipynb`

### What it does
- This is the notebook entry point for **ANN training**.
- It calls `run_ann_training()` and shows metrics/predictions.

### Why it exists
- Same reason as above: it gives a notebook-based entry point while keeping the real logic in `.py` files.

---

## 3. Current CPP judgment standard

This is the most important part of the documentation.

### 3.1 Literature idea behind the implementation

The current CPP logic was designed from the general literature idea that CPP is:

- a **centroparietal positive signal**,
- that **builds up before the response**,
- reaches a high level near the response,
- and then should **fall off after the response**.

So the implementation focuses on three practical ideas:

1. **pre-response buildup**,
2. **late positive amplitude near the response**,
3. **post-response decrease**.

---

### 3.2 Channel choice

The current default CPP channel set is:

```python
DEFAULT_CPP_CHANNELS = ["CP1", "CPz", "CP2", "Pz"]
```

### Why this choice was made
- These are all centroparietal / parietal channels.
- They are close to the canonical CPP topography described in the literature.
- Averaging them reduces single-channel noise.

### Important limitation
- This is still a **manual ROI choice**.
- It is reasonable, but not yet fully validated against the earlier notebook’s exact original channel mapping.

---

### 3.3 Epoch locking strategy

Epochs are currently **response-locked**.

### Current epoch window

Defined in `EpochConfig`:

```python
tmin = -0.6
tmax = 0.2
baseline = (-0.6, -0.4)
```

So each epoch spans:

- **600 ms before response**
- to **200 ms after response**

### Why response-locked?
- Because CPP is often interpreted as a build-to-bound / build-to-response signal.
- Response-locked epochs make the pre-response rise easier to see than stimulus-locked epochs.

### Important limitation
- In the current implementation, response timing is derived mainly from behavior columns:
  - `resp_onset_sample`, if it looks usable,
  - otherwise `stim_onset + probe_rt * sampling_rate`.
- This is practical, but it has **not yet been fully validated against EEG-side response annotations**.

---

### 3.4 Features used to judge CPP

The pipeline computes four trial-level features:

#### 1. `pre_response_slope`
- Computed in the window:

```python
slope_start = -0.25
slope_end = -0.10
```

- This measures whether the waveform is rising before the response.

#### 2. `late_amplitude`
- Computed in the window:

```python
amplitude_start = -0.05
amplitude_end = 0.05
```

- This measures whether the signal is positive around response time.

#### 3. `post_response_drop`
- The code first computes a post-response mean amplitude in:

```python
post_start = 0.05
post_end = 0.15
```

- Then:

```python
post_response_drop = late_amplitude - post_response_amplitude
```

- A positive value means the signal is lower after the response than at response time.

#### 4. `pre_response_monotonicity`
- Computed over:

```python
monotonicity_start = -0.50
monotonicity_end = -0.05
```

- This measures how much the waveform behaves like a generally increasing signal over the pre-response period.

---

### 3.5 How `cpp_score` is computed

First, each feature is **z-scored within subject**:

- `pre_response_slope_z`
- `late_amplitude_z`
- `post_response_drop_z`
- `pre_response_monotonicity_z`

Then the score is:

```python
cpp_score = (
    pre_response_slope_z
    + late_amplitude_z
    + post_response_drop_z
    + 0.5 * pre_response_monotonicity_z
) / 3.5
```

### Why this formula was used
- It gives the main weight to the three most interpretable CPP-like properties:
  - rising before response,
  - positive near response,
  - dropping after response.
- Monotonicity is given smaller weight because it is supportive, not the main definition.

---

### 3.6 How `cpp_label` is computed

The label is not assigned directly from score alone.

The code first builds a **morphology gate**:

```python
morphology_gate = (
    pre_response_slope > 0
    and late_amplitude > 0
    and post_response_drop > 0
)
```

In the real code this is vectorized with pandas, but conceptually it means:

- the signal must be rising before response,
- must be positive near the response,
- and must be lower after the response.

Then a subject-specific threshold is computed:

```python
cpp_threshold = median(cpp_score within subject)
```

Finally:

```python
cpp_label = 1 only if:
    morphology_gate is true
    and cpp_score >= subject median
```

Otherwise:

```python
cpp_label = 0
```

---

## 4. Why this CPP standard was implemented this way

### Practical reason
Single-trial EEG is noisy. A strict fixed voltage cutoff would be unstable across subjects.

### So the code uses three ideas:
1. **shape constraints** via `morphology_gate`,
2. **relative within-subject normalization** via z-scores,
3. **relative thresholding** via subject median.

This makes the heuristic more robust than using one raw amplitude threshold.

---

## 5. Important scientific limitation of the current CPP standard

This point is critical.

The current `cpp_label` is **not an independent biological ground-truth label**.

It is a **heuristic label created from the same EEG waveform that is later used as ANN input**.

That means:

- the ANN is mainly learning to reproduce the heuristic,
- not necessarily discovering an externally validated CPP phenomenon.

In other words, the current workflow is best interpreted as:

> "train a network to approximate the current CPP heuristic"

not yet:

> "train a network on an independent gold-standard CPP label"

This is already useful for engineering and prototyping, but it is **not the same thing as a validated CPP detector**.

---

## 6. Exported outputs

### CPP export outputs

Located in:

```text
2_Analysis/outputs/cpp_epochs/
```

Files:

- `epoch_metadata.csv`
  - one row per epoch
  - includes behavior columns, CPP features, score, label, subject ID, tensor row index

- `epoch_tensor.npy`
  - shape: `(n_epochs, n_channels, n_timepoints)`
  - contains the selected CPP channel epochs

- `epoch_cpp_waveform.npy`
  - channel-averaged CPP waveform per epoch

- `epoch_score.npy`
  - `cpp_score` per epoch

- `epoch_label.npy`
  - `cpp_label` per epoch

- `epoch_times.npy`
  - time axis of the epoch

---

### ANN outputs

Located in:

```text
2_Analysis/outputs/ann/
```

Files:

- `metrics.csv`
- `predictions.csv`
- `mlp_classifier.joblib`

---

## 7. ANN design and why it was done this way

### Current ANN pipeline

The training code currently does:

1. flatten epoch tensors,
2. split data by subject,
3. standardize features,
4. reduce dimensionality with PCA,
5. train an `MLPClassifier`.

### Why subject-wise split was used
- To reduce identity leakage.
- The goal is to avoid having the same subject in both train and test.

### Important limitation
- Even with subject-wise splitting, the target label is still heuristic and derived from the same EEG signal.
- So generalization metrics must be interpreted carefully.

---

## 8. Recommended interpretation for other users

If someone else uses this code, the safest interpretation is:

### This pipeline currently provides
- a reproducible response-locked CPP feature extraction pipeline,
- a transparent heuristic CPP score,
- a default binary label,
- and a prototype ANN baseline.

### This pipeline does **not yet** provide
- a definitive biological CPP ground truth,
- a validated external label,
- or a final publishable CPP classifier.

---

## 9. What should be improved next

The most important next steps are:

1. **Validate response-locking against EEG-side response annotations**
   - not only behavior-derived timing.

2. **Verify the CPP channel ROI**
   - especially against the earlier notebook logic and dataset-specific topography.

3. **Add external validation**
   - for example, condition-based expected differences,
   - or manual review,
   - or labels derived from a stronger experimental criterion.

4. **Avoid circular evaluation**
   - if the ANN is intended to claim real CPP detection rather than heuristic imitation.

---

## 10. Short summary for a new reader

If you only remember one thing, remember this:

> The current code builds response-locked EEG epochs, computes a CPP-like score from centroparietal waveform shape, converts that into a default binary label, and trains an ANN to predict that label. It is a useful working pipeline, but the CPP label is still heuristic rather than gold-standard.
