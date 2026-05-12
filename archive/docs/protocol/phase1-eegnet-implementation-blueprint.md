# Phase-1 EEGNet Implementation Blueprint for Kosciessa `task-dynamic`

## 1. Goal

The goal of phase 1 is to test whether EEGNet can learn a **CPP-like trial representation that generalizes across unseen subjects within the same task**.

This phase is **not** intended to prove cross-task transfer, confirm the full theory of CPP, or establish that the model has learned a uniquely canonical evidence-accumulation signal.

The deliverable of phase 1 is a reproducible training and evaluation pipeline for:

- one dataset: `CPP_low-level-2_Kosciessa_et_al_2021`
- one task family: `task-dynamic`
- one model family: `EEGNet`
- one generalization target: **subject-held-out classification**

---

## 2. High-Level Design

### Core question

Can EEGNet classify **high CPP-like** versus **low CPP-like** response-locked EEG epochs in a way that generalizes to unseen subjects?

### Why this framing

The current dataset contains:

- continuous EEG
- trial-aligned behavior tables
- response timestamps
- subject-level grouping

but it does **not** contain an explicit ground-truth CPP label. Therefore, the safest first implementation is:

1. derive a **CPP-like score** from interpretable EEG features,
2. turn that score into a clean binary target,
3. train EEGNet on the resulting binary labels,
4. evaluate whether the learned representation transfers across subjects.

---

## 3. Source Data and Required Inputs

### Dataset root

Preferred:

`data/raw/CPP_low-level-2_Kosciessa_et_al_2021/`

Backward-compatible fallback during the current transition:

`1_Data/CPP_low-level-2_Kosciessa_et_al_2021/`

### Key files

- `processed_behavior.csv`
- `trial_exclusion_summary.csv`
- `participants.tsv`
- `sub-xxxx/beh/sub-xxxx_task-dynamic_beh.csv`
- `sub-xxxx/eeg/sub-xxxx_task-dynamic_eeg.eeg`
- `sub-xxxx/eeg/sub-xxxx_task-dynamic_eeg.vhdr`
- `sub-xxxx/eeg/sub-xxxx_task-dynamic_events.tsv`
- `sub-xxxx/eeg/sub-xxxx_task-dynamic_channels.tsv`

### Verified dataset facts

- recording type: continuous EEG
- sampling rate: 256 Hz
- channels in raw file metadata include scalp EEG plus auxiliary channels, but the phase-1 implementation uses **60 scalp EEG channels** after excluding `A1`, EOG, and ECG
- task name in metadata: `dynamic`
- per-subject valid trials: approximately 256 before additional exclusions

---

## 4. Proposed File-Level Pipeline

Implement the phase-1 system as four logical layers.

### Layer A: dataset assembly

Responsibility:

- read behavior tables and EEG files
- align subject-level behavior rows to EEG sample indices
- produce one trial table per subject

#### Exact join rule

For phase 1, the behavior table is the primary trial table.

For each subject:

1. load `sub-xxxx/beh/sub-xxxx_task-dynamic_beh.csv`
2. load `sub-xxxx/eeg/sub-xxxx_task-dynamic_events.tsv`
3. keep one behavior row as one candidate trial
4. use `stim_onset` and `resp_onset_sample` from the behavior row as the authoritative trial anchors
5. use `events.tsv` only as a consistency check, not as the primary join table

#### Consistency check rule

For each retained trial:

- confirm that `resp_onset_sample` is not null
- confirm that `resp_onset_sample` lies within the EEG recording range
- optionally confirm there is at least one event in `events.tsv` within ±5 samples of `resp_onset_sample`

If the behavior row has a valid `resp_onset_sample` but the nearby event marker is missing, keep the trial and log a warning rather than dropping it. This avoids turning the noisier event stream into the primary trial definition.

Suggested outputs:

- `subject_id`
- `trial_index`
- `stim_onset`
- `resp_onset_sample`
- `probe_accuracy`
- `probe_rt`
- `is_valid_trial`
- `is_missing_response`
- `is_rt_outlier`
- condition columns such as `probe_attribute`, `cue_dimensionality`, `stim_code`

### Layer B: epoch extraction

Responsibility:

- load continuous EEG
- keep only EEG channels
- create response-locked trial epochs
- return epochs in consistent shape

Suggested output shape before model formatting:

`(trials, chans, samples)`

### Layer C: label construction

Responsibility:

- compute CPP-like score from centroparietal channels
- convert score into high-vs-low binary labels
- optionally drop ambiguous middle-score trials

### Layer D: modeling and evaluation

Responsibility:

- format inputs for EEGNet
- split by subject
- train EEGNet
- save best validation checkpoint
- evaluate on unseen subjects

---

## 5. Data Cleaning Rules

The first implementation should use conservative filtering.

### Keep only trials satisfying all of the following

- `is_valid_trial == True`
- `is_missing_response == False`
- `is_rt_outlier == False`
- `resp_onset_sample` exists

### Channel policy

Use only the 60 scalp EEG channels for the phase-1 model input.

Exclude:

- `VEOG`
- `HEOGL`
- `HEOGR`
- `ECG`

### Subject policy

Start with all subjects that pass the dataset-level quality summaries.

If later preprocessing reveals subject-level loading or alignment failures, exclude those subjects explicitly and log the reason.

### Fixed EEG preprocessing rule for phase 1

Use the following exact preprocessing sequence for every subject before epoch extraction:

1. load raw EEG with preload enabled
2. keep only the 60 scalp EEG channels
3. apply average reference across the retained EEG channels
4. apply band-pass filter from **1.0 Hz to 30.0 Hz**
5. do **not** run ICA in phase 1
6. do **not** interpolate bad channels in phase 1 unless a channel is explicitly marked unusable in the source metadata
7. rely on the existing trial-level filtering (`is_valid_trial`, `is_missing_response`, `is_rt_outlier`) rather than adding a second custom epoch-rejection rule in phase 1

Why this fixed choice:

- it stays close to the local dataset channel metadata, which already reports low/high cutoffs of 1.0 and 30.0 Hz
- it avoids introducing another layer of subjective preprocessing decisions
- it keeps the first experiment deterministic and easy to audit

---

## 6. Epoch Definition

### Epoch anchor

Use **response-locked** epochs anchored on `resp_onset_sample`.

### First recommended window

- start: `-0.6 s`
- end: `+0.2 s`

At 256 Hz, that corresponds to:

- 0.8 s total window
- 205 samples if kept at 256 Hz

### Sampling-rate decision

For the first pass, resample epochs from **256 Hz to 128 Hz** before EEGNet.

Why:

- current local EEGNet defaults are written around 128 Hz assumptions
- this reduces architecture uncertainty in phase 1
- it makes the temporal scale more compatible with the existing EEGNet hyperparameters

After resampling, the epoch becomes approximately:

- 0.8 s total window
- about 102 or 103 samples depending on implementation

### Baseline handling

Use **one fixed baseline rule** for both scoring and model input.

#### Chosen baseline rule for phase 1

- subtract the mean voltage in the window **-600 ms to -400 ms** relative to response

Why this window:

- it stays inside the epoch
- it is early enough to avoid the target buildup and response-near peak windows
- it is simple and reproducible

Important: the same baseline-corrected epoch must be used both for CPP-score construction and for EEGNet input.

---

## 7. CPP-Relevant Channel Set for Scoring

The dataset includes the CPP ROI channels used in phase 1:

- `CPz`
- `CP1`
- `CP2`

### Recommended use

- use **all 60 scalp EEG channels** as model input
- use **`CPz`, `CP1`, `CP2`** as the primary region of interest for score construction and physiological validation

This keeps the classifier expressive while preserving a focused interpretation of what “CPP-like” means.

---

## 8. Label Construction

### Step 1: compute a CPP-like score per trial

For each trial, compute the average ERP across the CPP ROI (`CPz`, `CP1`, `CP2`), then derive three features.

#### Feature A: AMS

Average amplitude in a pre-response window.

Fixed phase-1 window:

- `-180 ms to -80 ms`

#### Feature B: PAMS

Peak amplitude around the response.

Fixed phase-1 window:

- `-50 ms to +50 ms`

#### Feature C: SLPS

Linear slope in the pre-response buildup period.

Fixed phase-1 window:

- `-250 ms to -50 ms`

#### Exact slope computation rule

Compute `SLPS` by fitting a least-squares linear regression to the ROI-averaged EEG amplitude over all sample points inside `-250 ms to -50 ms`, using time in **seconds** as the predictor and voltage in **baseline-corrected microvolts** as the response. The fitted regression coefficient is the slope feature.

### Step 2: standardize features

Within the training fold only:

- z-score AMS
- z-score PAMS
- z-score SLPS

Then apply the same training-fold parameters to validation and test data.

### Step 3: combine into one score

Fixed phase-1 definition:

`cpp_score = z(AMS) + z(PAMS) + z(SLPS)`

This keeps the label definition simple, interpretable, and aligned with the intended CPP shape.

### Step 4: convert to binary labels

Recommended first-pass rule:

- top 25% of `cpp_score` -> label `1` (high CPP-like)
- bottom 25% of `cpp_score` -> label `0` (low CPP-like)
- middle 50% -> exclude from phase-1 training

#### Exact thresholding procedure

Thresholds must be computed **within the training subjects of each fold only**.

Then:

1. fit the 25th and 75th percentile thresholds on the training-fold `cpp_score`
2. apply those thresholds unchanged to validation and test subjects
3. drop validation/test trials that fall into the middle region as well

This prevents label leakage from validation or test subjects into the training rule.

Why:

- cleaner labels
- lower ambiguity
- better first-pass training stability

Later phases can use all trials or continuous-score prediction.

---

## 9. Subject-Held-Out Split Strategy

### Non-negotiable rule

Split by **subject**, never by trial.

### Recommended primary strategy

Use **grouped 5-fold cross-validation** with:

- `group = subject_id`

#### Exact grouped-fold construction rule

1. sort unique subject IDs alphabetically
2. assign subjects to folds by round-robin order over the sorted list
3. use one fold as test and the remaining four folds as the outer training pool

This makes fold construction deterministic even before a dedicated grouped splitter helper is written.

Within each outer fold:

1. hold out one subject group block as test
2. split the remaining training subjects into train/validation at the **subject level**
3. fit all normalization statistics on training subjects only
4. train on training subjects
5. select checkpoint on validation subjects
6. report final metrics on unseen test subjects

#### Exact validation split rule

Within each outer fold:

- use **80% of remaining subjects for train**
- use **20% of remaining subjects for validation**
- assign subjects to train/validation by sorted subject ID for deterministic reproducibility, unless a seeded grouped splitter is later introduced

#### Exact operational train/validation rule

After removing the test-fold subjects:

1. sort the remaining subject IDs alphabetically
2. compute `n_val = max(1, round(0.2 * n_remaining_subjects))`
3. use the first `n_val` sorted subjects as validation
4. use all remaining subjects as training

This gives one exact operational rule for phase 1 and avoids leaving validation ambiguous.

### Why not random trial split

Random trial splitting lets the model exploit subject-specific structure and gives an overly optimistic estimate of generalization.

---

## 10. Model Architecture Decision

Use the current transfer-ready `EEGNet` implementation in `EEGModels_PyTorch.py`.

### Why this is the correct phase-1 choice

- `forward()` returns logits, which is correct for `CrossEntropyLoss`
- `forward_features()` exists for later analysis and transfer
- backbone and classifier head are already separated
- no extra architectural complexity is required to use it as a normal classifier now

### What to use now

- one fixed 2-class head
- normal supervised training

### What to defer

- `reset_classifier()` workflows
- linear probing
- frozen-backbone transfer schedules

Those belong to later cross-dataset or cross-task phases.

---

## 11. Input Formatting for EEGNet

The current EEGNet accepts:

- `(batch, Chans, Samples, 1)`
- `(batch, Chans, Samples)`

For consistency with `ERP_PyTorch.py`, use:

`(trials, chans, samples, 1)`

### Example

If response-locked epochs are stored as:

`X.shape == (N, 60, T)`

then reshape to:

`X = X.reshape(N, 60, T, 1)`

This keeps the training script close to existing local patterns.

---

## 12. Training Skeleton to Reuse from `ERP_PyTorch.py`

The following training components can be reused almost directly:

- `TensorDataset`
- `DataLoader`
- `CrossEntropyLoss`
- `Adam`
- batch training loop
- validation loop
- best-validation checkpoint saving
- `model.apply_max_norm_constraints()` after optimizer step

### Parts that must be rewritten

- data loading source
- epoch extraction logic
- label construction
- split logic
- evaluation metrics

#### Non-reusable assumption from `ERP_PyTorch.py`

Do **not** reuse its simple sequential dataset slicing (`X[0:144]`, etc.). That script is only a training-loop template. All splitting for phase 1 must be subject-grouped.

### Pseudocode skeleton

```python
# 1. build trial table and extract epochs
X, y, subjects = build_phase1_dataset(...)

# 2. grouped folds by subject
for fold_idx, (train_subjects, test_subjects) in grouped_subject_split(subjects):
    X_train, y_train = ...
    X_val, y_val = ...
    X_test, y_test = ...

    # 3. format for EEGNet
    X_train = X_train.reshape(n_train, chans, samples, 1)
    X_val = X_val.reshape(n_val, chans, samples, 1)
    X_test = X_test.reshape(n_test, chans, samples, 1)

    # 4. model
    model = EEGNet(
        nb_classes=2,
        Chans=chans,
        Samples=samples,
        dropoutRate=0.5,
        kernLength=32,
        F1=8,
        D=2,
        F2=16,
        dropoutType='Dropout',
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 5. training loop
    for epoch in range(num_epochs):
        train_one_epoch(...)
        val_metrics = evaluate(...)
        save_best_checkpoint_if_needed(...)

    # 6. final test on unseen subjects
    test_metrics = evaluate(...)
    collect_fold_results(test_metrics)
```

---

## 13. First-Pass Hyperparameters

These are not final truths. They are a stable starting point.

### Model

- `nb_classes = 2`
- `dropoutRate = 0.5`
- `F1 = 8`
- `D = 2`
- `F2 = 16`
- `dropoutType = 'Dropout'`

### Temporal settings

If using 128 Hz-resampled epochs:

- start with `kernLength = 32`

### Optimization

- optimizer: `Adam`
- learning rate: `1e-3`
- batch size: `16`
- epochs: `100` to `150`

### Checkpoint rule

- keep best model by validation balanced accuracy if implemented
- otherwise by validation accuracy as a fallback

---

## 14. Metrics to Report

Do not rely on plain accuracy alone.

### Required metrics

- balanced accuracy
- ROC-AUC
- precision
- recall
- per-fold test metrics
- per-subject test metrics

#### Exact sparse-class handling rule

- If a validation or test split contains only one class after middle-score exclusion, compute and report:
  - balanced accuracy
  - precision
  - recall
  - confusion counts
- In that case, record ROC-AUC as `NA` for that split instead of forcing a value.

The final cross-fold summary should report:

- mean and standard deviation for balanced accuracy
- mean and standard deviation for precision and recall
- mean ROC-AUC over folds where both classes are present
- the number of folds with undefined ROC-AUC

### Why balanced accuracy matters

The binary labels are formed by score thresholds and may still become imbalanced after exclusions.

---

## 15. What Counts as Success in Phase 1

### Minimum acceptable success

- held-out subject performance consistently above chance
- no evidence that results depend on a few extreme subjects only

### Strong phase-1 success

- mean test balanced accuracy is stably in a useful range across folds
- fold-to-fold variance is not extreme
- predicted positive trials show more CPP-like physiology than predicted negative trials in the ROI

### What does not count as success

- only random-trial split performance
- high average accuracy with severe subject-level collapse
- good classifier metrics with no interpretable CPP-related waveform/topography difference

---

## 16. Required Result Figures

At minimum, generate these outputs.

### Figure 1: fold-wise performance

- balanced accuracy by fold
- AUC by fold

### Figure 2: per-subject performance

- each unseen test subject as one point or bar

### Figure 3: response-locked ERP at ROI channels

- compare predicted positive vs predicted negative trials
- channels: `CPz`, `CP1`, `CP2`

### Figure 4: scalp topography near response

- compare predicted positive vs predicted negative average spatial pattern

These figures help determine whether the classifier learned a plausible CPP-related signal.

---

## 17. Suggested Code Organization

One clean phase-1 structure would be:

### `src/data/data_kosciessa.py`

Responsibilities:

- read BIDS-style subject files
- load behavior and EEG
- align trial rows to sample indices
- return cleaned subject-level trial tables

### `src/preprocessing/epoching.py`

Responsibilities:

- extract response-locked EEG epochs
- keep EEG channels only
- optional resampling to 128 Hz
- return `(trials, chans, samples)` arrays

### `src/features/labels_cpp.py`

Responsibilities:

- compute AMS / PAMS / SLPS
- build `cpp_score`
- threshold score into binary labels

### `scripts/train_phase1_eegnet.py`

Responsibilities:

- grouped subject splits
- train / validate / test loop
- save best checkpoint
- save metrics and plots

### `src/evaluation/evaluate_phase1.py`

Responsibilities:

- aggregate fold metrics
- compute per-subject summaries
- generate ERP and topography plots

---

## 18. Main Risks

### Risk 1: labels are too noisy

If the CPP-like score does not correspond well to meaningful waveform structure, EEGNet may learn unstable targets.

### Risk 2: model learns subject or preprocessing artifacts

This is why subject-held-out splitting is mandatory.

### Risk 3: architecture and sampling assumptions mismatch

If epoch sampling or window length is inconsistent, the temporal assumptions of EEGNet become less reliable.

### Risk 4: success is overinterpreted

Even strong same-task cross-subject results do **not** prove cross-task transfer or settle the full CPP theory.

---

## 19. Phase-1 Exit Condition

Phase 1 is complete when all of the following are true:

1. subject-held-out EEGNet training runs end-to-end on Kosciessa `task-dynamic`
2. labels are constructed reproducibly from the CPP-like score
3. fold-wise test metrics are available
4. per-subject test results are available
5. predicted positive vs negative trials show interpretable ERP differences in the CPP ROI

Only after that should the project move to:

- cross-dataset testing
- classifier reset and transfer workflows
- alternative label definitions
- more ambitious transfer-learning stages

---

## 20. One-Sentence Summary

Phase 1 should implement a **subject-held-out, response-locked, CPP-score-derived binary EEGNet classifier** on Kosciessa `task-dynamic`, using the existing transfer-ready EEGNet architecture and the local `ERP_PyTorch.py` training skeleton as the execution template.
