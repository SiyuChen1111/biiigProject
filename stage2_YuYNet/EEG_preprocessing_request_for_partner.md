# EEG Preprocessing Request for CPP Latent Dynamics Modeling

Hi, I am working on the neural-network modeling and latent/PCA analysis part of the project. To make sure the model input is scientifically interpretable and technically consistent, I would like to ask for the EEG data to be prepared in the following format.

The goal is to model CPP-related single-trial EEG dynamics from **CP1, CP2, and CPz** during a perceptual decision-making task. The model will use a causal GRU to learn a time-varying latent state `z_t`, and we will later test whether this latent state behaves like an evidence accumulation signal.

---

## 1. Main data output requested

Please provide a clean single-trial EEG array:

```text
X: [n_trials, n_timepoints, 3]
channel order: [CP1, CP2, CPz]
```

Preferably also provide an averaged CPP channel as an additional file or array:

```text
X_cpp_avg: [n_trials, n_timepoints, 1]
X_cpp_avg = mean(CP1, CP2, CPz)
```

The three-channel version will be the main model input. The averaged channel will be used as a robustness/control analysis.

---

## 2. Recommended epoch window

Preferred stimulus-locked epoch:

```text
-200 ms to +1200 ms relative to stimulus onset
```

If +1200 ms is not feasible, then:

```text
-200 ms to +1000 ms
```

Please include the exact time vector:

```text
times_ms: [n_timepoints]
```

Example:

```text
times_ms = [-200, -196, -192, ..., 1200]
```

Reason:

The model should learn the temporal evolution from pre-stimulus baseline through the decision period. We will later realign the learned latent states to response time for response-locked analysis. Training directly on response-locked data may introduce circularity and motor-response contamination.

---

## 3. Sampling rate

Please provide the final sampling rate:

```text
fs = ? Hz
```

Recommended if possible:

```text
250 Hz
```

or:

```text
500 Hz
```

Reason:

The neural-network model will predict short future windows of about 50–100 ms. The number of prediction steps depends on the sampling rate.

Examples:

```text
250 Hz:
  50 ms ≈ 12 samples
  100 ms ≈ 25 samples

500 Hz:
  50 ms ≈ 25 samples
  100 ms ≈ 50 samples
```

---

## 4. Trial-level metadata needed

Please provide a trial metadata table with one row per trial. Ideally as `.csv` or `.tsv`.

Required columns:

```text
subject_id
session_id
trial_id
condition
evidence_strength or difficulty
choice
correctness
RT_ms
response_hand
artifact_rejection_flag
```

If available, also include:

```text
stimulus_onset_sample
response_sample
stimulus_onset_time
response_time
block_id
trial_index_in_block
confidence
feedback
```

Reason:

The modeling analysis needs RT, difficulty/evidence strength, choice, correctness, and response hand. RT is especially important because we will convert stimulus-locked latent states into response-locked latent trajectories.

---

## 5. Preprocessing preferences

I do not want to over-specify the EEG preprocessing because you know this part better, but for the modeling to be interpretable, the following would be ideal.

### 5.1 Filtering

Please apply standard filtering appropriate for ERP/CPP analysis.

Possible range:

```text
high-pass: around 0.1 Hz
low-pass: around 30–40 Hz
notch: line noise frequency if needed
```

Reason:

CPP is a slow ERP-like component, so aggressive high-pass filtering may distort the slow buildup. A very high high-pass cutoff should probably be avoided unless there is a strong reason.

### 5.2 Referencing

Please specify the reference used, for example:

```text
average reference
linked mastoids
specific reference electrode
```

Reason:

CPP amplitude and topography can depend on reference choice. We need this information for interpretation and reproducibility.

### 5.3 Artifact handling

Please apply artifact rejection/correction as appropriate, and report what was done.

Useful details:

```text
ICA or other ocular correction?
EOG channels used?
criteria for rejecting trials?
muscle/EMG rejection?
bad channels interpolated?
```

Reason:

The neural model may otherwise learn eye movements, muscle activity, or response artifacts instead of CPP dynamics.

### 5.4 Baseline correction

Preferred:

```text
baseline window: -200 ms to 0 ms
baseline correction applied per trial and channel
```

Please specify whether the data are already baseline-corrected.

Reason:

The model needs consistent baseline handling. Baseline differences can otherwise dominate the latent space.

---

## 6. Trial rejection and quality control

Please provide:

```text
number of total trials
number of retained trials
number of rejected trials
rejection reason if available
```

Preferably per subject/session:

```text
subject_id
n_total_trials
n_good_trials
n_rejected_trials
```

Reason:

The model will use single-trial data. Uneven trial counts or heavy rejection in some subjects may affect training and interpretation.

---

## 7. Response time handling

Please provide RT in milliseconds relative to stimulus onset:

```text
RT_ms = response_time - stimulus_onset_time
```

If trials have no response, too-fast response, or too-slow response, please mark them clearly:

```text
valid_RT_flag
```

Reason:

The model will be trained on stimulus-locked data, but the latent states will be realigned to response time afterwards. We also plan to compute losses mainly before `RT - 50 ms` to reduce contamination from motor execution.

---

## 8. Response hand / motor information

Please include:

```text
response_hand
```

Example values:

```text
left
right
none
```

If the task uses different response keys, please provide the mapping:

```text
key → response hand → choice
```

Reason:

CP1/CP2/CPz are near central areas, and activity close to response time may include motor preparation. We need response-hand information to test whether the learned latent dynamics are mainly decision-related or motor-related.

---

## 9. Output file suggestions

A convenient structure would be:

```text
eeg_cpp_trials.npy
metadata.csv
times_ms.npy
channel_names.txt
preprocessing_notes.md
```

Where:

```text
eeg_cpp_trials.npy:
  shape = [n_trials, n_timepoints, 3]
  channel order = [CP1, CP2, CPz]

metadata.csv:
  one row per trial

times_ms.npy:
  shape = [n_timepoints]

channel_names.txt:
  CP1
  CP2
  CPz

preprocessing_notes.md:
  short description of filtering, referencing, artifact handling, baseline correction, sampling rate, and rejection criteria
```

If data are separated by subject, this is also fine:

```text
sub-01_eeg_cpp_trials.npy
sub-01_metadata.csv
sub-02_eeg_cpp_trials.npy
sub-02_metadata.csv
...
```

---

## 10. Important notes for modeling

### Please keep trials single-trial, not only averaged ERP

The model will be trained on single-trial EEG. We will later average the model outputs to compare generated/reconstructed ERP with real ERP.

Reason:

If we only receive averaged ERP, we cannot learn trial-wise latent dynamics or analyze trial-wise PCA structure.

### Please do not response-align the main training data only

Response-locked data are useful, but the main model input should be stimulus-locked.

Reason:

The model is intended to learn a causal latent process from stimulus onset toward response. If the input is already response-locked, the model may learn response-related structure rather than evidence accumulation.

### Please preserve trial order if possible

Trial order may be useful for checking block effects or slow drifts.

---

## 11. Minimal required version

If time is limited, the minimum I need is:

```text
1. X = [n_trials, n_timepoints, 3] for CP1, CP2, CPz
2. times_ms
3. sampling rate
4. metadata with trial_id, subject_id, condition, difficulty/evidence_strength, choice, correctness, RT_ms, response_hand
5. preprocessing notes
```

---

## 12. Why these requirements matter

The neural model will produce a latent state:

```text
z_t = GRU(x_1, x_2, ..., x_t)
```

We will then test whether `z_t` shows evidence-accumulation-like dynamics:

```text
low-dimensional PCA trajectory
response-locked convergence
different latent velocity for easy vs difficult trials
different trajectory timing for fast vs slow RT trials
control for response-hand/motor effects
```

To make those analyses valid, we need clean single-trial CPP-region EEG, accurate RTs, condition labels, and response-hand information.

Thanks!
