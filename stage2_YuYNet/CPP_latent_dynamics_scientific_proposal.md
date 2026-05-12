# Scientific Proposal: Causal Latent Dynamics of CPP in Perceptual Decision-Making EEG

## Working title

**Learning causal CPP latent dynamics from single-trial EEG to test whether centro-parietal positivity reflects evidence accumulation**

---

## 1. Research motivation

Perceptual decision-making is often modeled as an evidence accumulation process: noisy sensory evidence is integrated over time until a decision is made. In EEG, the centro-parietal positivity (CPP) has been proposed as a candidate neural marker of this accumulation process. Traditional CPP analyses usually rely on averaged ERP waveforms, CPP slope, peak amplitude, or response-locked activity. These analyses are interpretable, but they may miss trial-wise and multidimensional temporal structure.

This project aims to use a causal neural sequence model to learn a trial-wise latent representation of CPP-related EEG activity from CP1, CP2, and CPz. The central goal is not simply to classify trials or generate EEG, but to test whether the learned latent dynamics show properties expected from an evidence accumulation process.

---

## 2. Main scientific question

**Can single-trial CPP-region EEG be represented by a low-dimensional causal latent state whose dynamics are consistent with evidence accumulation during perceptual decision-making?**

More specifically:

1. Does the learned latent state `z_t` show a gradual evolution during the decision period?
2. Does `z_t` become more low-dimensional or more convergent near response time?
3. Do latent trajectories differ systematically by RT, evidence strength/difficulty, choice, or correctness?
4. Can the model reconstruct average CPP-like ERP waveforms from the learned latent state?
5. Are the latent dynamics robust to controls for motor preparation, response alignment, time-index artifacts, and overfitting?

---

## 3. Hypotheses

### H1. CPP latent dynamics are low-dimensional during decision formation

If CP1/CP2/CPz jointly reflect an evidence accumulation process, then the learned latent representation should contain a dominant low-dimensional temporal structure. In PCA space, CPP latent trajectories should be explainable by a small number of PCs, especially during the decision period.

### H2. Latent trajectories converge before response

If the CPP behaves like an accumulation-to-bound signal, response-locked latent trajectories should become more similar across trials as response time approaches.

Expected pattern:

```text
Distance to response-state decreases as t approaches RT.
Participation ratio or n_PC_90 may decrease near response.
```

### H3. Evidence strength modulates latent velocity or trajectory slope

High-evidence or easier trials should show faster movement through latent space, steeper CPP-like buildup, or shorter distance-to-response-state compared with low-evidence or difficult trials.

### H4. RT bins modulate trajectory duration and speed

Fast RT trials should reach the response-state more quickly, while slow RT trials should show a longer or slower latent trajectory.

### H5. The learned latent state should not be fully explained by motor preparation

If the latent structure is mainly evidence accumulation rather than motor execution, response-hand decoding should not fully account for the key latent effects.

---

## 4. Data assumptions

Expected data format:

```text
X: [n_trials, n_timepoints, n_channels]
channels = [CP1, CP2, CPz]
```

Expected metadata per trial:

```text
subject_id
session_id
trial_id
condition / task condition
evidence_strength or difficulty
choice
correctness
RT_ms
response_hand
artifact_rejection_flag
stimulus_onset_time
response_time
```

Recommended EEG epoch:

```text
stimulus-locked epoch: -200 ms to +1200 ms
baseline window: -200 ms to 0 ms
main analysis window: stimulus onset to RT - 50 ms
response-locked analysis window: RT - 600 ms to RT + 100 ms
primary interpretation window: RT - 600 ms to RT - 50 ms
```

The response-aligned window should be used for analysis, not as the main training input.

---

## 5. Recommended model architecture

### 5.1 Main model: causal GRU latent dynamics model

The recommended first model is a **causal, unidirectional GRU encoder**. In PyTorch, this means:

```python
nn.GRU(..., bidirectional=False)
```

This is what we mean by "forward GRU."

The model should not be bidirectional for the primary evidence accumulation analysis because a bidirectional model sees future time points. A BiGRU latent state at time `t` would contain information from after `t`, making it difficult to interpret as the evidence available at time `t`.

### 5.2 Input

```text
X: [batch_size, T, 3]
```

Channel order:

```text
[CP1, CP2, CPz]
```

Optional control input:

```text
X_avg: [batch_size, T, 1]
X_avg = mean(CP1, CP2, CPz)
```

The three-channel input should be the main analysis. The averaged CPP channel should be a robustness check.

---

## 6. Encoder design

Recommended first version:

```text
Input x_t: R^3
    ↓
Linear projection: 3 → 16
    ↓
LayerNorm
    ↓
GRU, hidden_dim = 32, num_layers = 1, bidirectional = False
    ↓
z_t: R^32
```

Suggested hyperparameter grid:

```text
hidden_dim: [16, 32, 64]
input projection dim: [8, 16, 32]
num_layers: 1 initially; 2 only as an ablation
dropout: 0.0 for 1 layer; 0.1 if using 2 layers
```

Recommended default:

```text
projection_dim = 16
hidden_dim = 32
num_layers = 1
```

Reason:

- The input has only three CPP-related channels.
- A small GRU is easier to interpret and less likely to overfit.
- A small latent dimension makes the PCA analysis more stable.
- Larger models may learn subject/session artifacts or time-index structure.

---

## 7. Decoder design

The decoder should be intentionally lightweight. The goal is to force the encoder GRU to learn the temporal dynamics, instead of allowing a powerful decoder to generate the sequence by itself.

### 7.1 Decoder A: current reconstruction decoder

Purpose:

- Ensures `z_t` preserves CPP waveform information.
- Provides a sanity check that the model can reconstruct ERP-like activity.

Architecture:

```text
z_t → MLP(32 → 32 → 3) → x_hat_t
```

Output:

```text
x_hat_current: [batch_size, T, 3]
```

Loss:

```text
L_recon = MSE(x_hat_t, x_t)
```

### 7.2 Decoder B: future prediction decoder

Purpose:

- Main self-supervised learning objective.
- Encourages `z_t` to represent the current state of CPP dynamics and predict upcoming signal evolution.

Architecture:

```text
z_t → MLP(32 → 64 → k * 3)
reshape → [batch_size, T, k, 3]
```

Output:

```text
x_hat_future[t] = predicted EEG from t+1 to t+k
```

Loss:

```text
L_future = MSE(x_hat_{t+1:t+k}, x_{t+1:t+k})
```

Recommended prediction horizon:

```text
50–100 ms
```

Examples:

```text
Sampling rate = 250 Hz:
  1 sample = 4 ms
  k = 12 ≈ 48 ms
  k = 25 ≈ 100 ms

Sampling rate = 500 Hz:
  1 sample = 2 ms
  k = 25 ≈ 50 ms
  k = 50 ≈ 100 ms
```

Recommended first run:

```text
k ≈ 50 ms
```

Then test:

```text
k ≈ 100 ms
```

---

## 8. Optional semi-supervised heads

Semi-supervised heads can help connect the learned latent state to behavior. However, the supervised loss should not dominate the model; otherwise, the latent structure may be overly shaped by labels.

### 8.1 Behavioral targets to consider

Recommended:

```text
evidence_strength / difficulty
RT bin
choice
correctness
```

Use cautiously:

```text
response_hand
```

Response hand should primarily be used as a diagnostic/control variable because strong response-hand information near response time may indicate motor contamination.

### 8.2 Trial-level behavior head

A simple version:

```text
z_summary → behavior prediction
```

Where:

```text
z_summary = mean(z_t over a decision-relevant window)
```

Possible windows:

```text
stimulus-locked: 300–800 ms
response-locked: RT - 300 ms to RT - 50 ms
```

Recommended behavior losses:

```text
choice: cross-entropy
correctness: cross-entropy
difficulty/evidence_strength: cross-entropy or regression loss
RT bin: cross-entropy
RT continuous: Huber loss or MSE after log-transform
```

### 8.3 Time-resolved behavior decoding

For analysis, train lightweight classifiers on frozen `z_t` at each time point:

```text
z_t → choice / RT bin / difficulty / response hand
```

This should be done after model training, preferably using cross-validation.

---

## 9. Recommended loss function

Main loss:

```text
L = L_future
  + λ_recon * L_recon
  + λ_smooth * L_smooth
  + λ_behavior * L_behavior
```

Default values:

```text
λ_recon = 0.3
λ_smooth = 0.001 to 0.01
λ_behavior = 0.1 to 0.5 initially
```

Latent smoothness loss:

```text
L_smooth = mean(||z_t - z_{t-1}||^2)
```

Caution:

- Smoothness regularization should be weak.
- If too strong, it may artificially remove fast decision-related transitions.

Recommended staged training:

```text
Stage 1: self-supervised only
  L = L_future + 0.3 * L_recon + λ_smooth * L_smooth

Stage 2: weak semi-supervised fine-tuning
  L = L_future + 0.3 * L_recon + λ_smooth * L_smooth + λ_behavior * L_behavior

Stage 3: freeze encoder and run PCA/decoding analyses
```

---

## 10. Time window strategy

### 10.1 Recommended training input

Use a stimulus-locked fixed-length window:

```text
-200 ms to +1200 ms relative to stimulus onset
```

or:

```text
-200 ms to +1000 ms
```

The model reads the sequence causally:

```text
baseline → stimulus onset → decision period → response/post-response region
```

### 10.2 Why not train only on response-locked windows?

A response-locked window such as:

```text
RT - 600 ms to RT + 100 ms
```

is useful for analysis, but not ideal as the primary training input.

Reason:

- It gives the model an implicit response-centered structure.
- It may amplify motor preparation or button-press-related components.
- It risks circular interpretation if the goal is to show that latent dynamics approach response time.

### 10.3 Recommended loss mask

Even if the input covers the full epoch, the loss should emphasize the decision period.

For each trial:

```text
valid_t = 0 ms <= t <= RT - 50 ms
```

Main loss should be computed mostly over valid pre-response time points.

Optional weighting:

```text
baseline: low weight
decision period: high weight
post-response: zero or low weight
```

Recommended first version:

```text
L_future and L_recon are computed only for t <= RT - 50 ms
```

This reduces contamination from motor execution, EMG, and post-response processing.

---

## 11. Training pipeline

### 11.1 Data split

Use trial-level splits, but keep subject/session leakage in mind.

Recommended:

```text
train / validation / test = 70% / 15% / 15%
```

If multiple subjects:

1. First run within-subject split for feasibility.
2. Then run subject-level or leave-one-subject-out validation if the goal is cross-subject generalization.

### 11.2 Normalization

Recommended:

```text
baseline correction per trial and channel
then z-score per subject and channel using training-set statistics
```

Important:

- Do not compute normalization statistics using the test set.
- Keep normalization consistent across train/validation/test.

### 11.3 Optimizer

Recommended:

```text
AdamW
learning_rate = 1e-3
weight_decay = 1e-4
batch_size = 64 or 128
gradient clipping = 1.0
early stopping patience = 10–20 epochs
```

### 11.4 Model selection

Select model based on validation self-supervised loss:

```text
validation L_future + 0.3 * validation L_recon
```

Do not select models based on whether the PCA result "looks good." That would bias the analysis.

---

## 12. PCA and latent dynamics analysis

After training, extract:

```text
Z: [n_trials, T, hidden_dim]
Z[:, t, :] = z_t
```

The primary analysis object is:

```text
z_t = causal GRU hidden state at time t
```

Not:

```text
decoder output
BiGRU concatenated state
post-hoc smoothed ERP
```

---

## 13. PCA analysis plan

### 13.1 Global PCA

Fit PCA on all trial-time latent states:

```text
Z_all = reshape(Z, [n_trials * T, hidden_dim])
PCA.fit(Z_all)
```

Then project:

```text
PC_scores = PCA.transform(Z)
```

Use this for:

```text
PC1/PC2 trajectories over time
condition-wise trajectories
RT-bin trajectories
difficulty-bin trajectories
response-locked trajectories
```

Reason:

- Shared PC axes make trajectories comparable across time and conditions.

### 13.2 Time-specific PCA

For each time point:

```text
Z_t = Z[:, t, :]
PCA.fit(Z_t)
```

Compute:

```text
n_PC_80(t)
n_PC_90(t)
participation_ratio(t)
```

Participation ratio:

```text
PR(t) = (sum_i λ_i)^2 / sum_i λ_i^2
```

where `λ_i` are PCA eigenvalues.

Use this for:

```text
time-varying latent dimensionality
response-locked dimensionality changes
near-response low-dimensional convergence
```

### 13.3 Response-locked latent analysis

Using trial RTs, realign latent states:

```text
τ = t - RT
Z_response_locked[trial, τ, hidden_dim]
```

Primary response-locked window:

```text
RT - 600 ms to RT - 50 ms
```

Avoid overinterpreting:

```text
RT - 50 ms to RT + 100 ms
```

because this interval may contain motor execution and button-press artifacts.

---

## 14. Key latent metrics

### 14.1 Latent dimensionality

```text
n_PC_90(t)
participation_ratio(t)
```

Interpretation:

- Decrease near response may indicate latent convergence.
- But this must be checked against motor and time-index controls.

### 14.2 Latent trajectory

Use global PCA:

```text
PC1(t), PC2(t), PC3(t)
```

Plot by:

```text
fast vs medium vs slow RT
high vs low evidence
correct vs error
choice A vs choice B
```

### 14.3 Latent velocity

```text
v_t = ||z_t - z_{t-1}||
```

or in PC space:

```text
v_t = ||PC_t - PC_{t-1}||
```

Expected:

```text
higher evidence → faster latent movement
fast RT → earlier/faster approach to response-state
```

### 14.4 Distance to response-state

Define:

```text
z_response_state = mean(z_t from RT - 100 ms to RT - 50 ms)
```

Then:

```text
d_t = ||z_t - z_response_state||
```

Expected:

```text
d_t decreases as response approaches
fast RT trials approach response-state earlier
```

### 14.5 Time-resolved decoding

Train simple classifiers/regressors on frozen `z_t`:

```text
z_t → choice
z_t → RT bin
z_t → evidence strength
z_t → correctness
z_t → response hand
```

Use cross-validation. The goal is not just high accuracy; the timing of information emergence matters.

---

## 15. Required control analyses

### 15.1 Time-index control

Problem:

The GRU may simply learn "where we are in the trial" rather than evidence accumulation.

Control:

```text
time-only baseline:
input = time index
output = expected ERP or condition-average ERP
```

Compare whether the GRU latent explains trial-wise behavior or condition differences beyond time.

### 15.2 Response-hand control

Problem:

Near-response latent features may reflect motor preparation.

Controls:

```text
response-hand decoding from z_t
separate analyses for left-hand and right-hand responses
CPz-only model
CP1/CP2 lateralization analysis
```

### 15.3 Average-channel control

Compare:

```text
three-channel model: CP1, CP2, CPz
average-channel model: mean(CP1, CP2, CPz)
CPz-only model
```

If results are robust across these versions, the interpretation is stronger.

### 15.4 Shuffle controls

Useful shuffles:

```text
shuffle RT labels
shuffle condition labels
phase-randomized EEG
trial-order shuffle
```

The key latent-response relationships should weaken under appropriate shuffles.

### 15.5 Unsupervised vs semi-supervised comparison

Run both:

```text
self-supervised model only
semi-supervised model
```

If the main PCA/trajectory results appear in both, the claim is stronger.

---

## 16. Recommended first implementation

### Model

```text
Input: [B, T, 3]
Encoder:
  Linear(3 → 16)
  LayerNorm
  GRU(hidden_dim = 32, bidirectional = False)

Latent:
  Z = [B, T, 32]

Decoder 1:
  MLP(32 → 32 → 3)
  current reconstruction

Decoder 2:
  MLP(32 → 64 → k*3)
  future prediction
  k ≈ 50 ms initially

Loss:
  L = L_future + 0.3 * L_recon + 0.001~0.01 * L_smooth

Mask:
  compute main loss for t <= RT - 50 ms
```

### Training

```text
epochs: up to 100–200
early stopping: validation loss
optimizer: AdamW
lr: 1e-3
weight_decay: 1e-4
batch size: 64 or 128
gradient clipping: 1.0
```

### Analysis

```text
1. reconstruct average ERP
2. extract Z = [trials, time, hidden_dim]
3. global PCA trajectories
4. time-specific PCA / participation ratio
5. response-locked latent convergence
6. RT-bin and difficulty-bin trajectories
7. response-hand decoding control
8. comparison with CPP average-channel baseline
```

---

## 17. Expected figures

### Figure 1. Model architecture

```text
CP1/CP2/CPz → causal GRU encoder → z_t → reconstruction/future-prediction decoder
```

### Figure 2. ERP reconstruction sanity check

```text
Real ERP vs reconstructed ERP
channels: CP1, CP2, CPz, and average CPP
conditions: overall, RT bins, difficulty bins
```

### Figure 3. Global PCA latent trajectories

```text
PC1-PC2 trajectory
fast vs slow RT
high vs low evidence
correct vs error
```

### Figure 4. Time-varying dimensionality

```text
n_PC_90(t)
participation_ratio(t)
stimulus-locked and response-locked
```

### Figure 5. Latent convergence to response-state

```text
distance_to_response_state over time
RT-bin comparison
difficulty-bin comparison
```

### Figure 6. Time-resolved decoding

```text
z_t → choice
z_t → RT bin
z_t → difficulty
z_t → response hand
```

Response-hand decoding is interpreted as a motor contamination diagnostic.

---

## 18. Interpretation logic

A strong result would look like:

1. The model reconstructs CPP-like average ERP waveforms.
2. The learned latent state has a small number of dominant PCs.
3. Response-locked latent trajectories converge before response.
4. High-evidence trials show faster latent movement or steeper latent buildup.
5. Slow RT trials show delayed or prolonged latent trajectories.
6. The pattern remains after controlling for response hand and time-index artifacts.
7. Similar patterns appear in a self-supervised model and a weakly semi-supervised model.

This would support the conclusion that CP1/CP2/CPz jointly contain a latent temporal structure consistent with evidence accumulation.

A weaker result would look like:

1. ERP reconstruction works, but latent dynamics are explained mostly by time index.
2. Near-response structure is dominated by response-hand decoding.
3. PCA patterns disappear after RT-label shuffling or response-hand controls.
4. Results only appear when strong behavior supervision is used.

This would suggest that the learned representation may not be strong evidence for CPP as a decision accumulation marker.

---

## 19. Possible extensions after the first model

### 19.1 BiGRU teacher, causal GRU student

Train a BiGRU autoencoder as a stronger representation teacher, then train a causal GRU student to approximate its useful latent representation without access to future time points.

### 19.2 Conditional ERP generation

Condition the decoder on:

```text
difficulty
RT bin
choice
correctness
```

Then generate condition-specific average ERP.

### 19.3 Cross-task generalization

Apply the same architecture and analysis pipeline to other perceptual decision-making EEG datasets.

Strong cross-task evidence would show that the learned CPP latent dynamics are not specific to one task's ERP shape.

### 19.4 Multi-task shared encoder

Train a shared causal encoder across multiple decision tasks with task-specific decoders.

---

## 20. References

1. O'Connell, R. G., Dockree, P. M., & Kelly, S. P. (2012). **A supramodal accumulation-to-bound signal that determines perceptual decisions in humans.** *Nature Neuroscience*. https://www.nature.com/articles/nn.3248

2. Loughnane, G. M., Newman, D. P., Bellgrove, M. A., Lalor, E. C., Kelly, S. P., & O'Connell, R. G. (2014). **Target selection signals influence perceptual decisions by modulating the onset and rate of evidence accumulation.** *Current Biology*. https://www.cell.com/current-biology/fulltext/S0960-9822(14)00963-7

3. Murphy, P. R., Robertson, I. H., Harty, S., & O'Connell, R. G. (2015). **Neural evidence accumulation persists after choice to inform metacognitive judgments.** *eLife*. https://elifesciences.org/articles/11946

4. van Vugt, M. K., Beulen, M. A., & Taatgen, N. A. (2019). **Relation between centro-parietal positivity and diffusion model parameters in both perceptual and memory-based decision making.** *Brain Research*. https://www.sciencedirect.com/science/article/pii/S0006899319301386

5. Tagliabue, C. F., Mazzi, C., Bagattini, C., & Savazzi, S. (2019). **The EEG signature of sensory evidence accumulation during decision formation closely tracks subjective perceptual experience.** *Scientific Reports*. https://www.nature.com/articles/s41598-019-41024-4

6. Chien, H.-Y. S., Goh, H., Sandino, C. M., & Cheng, J. Y. (2022). **MAEEG: Masked Auto-encoder for EEG Representation Learning.** arXiv. https://arxiv.org/abs/2211.02625

7. Pandarinath, C., O'Shea, D. J., Collins, J., et al. (2018). **Inferring single-trial neural population dynamics using sequential auto-encoders.** *Nature Methods*. https://www.nature.com/articles/s41592-018-0109-9

---

## 21. Short summary

The recommended first-pass model is a causal GRU encoder trained with self-supervised future prediction and current reconstruction. The main latent state is the forward GRU hidden state `z_t`. After training, `z_t` is analyzed with global PCA, time-specific PCA, response-locked trajectory convergence, latent velocity, distance-to-response-state, and time-resolved decoding. The key scientific claim should focus on whether CPP-region EEG contains a low-dimensional, causal latent dynamic consistent with evidence accumulation, not merely on whether the model can generate EEG.
