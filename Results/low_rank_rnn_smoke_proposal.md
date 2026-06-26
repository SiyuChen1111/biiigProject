# Low-Rank Recurrent Dynamics for Interpretable Single-Trial CPP Modeling in Human EEG

## Background and Motivation

The current project asks whether neural-network representations learned from
single-trial EEG can preserve centro-parietal positivity (CPP) structure and
carry behaviorally relevant information about response time. The existing
response-locked GRU model already provides a strong proof of feasibility: it
reconstructs CPP-related EEG from CP1, CP2, and CPz with high fidelity, and its
hidden states retain substantial CPP amplitude information.

However, the current GRU hidden state is 32-dimensional. This is useful for
prediction and reconstruction, but it is not yet an ideal object for mechanistic
interpretation. In particular, it remains difficult to say whether the model has
learned a compact evidence-accumulation process, whether response-proximal CPP
activity follows a small number of latent trajectories, or whether fast and slow
responses differ along a clear dynamical axis.

Low-rank recurrent neural networks offer a natural next step. They constrain
recurrent dynamics to a small number of latent variables, making it possible to
ask whether CPP can be modeled as a compact dynamical process rather than only
as a high-dimensional neural reconstruction problem.

## Research Question

Can a low-rank recurrent neural network recover interpretable low-dimensional
dynamics underlying single-trial response-locked CPP in human EEG?

More specifically, we ask whether a rank-2, rank-3, or rank-5 recurrent model
can:

1. preserve the empirical CPP waveform and trial-level CPP features;
2. expose low-dimensional trajectories that separate fast versus slow responses,
   correct versus error trials, or difficulty levels;
3. provide a more interpretable candidate mechanism than the existing
   32-dimensional GRU hidden state.

## Existing Project Foundation

The project already contains the necessary data and baseline results:

- EEG input: response-locked CP1, CP2, and CPz signals;
- dataset size: 7,297 retained trials, 308 time points, 41 participants;
- existing model: causal forward GRU trained without behavior labels;
- current validation: strong CPP reconstruction and strong hidden-state
  encoding of CPP amplitude;
- current limitation: behavioral interpretation remains cautious, and the
  high-dimensional hidden state is not yet a compact mechanistic model.

This makes the low-rank experiment a targeted extension rather than a new
project from scratch.

## Proposed Low-Rank RNN Approach

We will add an exploratory low-rank recurrent baseline that keeps the recurrent
state itself low-dimensional. The model receives the same CP1, CP2, and CPz EEG
input as the existing GRU and is trained with the same broad self-supervised
goal: reconstruct the current EEG signal and preserve CPP-related temporal
structure.

The key difference is that the latent state has rank `R`, where the first smoke
test scans `R = 2, 3, 5`. These states will be directly analyzed as the model's
candidate CPP dynamics. This design supports a simple interpretation: if a
rank-3 model can preserve CPP and produce meaningful trajectories, then the
model suggests that a small number of latent variables may be sufficient to
describe response-proximal CPP dynamics.

This first version will be a lightweight PyTorch low-rank recurrent baseline,
not a full stochastic low-rank RNN inference model. The stochastic formulation
is a plausible future direction if the smoke test produces promising structure.

## Smoke Test Design

The smoke test is intentionally exploratory and will be stored separately from
the existing GRU results.

Planned settings:

- input data: `Data/ProcessedData`;
- output directory: `Results/low_rank_rnn_smoke`;
- ranks: 2, 3, and 5;
- training budget: 10-20 epochs per rank;
- data budget: a fixed subject-balanced subsample for quick CPU execution;
- evaluation focus: CPP fidelity and low-dimensional trajectory clarity.

The smoke test will save:

- per-rank training history;
- per-rank CPP reconstruction metrics;
- low-rank latent states for the test split;
- average CPP reconstruction plots;
- latent trajectory plots grouped by response time, correctness, and difficulty.

## Planned Analyses

The first analysis asks whether low-rank models preserve CPP structure:

- compare empirical and reconstructed average CPP waveforms;
- compute CPP amplitude and slope agreement in standard pre-response windows;
- compare full-signal reconstruction only as a secondary check.

The second analysis asks whether the latent state is interpretable:

- plot mean low-dimensional trajectories by fast, medium, and slow RT groups;
- plot trajectories by correct versus error trials;
- plot trajectories by difficulty level;
- inspect whether one latent direction behaves like a response-proximal build-up
  axis.

The third analysis is a cautious behavioral check:

- test whether window-averaged low-rank states show preliminary association
  with RT or correctness;
- avoid treating behavior prediction as the main smoke-test success criterion.

## Expected Outcomes

A positive smoke result would not prove a full evidence-accumulation mechanism.
It would show that low-rank recurrent dynamics are worth pursuing because they
can preserve CPP while exposing a small number of analyzable trajectories.

The strongest promising pattern would be:

- rank 2, 3, or 5 preserves the broad CPP waveform;
- latent trajectories differ systematically between fast and slow responses;
- correct and error trials show visible response-proximal divergence;
- the low-rank state is easier to interpret than the existing GRU hidden state.

A negative result would also be useful. If very low ranks fail to preserve CPP,
then the project can justify keeping the GRU as the primary model and using
low-rank analysis only as a post-hoc dimensionality-reduction tool.

## Risks and Limitations

The first smoke test is not a final model comparison. It uses a small training
budget and a simplified low-rank architecture. Because EEG is noisy and
single-trial CPP can be variable, weak behavior prediction in this smoke test
should not be interpreted as evidence against the broader research idea.

The main risk is underfitting: a very low-rank model may be too constrained to
reconstruct trial-level EEG. A second risk is overinterpretation: visually clear
trajectories may reflect CPP amplitude differences rather than a genuine
decision mechanism. Follow-up analyses should therefore include stronger
controls, subject-aware validation, and comparisons against hand-crafted CPP
features.

## Next Steps

If the smoke test is promising, the next phase should train the best rank on
more data, add stronger cross-validation, and compare low-rank latents against
the current GRU hidden states and hand-crafted CPP features. A later version
could implement a stochastic low-rank RNN closer to recent neural-data
inference work and use fixed-point or slow-point analysis to test whether the
model contains candidate attractor-like accumulation dynamics.
