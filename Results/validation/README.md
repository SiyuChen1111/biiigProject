# Hidden-CPP audit summary

## 1. Why we did this

We trained a neural model on response-locked CPP/EEG signals. Before interpreting its hidden states, we needed to check whether the hidden states actually preserve empirical CPP-related information.

Main question:

"Can the model hidden states be used for exploratory hidden-CPP mapping?"

This is a neural validation step, not yet a strong behavioral mechanism claim.

## 2. What data and model were used

- Trained model checkpoint: current frozen forward-GRU checkpoint used in the stage-2 audit
- Latent file: full latent export from the frozen model
- EEG/CPP source: response-locked CP1, CP2, CPz trial-wise EEG
- Metadata: trial-level behavioral and condition table
- Trials: 7,297
- Time points: 308
- Hidden dimension: 32
- CPP channels: CP1, CP2, CPz
- Response-locked windows:
  - `-600 to -300 ms`
  - `-300 to -120 ms`
  - `-120 to -50 ms`
  - `-600 to -50 ms`

## 3. What regression did we run?

For each trial, we averaged the model hidden states within a response-locked time window. These hidden-state features were then used to predict empirical CPP features.

`CPP feature = linear combination of hidden-state dimensions + error`

- Independent variables: hidden-state features averaged within a selected response-locked window
- Dependent variables: empirical CPP amplitude, CPP slope, and CPP AUC
- Model: ridge regression with cross-validation
- Why ridge: the hidden dimensions are correlated and moderately high-dimensional, so L2 regularization helps keep the regression stable

## 4. What controls did we use?

- Trial-shuffled hidden control
- Within-subject shuffled hidden control
- Within-subject x condition shuffled hidden control
- Time-window mismatch control
- Subject-demeaned check

The most conservative metric is not raw `R^2`, but:

`real hidden R^2 - best control R^2`

## 5. Main results

- Hidden states strongly predicted empirical CPP amplitude and CPP AUC.
- Raw `R^2` values were very high in matched windows.
- This is expected because hidden states are extracted from the same EEG trials used to compute CPP features.
- Raw `R^2` should therefore be interpreted as neural information preservation, not as a new mechanism.
- The most convincing control-corrected results were for late pre-response amplitude and AUC, especially around `-120 to -50 ms`.
- In the strict within-subject audit, `-120 to -50 ms` CPP amplitude reached raw `R^2 = 0.99` and control-corrected `delta R^2 = 0.31`.
- `-120 to -50 ms` CPP AUC reached raw `R^2 = 0.99` and control-corrected `delta R^2 = 0.31`.
- CPP slope prediction was weaker and less stable. For the broad `-600 to -50 ms` window, slope remained positive in raw `R^2` (`0.35`) but fell below the best control (`delta R^2 = -0.12`).
- Task-variable decoding was limited: RT bin was the most stable; correctness was weak; condition and difficulty were near chance; choice and arrangement were not reliable.

## 6. Interpretation

### Supported

- The current model is suitable for exploratory hidden-CPP mapping.
- Hidden states preserve empirical CPP amplitude and AUC information.
- Hidden states can be treated as CPP-related neural latent representations for validation purposes.

### Not yet supported

- Strong task-specific decision mechanism claims
- Robust encoding of difficulty, choice, or arrangement
- Strong behavioral mechanism claims from hidden states
- Strong time-specific dynamics claims based only on raw `R^2`

## 7. Figures

- Main Figure 2: hidden-state relation to CPP features and task variables
  - Panel a: raw hidden-to-CPP prediction
  - Panel b: control-corrected hidden-to-CPP prediction
  - Panel c: task decoding relative to controls
- Supplementary behavioral validation figure: external RT prediction
- Supplementary waveform figure: real vs reconstructed CPP waveforms

## 8. Current conclusion

The current model passes as suitable for exploratory hidden-CPP mapping, but only as neural validation. It does not yet pass as evidence for a strong behavioral decision mechanism.
