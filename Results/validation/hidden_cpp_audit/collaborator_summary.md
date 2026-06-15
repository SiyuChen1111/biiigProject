# Hidden-CPP audit: short summary

## What we checked

We asked a simple question: do the hidden states from the current neural model actually preserve empirical CPP-related information strongly enough to justify exploratory hidden-CPP analysis? To answer that, we reran the hidden-to-CPP regression with stricter controls and kept the interpretation focused on neural validation, not behavioral mechanism claims.

## What changed from the previous analysis

The main change is that the strict audit used raw empirical CPP derived directly from CP1, CP2, and CPz. The earlier figure pipeline used CPP targets derived from the normalized EEG array returned by the stage-2 loading pipeline. The new audit is therefore a cleaner check of hidden-state relation to empirical CPP.

## Main result

- Hidden states strongly predicted empirical CPP amplitude and CPP AUC.
- Raw matched-window `R^2` values were very high, especially in matched windows.
- The strongest control-corrected evidence was for late pre-response amplitude and AUC around `-120 to -50 ms`.
- In that late window, both amplitude and AUC kept a control-corrected advantage of about `0.31`.
- CPP slope prediction was clearly weaker and less stable than amplitude and AUC.
- Task decoding was limited:
  - RT bin was the clearest positive result
  - correctness was weak
  - condition and difficulty were near chance
  - choice and arrangement were not reliable

## How to interpret it

- The current model does preserve CPP-related neural information in its hidden states.
- That supports using the hidden states for exploratory hidden-CPP mapping.
- The very high raw `R^2` values should be interpreted carefully, because the hidden states and CPP targets come from the same EEG trial segments.
- For that reason, the more useful number is not raw `R^2`, but the margin above the best shuffled or mismatch control.

## What we should not claim yet

- We should not claim a strong behavioral decision mechanism from these hidden states.
- We should not claim robust encoding of choice, arrangement, difficulty, or condition.
- We should not claim strong time-specific dynamics from raw `R^2` alone.
- We should not treat the behavioral RT figure as the main evidence.

## Next step

The best next step is to freeze the current model and move from broad hidden-to-CPP prediction toward explicit CPP-related latent axes. Those axes can then be tested for stability across folds and subjects, and only after that should we ask whether they add anything to RT prediction beyond empirical CPP features themselves.
