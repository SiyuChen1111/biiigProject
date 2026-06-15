# Neural Model Validation Audit

## What Was Evaluated

- Model checkpoint: `/Users/siyu/Documents/GitHub/biiigProject/Results/model_checkpoints/best_model.pt`
- Neural data: `/Users/siyu/Documents/GitHub/biiigProject/Data/ProcessedData/eeg_cpp_trials.npy`
- Alignment: response-locked only in this dataset
- Model input features: CP1, CP2, CPz EEG channels
- Model output target: current EEG reconstruction; future EEG prediction was also used during training
- Hidden-state tensor shape: [7297, 308, 32] = trials x time points x hidden dimensions
- Dataset size: 7297 trials, 308 time points, 3 channels
- Test split size used for neural reconstruction validation: 1094 trials

## Neural Reconstruction Validation

Held-out neural reconstruction was strong.

- Test RMSE: 0.3527
- Test R2: 0.8810
- Test empirical-predicted correlation: 0.9407

Time-window results:

| window               |     rmse |       r2 |     corr |
|:---------------------|---------:|---------:|---------:|
| minus600_to_minus300 | 0.214428 | 0.959463 | 0.984279 |
| minus300_to_minus120 | 0.196377 | 0.964076 | 0.98494  |
| minus120_to_minus50  | 0.156177 | 0.974108 | 0.988936 |
| minus600_to_minus50  | 0.202001 | 0.962728 | 0.984512 |

Channel-level performance was also consistent across CP1, CP2, and CPz, with correlations around 0.94 and R2 around 0.88.

## CPP/ERP Signature Validation

The model captured the response-locked CPP-like signal well.

- CPP amplitude, -600 to -50 ms: R2 0.9929, correlation 0.9981, RMSE 0.0492
- CPP slope, -600 to -50 ms: R2 0.9861, correlation 0.9981, RMSE 0.000233

Figures were saved for overall CPP trajectory, condition trajectories, difficulty trajectories, correct/error trajectories, time-resolved prediction quality, and empirical-vs-predicted scatter.

## Hidden-State Representational Validation

Fast subject-balanced cross-validation was used for the representational checks to keep the audit tractable. This used 2,378 sampled trials balanced across subjects, with shuffled-hidden, shuffled-label, and majority-class controls.

Strongest classification margins over controls:

| target      | window               |   balanced_accuracy |   control_max_balanced_accuracy |       margin |      auc |
|:------------|:---------------------|--------------------:|--------------------------------:|-------------:|---------:|
| rt_bin      | minus600_to_minus300 |            0.501148 |                        0.367101 |  0.134048    | 0.694864 |
| rt_bin      | minus600_to_minus50  |            0.509177 |                        0.376305 |  0.132872    | 0.679528 |
| rt_bin      | minus300_to_minus120 |            0.441507 |                        0.333832 |  0.107674    | 0.611655 |
| rt_bin      | minus120_to_minus50  |            0.431793 |                        0.346074 |  0.085719    | 0.603373 |
| correctness | minus120_to_minus50  |            0.518217 |                        0.500843 |  0.0173741   | 0.531251 |
| condition   | minus600_to_minus50  |            0.265107 |                        0.25     |  0.0151067   | 0.515689 |
| condition   | minus600_to_minus300 |            0.258078 |                        0.253768 |  0.00430985  | 0.505281 |
| condition   | minus120_to_minus50  |            0.24968  |                        0.25     | -0.000320135 | 0.499968 |

Strongest continuous-target margins over controls:

| target                        | window               |        r2 |   control_max_r2 |   margin |     corr |
|:------------------------------|:---------------------|----------:|-----------------:|---------:|---------:|
| cpp_amp_minus600_to_minus50   | minus600_to_minus50  | 0.970393  |        0.0176896 | 0.952704 | 0.987007 |
| cpp_amp_minus600_to_minus50   | minus120_to_minus50  | 0.650652  |       -0.217785  | 0.868437 | 0.811505 |
| cpp_amp_minus600_to_minus50   | minus300_to_minus120 | 0.29856   |       -0.485222  | 0.783782 | 0.785895 |
| log_RT_ms                     | minus300_to_minus120 | 0.0577567 |       -0.555649  | 0.613405 | 0.254807 |
| cpp_slope_minus600_to_minus50 | minus120_to_minus50  | 0.356933  |       -0.253378  | 0.610311 | 0.636503 |
| cpp_amp_minus600_to_minus50   | minus600_to_minus300 | 0.589383  |        0.0042064 | 0.585177 | 0.819183 |

Interpretation: hidden states clearly encode neural CPP amplitude and RT-bin structure above shuffled controls. Choice and condition decoding are weak and near baseline in the early window.

## Behavioral External Validation

Log RT prediction did not show a reliable incremental benefit from hidden states in this fast check.

- Behavior-only R2: 0.0857
- CPP features only R2: 0.0999
- Behavior + CPP R2: 0.1680
- Behavior + hidden R2: -0.1155
- Behavior + CPP + hidden R2: -0.1897

The best external RT result came from behavior + hand-crafted CPP features, not from adding hidden states. This means behavioral relevance should be treated cautiously and not used as the main justification for downstream hidden-state interpretation.

## Decision

Recommendation: **Pass for neural reconstruction and CPP/ERP representation; partial pass for downstream hidden-state behavioral interpretation.**

The model is sufficiently trained to support downstream analysis of hidden states as learned neural representations because held-out neural reconstruction is strong and CPP/ERP signatures are preserved. However, the hidden states should not yet be described as a robust behavioral model: choice/condition decoding is weak, and adding hidden states did not improve log RT prediction over simpler behavioral/CPP feature baselines in the fast external-validation check.

Use the hidden states for cautious neural latent-dynamics analyses, especially CPP amplitude and response-proximal state structure. Avoid strong claims that they explain choice or RT behavior unless a slower, fuller validation confirms stable incremental behavioral prediction.

## Output Files

- `neural_goodness_of_fit.csv`
- `time_resolved_neural_fit.csv`
- `cpp_erp_signature_fit.csv`
- `hidden_state_classification_decoding_fast.csv`
- `hidden_state_regression_decoding_fast.csv`
- `behavioral_external_validation_rt_fast.csv`
- `figures/`
