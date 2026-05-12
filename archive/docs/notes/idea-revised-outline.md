# Toward a Generalizable Detector of Centroparietal Positivity (CPP) from EEG

## 1. Background
Centroparietal positivity (CPP) is often regarded as an EEG correlate of evidence accumulation during decision-making. However, CPP-like centroparietal positive activity may also appear in tasks that do not clearly involve canonical evidence accumulation. Therefore, building an automatic CPP detector is useful not only as a technical tool, but also as a way to test how specific the current operational definition of CPP really is.

## 2. Core Question
Can we build a model that detects CPP-like activity from EEG epochs, and does that detector generalize across tasks while remaining specific to canonical CPP rather than broader decision-related positivity?

## 3. Main Claim
This project does not aim to prove that CPP exists or does not exist. Instead, it asks whether a detector trained on canonical CPP data learns:

1. a task-general and interpretable CPP representation, or
2. a broader, less specific decision-related centroparietal signal.

## 4. Hypotheses

### H1. Feasibility hypothesis
A model trained on preprocessed, epoched EEG can classify CPP-like versus non-CPP-like activity above baseline.

### H2. Generalization hypothesis
A detector trained on canonical evidence-accumulation tasks will partially generalize to unseen tasks if it captures a real CPP-related representation.

### H3. Specificity hypothesis
If the detector frequently identifies CPP in tasks that theoretically should not contain canonical evidence accumulation, then the learned representation is likely not specific enough and may reflect broader centroparietal decision-related activity.

### H4. Simplicity hypothesis
A compact model such as EEGNet may perform competitively with more complex models, suggesting that the bottleneck is label definition and task design rather than model complexity.

## 5. Study Design
This study has two goals:

### Goal A: Tool-building
Develop a CPP detector from preprocessed EEG epochs.

### Goal B: Theory-testing
Use detector performance across task types to evaluate whether the current operational definition of CPP is sufficiently specific.

## 6. Input Representation
The first version will use:

- preprocessed EEG
- epoched trials
- time-locked windows, stimulus-locked and/or response-locked
- centroparietal-relevant channels or full montage with later interpretability analysis

Raw continuous EEG will not be the primary starting point, because the signal-to-noise ratio is too low and task timing information is critical for defining CPP.

## 7. Data Groups
Three types of datasets are needed:

### Group 1: Canonical CPP-positive datasets
Tasks with evidence accumulation where CPP has already been reported.

### Group 2: CPP-negative or theoretically weak-CPP datasets
Tasks without clear evidence accumulation, used as negative controls.

### Group 3: Boundary-case datasets
Tasks involving decision formation or related processes, but where CPP has not been consistently reported. These are the most informative datasets for testing specificity.

## 8. Label Strategy
A simple binary label at the dataset level, such as “this dataset has CPP” or “does not have CPP,” is too coarse. Instead, labels should be defined in stages.

### Stage 1: Canonical CPP criteria
A trial or condition is more likely to be labeled CPP-like if it shows:

- centroparietal topography
- ramping before response
- reduction after response
- expected timing window
- consistency with condition effects such as evidence strength

### Stage 2: Soft label or CPP score
Construct a CPP score from multiple features such as:

- mean amplitude
- peak amplitude
- slope
- temporal alignment
- topographic consistency

### Stage 3: Binary conversion
For the first classifier, convert the CPP score into binary labels if needed.

This is better than directly assigning hard labels from task category alone.

## 9. Models
The first model should be simple and interpretable.

### Baseline models
- rule-based or template-based CPP scoring
- logistic regression or LDA on handcrafted CPP features

### Neural models
- EEGNet as the primary deep-learning baseline
- ShallowConvNet as an additional comparison
- DeepConvNet only if dataset size is sufficient

## 10. Evaluation
Performance should not be judged only by accuracy.

### Required evaluation dimensions
- within-dataset performance
- cross-subject generalization
- cross-task generalization
- cross-dataset generalization

### Metrics
- balanced accuracy
- AUC
- precision / recall
- confusion matrix by dataset type

### Interpretability checks
The detector should also be examined for:

- which channels contribute most
- which time windows contribute most
- whether the model focuses on canonical CPP-relevant patterns

## 11. Key Risk
The main risk is that the model may learn task identity rather than CPP. For example, it may exploit dataset-specific preprocessing artifacts, timing structure, or unrelated spectral differences.

Therefore, good performance alone is not enough. The model must also show interpretable and transferable behavior.

## 12. Possible Outcomes and Their Meaning

### Outcome 1
The detector generalizes well to unseen CPP-positive tasks and rejects negative-control tasks.

-> This supports the usefulness of the current CPP operationalization.

### Outcome 2
The detector performs well within dataset but fails across tasks.

-> This suggests the learned representation is task-specific rather than CPP-specific.

### Outcome 3
The detector often finds CPP-like signals in nominally non-accumulation tasks.

-> This suggests the current definition may be too broad or that CPP-like positivity reflects a more general decision-related process.

### Outcome 4
Simple baselines perform as well as deep models.

-> This suggests the main challenge is not model complexity, but label definition and theoretical clarity.

## 13. First-Step Plan
1. Preprocess and epoch one canonical CPP-positive dataset.
2. Define a first CPP scoring rule based on known waveform properties.
3. Build binary or soft labels from that score.
4. Train baseline models.
5. Train EEGNet.
6. Compare within-subject and cross-subject performance.
7. Test on one negative-control dataset.
8. Test on one boundary-case dataset.

## 14. Contribution
This project contributes both:

1. a practical EEG-based CPP detection tool, and
2. a framework for testing the specificity of CPP as an evidence-accumulation marker.
