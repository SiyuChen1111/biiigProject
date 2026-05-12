# Testing the Specificity of Centroparietal Positivity with EEG-Based Classification

## Abstract
Centroparietal positivity (CPP) is widely discussed as an electrophysiological correlate of evidence accumulation in human decision-making. At the same time, centroparietal positive activity with similar morphology may appear in tasks that do not cleanly instantiate canonical evidence accumulation, raising an unresolved question about how specific current operational definitions of CPP really are. This project proposes to develop an EEG-based CPP detector using preprocessed, epoched data and then use that detector as both a practical tool and a theoretical probe. The technical objective is to determine whether compact neural models, especially EEGNet, can distinguish CPP-like from non-CPP-like activity in a robust and interpretable way. The scientific objective is to test whether a detector trained on canonical CPP-positive tasks generalizes across subjects, tasks, and datasets while remaining specific to canonical CPP rather than broader decision-related centroparietal positivity. By comparing rule-based baselines, linear models, and compact convolutional networks across positive-control, negative-control, and boundary-case datasets, this study aims to clarify whether successful detection reflects a transferable CPP representation or merely task-specific regularities.

## Motivation and Significance
An automatic CPP detector would be valuable for at least two reasons. First, it would provide a reusable tool for screening and quantifying CPP-like activity in EEG datasets. Second, and more importantly, it would offer a new way to test the specificity of CPP as a candidate evidence-accumulation signal. If a detector trained on canonical CPP data generalizes only within the same task family, then the learned representation is likely task-specific. If it generalizes broadly but also produces frequent detections in tasks without canonical evidence accumulation, then the result would suggest that the operational definition of CPP is too broad or that CPP-like centroparietal positivity indexes a more general class of decision-related computations.

## Overall Objective
The overall objective of this project is to build and evaluate a generalizable and interpretable EEG-based detector of CPP-like activity, and to use its successes and failures to refine the theoretical interpretation of CPP.

## Specific Aims

### Aim 1. Build a first-pass CPP detector from preprocessed EEG epochs.
This aim focuses on the engineering problem. Preprocessed, epoched EEG will be used as model input, rather than raw continuous EEG, because CPP is defined relative to task timing and because single-trial EEG has a low signal-to-noise ratio. The first detector will be trained on canonical CPP-positive data and implemented with a hierarchy of baselines: rule-based template scoring, linear classification on handcrafted features, and a compact convolutional model such as EEGNet.

### Aim 2. Test whether the detector learns a transferable CPP representation.
This aim evaluates whether the learned representation generalizes beyond the training dataset. Performance will be assessed within dataset, across subjects, across tasks, and across datasets. Strong performance under subject- and task-held-out evaluation would support the claim that the detector has captured a stable CPP-related representation rather than superficial properties of a single dataset.

### Aim 3. Test the specificity of the current operational definition of CPP.
This aim treats the detector as a theoretical probe. If the detector frequently identifies CPP-like signals in negative-control or boundary-case tasks, the result will not be interpreted simply as proof that CPP is present everywhere. Instead, it will motivate a more careful interpretation: the detector may be capturing a broader decision-related centroparietal positivity rather than a specific evidence-accumulation signal. This aim therefore uses model errors and out-of-domain detections as informative results rather than mere failures.

## Working Hypotheses
1. A model trained on preprocessed, epoched EEG can classify CPP-like versus non-CPP-like activity above baseline.
2. A detector trained on canonical evidence-accumulation tasks will generalize partially to unseen tasks if it captures a real CPP-related representation.
3. Frequent positive detections in nominally non-accumulation tasks will indicate insufficient specificity in the current operational definition or the presence of a broader decision-related signal.
4. Compact models such as EEGNet will perform competitively with larger models, implying that label definition and experimental design matter more than architectural complexity in the first stage of the project.

## Methods

### Data strategy
Datasets will be organized into three groups:

1. **Canonical CPP-positive datasets**: tasks with evidence accumulation where CPP has already been reported.
2. **Negative-control datasets**: tasks without clear evidence accumulation, used to challenge detector specificity.
3. **Boundary-case datasets**: tasks involving decision formation or related processes, but where CPP has not been consistently reported.

This three-group design is essential because it prevents the project from collapsing into a simple within-task classification problem.

### Input representation
The first version of the detector will use preprocessed EEG epochs that are either stimulus-locked, response-locked, or both. Full montage input may be retained for modeling, but later analyses will test whether the learned representation concentrates on the expected centroparietal channels and time windows. Raw continuous EEG will be deferred to later stages because it substantially increases ambiguity in both labeling and interpretation.

### Label construction
Hard binary labels assigned at the dataset level are too coarse for this problem. Instead, labels will be constructed in three stages:

1. **Canonical CPP criteria**: define waveform and topographic criteria associated with canonical CPP, including centroparietal distribution, pre-response buildup, post-response decline, and expected timing.
2. **Soft scoring**: compute a CPP score from features such as mean amplitude, peak amplitude, slope, temporal alignment, and topographic consistency.
3. **Classifier targets**: convert the soft score into binary or ordinal targets for the first detector while preserving the possibility of returning to continuous scoring later.

This strategy reduces the risk of forcing an oversimplified label on a signal that is likely continuous and partially ambiguous.

### Models and baselines
The project will compare three levels of modeling complexity:

- **Rule-based baseline**: template-based or feature-threshold CPP scoring.
- **Linear baseline**: logistic regression or LDA using handcrafted CPP-related features.
- **Neural baseline**: EEGNet as the primary compact model, with ShallowConvNet as a secondary comparison and DeepConvNet only when data volume is sufficient.

The purpose of this hierarchy is not merely to maximize accuracy, but to determine whether more complex models capture genuinely useful structure beyond interpretable handcrafted features.

### Evaluation plan
Evaluation will emphasize robustness and interpretability rather than accuracy alone. The main dimensions of evaluation will be:

- within-dataset performance
- cross-subject generalization
- cross-task generalization
- cross-dataset generalization

Primary metrics will include balanced accuracy, AUC, precision, recall, and confusion matrices broken down by dataset type. Interpretability analyses will examine whether model decisions rely on expected channels and time windows.

### Risks and safeguards
The main methodological risk is that the detector may learn task identity, preprocessing artifacts, or dataset-specific regularities rather than CPP itself. To reduce this risk, train/test splits should be constructed at the subject and, when possible, dataset level rather than random trial level. Negative-control and boundary-case datasets are also necessary to ensure that good performance does not simply reflect trivial discrimination between unrelated task families.

## Expected Outcomes
Several outcomes would be scientifically informative:

- If the detector generalizes well to unseen CPP-positive tasks and rejects negative controls, the current operationalization of CPP gains support.
- If the detector performs well within dataset but fails across tasks, CPP detection may be more task-specific than previously assumed.
- If the detector often identifies CPP-like activity in nominally non-accumulation tasks, then either the operational definition of CPP is too broad or CPP-like centroparietal positivity reflects a wider class of decision-related computations.
- If simple rule-based or linear baselines match deep models, then the primary bottleneck is likely theoretical and labeling-related rather than architectural.

## First-Phase Milestones
1. Select one canonical CPP-positive dataset and complete preprocessing and epoching.
2. Define a first-pass CPP scoring rule.
3. Build soft labels and a binary target version.
4. Train rule-based, linear, and EEGNet baselines.
5. Evaluate within dataset and across subjects.
6. Test on one negative-control dataset.
7. Test on one boundary-case dataset.

## Contribution
This project aims to contribute both a practical EEG-based CPP detection tool and a stronger framework for evaluating the specificity of CPP as an evidence-accumulation marker. Its value lies not only in positive classification results, but also in the interpretation of where and why the detector succeeds or fails.
