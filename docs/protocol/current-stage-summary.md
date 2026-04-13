# Current Stage Summary

## Project objective

The current project aims to study centroparietal positivity (CPP)-related EEG structure using a reproducible trial-level modeling pipeline. The long-term goal is not only to classify EEG trials, but also to understand whether the learned representations are physiologically meaningful and, eventually, to generate simulated EEG waveforms that preserve relevant CPP-like structure.

This document summarizes what has already been implemented, what has been learned from the current experiments, and what the most defensible next step is.

---

## 1. What is already implemented

### 1.1 Reproducible project structure

The repository has been reorganized into a TIER-inspired transitional structure with:

- `src/` for reusable code
- `scripts/` for runnable experiment entry points
- `docs/` for notes, blueprints, and protocol documents
- `README.md` and `requirements.txt` at the root

The expected working directory is the repository root, and a master script entry point is available through `scripts/run_master.py`.

### 1.2 Data and preprocessing pipeline

The current code can:

- read the Kosciessa `task-dynamic` dataset from local raw files
- build subject-level trial tables from behavior and EEG timing information
- filter invalid trials
- extract response-locked EEG epochs
- keep 60 scalp EEG channels after removing auxiliary channels
- apply average reference, 1–30 Hz filtering, baseline correction, and resampling to 128 Hz

### 1.3 Phase-1 discriminative pipeline

The phase-1 pipeline is implemented and runnable through:

- `scripts/train_phase1_eegnet.py`
- `scripts/baseline_phase1_cpp_features.py`

This pipeline defines CPP-like labels from EEG itself using the ROI:

- `CPz`
- `CP1`
- `CP2`

and the features:

- `AMS`
- `PAMS`
- `SLPS`

The resulting `cpp_score` is thresholded into high-vs-low CPP-like labels using top and bottom quartiles, and all evaluation is done with subject-held-out cross-validation.

### 1.4 Exploratory phase-2 discriminative tasks

Two external-label experiments were implemented:

- `scripts/train_phase2_cue_dimensionality.py`
- `scripts/train_phase2_rt_fastslow.py`

These were intended to test whether the pipeline could learn an external behavioral or task-related label rather than simply recovering a hand-crafted EEG score.

### 1.5 Conditional EEG generation pipeline

A conditional VAE-based generator is now implemented through:

- `src/models/conditional_eeg_vae.py`
- `scripts/train_phase1_conditional_vae.py`
- `scripts/analyze_conditional_vae_samples.py`

This model is conditioned on the phase-1 high-vs-low CPP-like labels and can generate simulated EEG epoch arrays.

---

## 2. What the current experiments show

## 2.1 Phase-1 EEGNet works well as a score-recovery model

The phase-1 EEGNet experiment produced very strong subject-held-out results. However, a simple logistic regression baseline using only `AMS`, `PAMS`, and `SLPS` also matched or exceeded EEGNet in the retained scaled-baseline outputs.

The most defensible interpretation is therefore:

> The current phase-1 system is highly effective at recovering a hand-crafted CPP-like distinction across subjects, but this does not show that EEGNet has learned a representation that is meaningfully deeper than the original feature rule.

This is still a useful result. It demonstrates that the current CPP-like scoring rule is stable and recoverable, and that the preprocessing and cross-subject evaluation pipeline are working as intended.

## 2.2 Phase-2 external-label tasks did not yet show reliable signal

The retained quick-check evidence currently supports near-chance performance for:

- `cue_dimensionality` extremes

An RT fast-vs-slow exploratory script also exists in the repository, but the retained local evidence after cleanup is strongest for the `cue_dimensionality` quick-check outputs.

The strongest current interpretation is not that the code is broken, but that these labels, as currently defined, do not yet provide a stable cross-subject EEG decoding target in the present setup.

This is important because it narrows the problem:

> The bottleneck is no longer whether the pipeline can run, but whether the chosen external label is the right scientific target.

## 2.3 The conditional VAE can generate EEG arrays, but not yet physiologically convincing EEG

The current conditional VAE successfully trains, produces losses, and writes generated samples to disk. That means the generative branch is operational in engineering terms.

However, in the retained current outputs, the generated high-vs-low condition waveforms are still noisy and not well aligned with real ROI ERP structure.

---

## 3. What was improved in the generator

To push the generator toward more physiologically plausible output, several targeted improvements were added:

### 3.1 Decoder structure

The current decoder uses a more time-structured design based on transposed temporal convolutions and explicit interpolation back to the target sample length.

### 3.2 Physiology-aware losses

The training objective now includes:

- reconstruction loss
- KL loss
- feature consistency loss on `AMS`, `PAMS`, `SLPS`
- temporal smoothness loss
- ROI-weighted reconstruction loss in `CPz/CP1/CP2`

### 3.3 Stronger condition preservation

An auxiliary label-prediction head was added from latent space so that the model is explicitly encouraged to preserve the high-vs-low condition distinction rather than ignoring the condition input.

---

## 4. What the improved generator currently achieves

The current conditional VAE has the following implemented properties:

- it trains stably
- it produces generated EEG epoch arrays
- it is explicitly constrained to preserve CPP-related features and ROI behavior

However, the current generated waveforms are still not convincing as realistic CPP-like EEG.

The most defensible interpretation is:

> The current model can generate arrays under explicit physiological constraints, but it has not yet reached the stage where the generated ROI ERP waveforms can be described as physiologically credible.

In the current comparison plots, generated waveforms remain too jagged, condition separation is still weak or inconsistent, and the curves do not yet resemble a clear ERP-like structure.

---

## 5. Current best understanding of the project state

At this point, the project has moved beyond pure implementation uncertainty.

### What is now clear

1. The preprocessing, epoching, labeling, training, and evaluation pipeline is operational.
2. The phase-1 CPP-like scoring rule is strong and reproducible across subjects.
3. EEGNet can recover that phase-1 distinction, but does not currently outperform simpler hand-crafted features.
4. The presently retained phase-2 external-label evidence does not yet provide a convincing next discriminative target.
5. The generative branch is alive and trainable, but its outputs are not yet physiologically convincing.

### What is no longer the main problem

- basic code implementation
- reproducible project structure
- ability to train a model end-to-end

### What is now the real problem

- finding a scientifically stronger external task label for phase 2
- making the conditional generator produce waveform-level outputs that look like real CPP-related EEG rather than merely low-loss reconstructions

---

## 6. Recommended next step

The most defensible next modeling step is **not** to keep scaling the full 60-channel generator immediately.

The current recommendation is:

> Build a ROI-focused conditional generator that targets `CPz`, `CP1`, and `CP2` directly.

Why this is the best next step:

1. The scientific target is currently CPP-like morphology in a small centroparietal ROI.
2. Generating all 60 channels spreads model capacity across many dimensions that are not equally important for the current question.
3. If a small ROI-focused generator still cannot produce plausible waveforms, then scaling the same logic to full-head EEG is unlikely to help.

This next step should be treated as a scientific narrowing of scope, not as a retreat.

---

## 7. One-paragraph conclusion

The project is now in a productive middle stage. The discriminative CPP-like pipeline is complete and reproducible, and it has established a stable subject-generalizable hand-crafted CPP-like score. The external-label phase-2 tasks tried so far have not yet produced convincing cross-subject signal. A conditional EEG generator is now implemented, but it still does not generate physiologically credible CPP-like waveforms. The best next move is therefore to stop broadening the task space and instead focus the generator on the CPP ROI itself, where physiological plausibility can be judged more directly and with less representational burden.
