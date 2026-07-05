# biiigProject: Response-Locked CPP Latent Dynamics

## Overview

This repository studies whether a compact neural dynamical model can learn useful latent structure from response-locked single-trial EEG, especially from the CPP-related channels `CP1`, `CP2`, and `CPz`.

The current main line is a **no-prior Rank-5 low-rank RNN**. Its role is to learn five latent variables, `z1` to `z5`, that summarize pre-response neural dynamics in a form that is easier to inspect than the older higher-dimensional models. A **CPP-prior Rank-5 version** is still kept in the repository, but it now serves as a robustness comparison rather than the default scientific target.

The practical question of the project is:

1. Can a Rank-5 low-rank RNN recover stable CPP-related latent dynamics from response-locked EEG?
2. Do those latents first pass a behavioural validation step through response-time analyses?
3. Do they then support the higher-priority drift-rate-oriented follow-up analyses aimed at mechanism rather than only prediction?

Older GRU work is retained only as background. If historical context is needed, see [logs.md](/Users/siyu/Documents/GitHub/biiigProject/logs.md).

## Current Scientific Position

- **Primary model:** no-prior Rank-5 low-rank RNN.
- **Comparison model:** CPP-prior Rank-5 low-rank RNN.
- **Default interpretation:** use the no-prior model as the main representation analysis, then check whether conclusions remain directionally stable in the CPP-prior comparison.
- **Behavioural priority:** RT is the earlier validation step, but drift-rate is the more important mechanistic target.
- **Not the default main line anymore:** the earlier GRU hidden-state workflow.

## Repository Map

These are the most important places to start from.

- [Data/ProcessedData](/Users/siyu/Documents/GitHub/biiigProject/Data/ProcessedData)  
  Main analysis-ready EEG, time axis, metadata, and preprocessing notes.
- [Data/IntermediateData/latents_full](/Users/siyu/Documents/GitHub/biiigProject/Data/IntermediateData/latents_full)  
  Exported legacy full hidden-state representations from the earlier GRU line.
- [Scripts/low_rank_rnn_rank5_pipeline.ipynb](/Users/siyu/Documents/GitHub/biiigProject/Scripts/low_rank_rnn_rank5_pipeline.ipynb)  
  Main Rank-5 notebook entry point.
- [Scripts/low_rank_rnn_rank5_no_cpp_prior_ablation.ipynb](/Users/siyu/Documents/GitHub/biiigProject/Scripts/low_rank_rnn_rank5_no_cpp_prior_ablation.ipynb)  
  Key no-prior follow-up notebook and ablation route.
- [Scripts/s2_training/low_rank_full_training.py](/Users/siyu/Documents/GitHub/biiigProject/Scripts/s2_training/low_rank_full_training.py)  
  Formal script for full-data Rank-5 training runs.
- [Scripts/s4_analysis/rank5_dual_prior_comparison.py](/Users/siyu/Documents/GitHub/biiigProject/Scripts/s4_analysis/rank5_dual_prior_comparison.py)  
  Side-by-side comparison of no-prior and CPP-prior Rank-5 latents.
- [Scripts/s5_regression/1_latentZ_v.ipynb](/Users/siyu/Documents/GitHub/biiigProject/Scripts/s5_regression/1_latentZ_v.ipynb)  
  Supplementary latent-to-drift-rate (`v`) regression notebook.
- [Results/low_rank_full_training_rank5_20260626_1427](/Users/siyu/Documents/GitHub/biiigProject/Results/low_rank_full_training_rank5_20260626_1427)  
  Formal full-data Rank-5 result package.
- [Results/rank5_dual_prior_comparison](/Users/siyu/Documents/GitHub/biiigProject/Results/rank5_dual_prior_comparison)  
  Committed no-prior versus CPP-prior comparison outputs.
- [Results/regression](/Users/siyu/Documents/GitHub/biiigProject/Results/regression)  
  Latest local supplemental RT regression results.

## Data Snapshot

The active processed dataset is in [Data/ProcessedData](/Users/siyu/Documents/GitHub/biiigProject/Data/ProcessedData).

- `7297` valid trials
- `41` subjects
- `308` time points per trial
- `3` channels: `CP1`, `CP2`, `CPz`
- Main EEG array shape: `(7297, 308, 3)`

Core files:

- [eeg_cpp_trials.npy](/Users/siyu/Documents/GitHub/biiigProject/Data/ProcessedData/eeg_cpp_trials.npy)
- [times_ms.npy](/Users/siyu/Documents/GitHub/biiigProject/Data/ProcessedData/times_ms.npy)
- [metadata.csv](/Users/siyu/Documents/GitHub/biiigProject/Data/ProcessedData/metadata.csv)
- [channel_names.txt](/Users/siyu/Documents/GitHub/biiigProject/Data/ProcessedData/channel_names.txt)
- [preprocessing_notes.md](/Users/siyu/Documents/GitHub/biiigProject/Data/ProcessedData/preprocessing_notes.md)

## Where To Start

Recommended reading and running order:

1. Read this file first.
2. Read [low_rank_rnn_drift_rate_followup_plan.md](/Users/siyu/Documents/GitHub/biiigProject/low_rank_rnn_drift_rate_followup_plan.md) to see the intended interpretation and next analysis direction.
3. Open [Scripts/low_rank_rnn_rank5_pipeline.ipynb](/Users/siyu/Documents/GitHub/biiigProject/Scripts/low_rank_rnn_rank5_pipeline.ipynb) as the main notebook route.
4. Then open [Scripts/low_rank_rnn_rank5_no_cpp_prior_ablation.ipynb](/Users/siyu/Documents/GitHub/biiigProject/Scripts/low_rank_rnn_rank5_no_cpp_prior_ablation.ipynb) for the no-prior follow-up path.
5. Check the RT validation outputs in [Results/regression](/Users/siyu/Documents/GitHub/biiigProject/Results/regression) to confirm that the latent space carries behavioural signal at all.
6. Then move to [Scripts/s5_regression/1_latentZ_v.ipynb](/Users/siyu/Documents/GitHub/biiigProject/Scripts/s5_regression/1_latentZ_v.ipynb) for the higher-priority drift-rate branch.
   This notebook uses an externally derived drift-rate target `v`, rather than a drift-rate column stored directly inside the active processed metadata.

Minimal command examples:

```bash
jupyter notebook /Users/siyu/Documents/GitHub/biiigProject/Scripts/low_rank_rnn_rank5_pipeline.ipynb
```

```bash
python /Users/siyu/Documents/GitHub/biiigProject/Scripts/s2_training/low_rank_full_training.py \
  --dataset-dir /Users/siyu/Documents/GitHub/biiigProject/Data/ProcessedData \
  --output-dir /Users/siyu/Documents/GitHub/biiigProject/Results/low_rank_full_training_rank5
```

`Scripts/master_pipeline.ipynb` is still available as a broader legacy overview, but it is no longer the default first entry point for the active low-rank line.

## Current Results

### 1. Formal full-data Rank-5 result

The current formal result package is [Results/low_rank_full_training_rank5_20260626_1427](/Users/siyu/Documents/GitHub/biiigProject/Results/low_rank_full_training_rank5_20260626_1427).

What it currently supports:

- The formal run used `41` subjects and `7297` valid response-locked trials.
- Across five seeds, the model consistently recovered a low-dimensional latent aligned with CPP-like response-proximal dynamics.
- The main value of the Rank-5 model is not just prediction score. It gives a compact and more interpretable latent description of CPP-related dynamics.

For a detailed summary, see [Results/low_rank_full_training_rank5_20260626_1427/README.md](/Users/siyu/Documents/GitHub/biiigProject/Results/low_rank_full_training_rank5_20260626_1427/README.md).

### 2. Committed dual-prior comparison

The committed no-prior versus CPP-prior comparison is in [Results/rank5_dual_prior_comparison](/Users/siyu/Documents/GitHub/biiigProject/Results/rank5_dual_prior_comparison).

What stands out:

- The strongest RT-related signal in both model versions appears at about `-554 ms`.
- The strongest latent-to-RT effect is therefore not unique to only one variant.
- This supports using the no-prior model as the main interpretive anchor while still checking directional consistency against the CPP-prior comparison.

Useful files:

- [dual_prior_summary.json](/Users/siyu/Documents/GitHub/biiigProject/Results/rank5_dual_prior_comparison/dual_prior_summary.json)
- [dual_prior_performance.csv](/Users/siyu/Documents/GitHub/biiigProject/Results/rank5_dual_prior_comparison/dual_prior_performance.csv)
- [dual_prior_strongest_z_rt.csv](/Users/siyu/Documents/GitHub/biiigProject/Results/rank5_dual_prior_comparison/dual_prior_strongest_z_rt.csv)

### 3. Behavioural sequencing: RT first, drift-rate next

The repository now contains two behavioural layers with different roles:

- **RT regression** is the earlier validation step. It asks whether the latent space carries usable behavioural information at all.
- **Drift-rate regression** is the more important mechanistic follow-up. It asks whether selected latent windows help explain the externally derived drift parameter `v`.

This ordering matters because a latent space can improve RT prediction without necessarily supporting the stronger claim that it aligns with drift-like evidence-accumulation structure.

### 4. Latest local supplemental RT regression

The newest local RT validation results are in [Results/regression](/Users/siyu/Documents/GitHub/biiigProject/Results/regression).

Current takeaway:

- The early window, `-600 to -300 ms`, shows the largest RT improvement when hidden or latent information is added.
- In the local summary files, the early-window RT gain is larger than the mid and late windows.
- This makes the early pre-response period an important target for follow-up interpretation.

Useful files:

- [ridge_rt_performance.csv](/Users/siyu/Documents/GitHub/biiigProject/Results/regression/ridge_rt_performance.csv)
- [ridge_rt_deltas.csv](/Users/siyu/Documents/GitHub/biiigProject/Results/regression/ridge_rt_deltas.csv)

### 5. Drift-rate branch

This distinction matters:

- **RT regression has already been run** and is represented in the local regression outputs above.
- **A drift-rate-oriented supplementary analysis already exists** in [Scripts/s5_regression/1_latentZ_v.ipynb](/Users/siyu/Documents/GitHub/biiigProject/Scripts/s5_regression/1_latentZ_v.ipynb).
- In that notebook, the dependent variable is `v`, a drift-rate quantity reconstructed from [Data/model_traces/m5_traces.csv](/Users/siyu/Documents/GitHub/biiigProject/Data/model_traces/m5_traces.csv) and merged with [Data/joint-modeling/data_joint_modeling_all.csv](/Users/siyu/Documents/GitHub/biiigProject/Data/joint-modeling/data_joint_modeling_all.csv).
- The current drift-rate significance map and ranked summary are exported in [Results/regression/drift_rate_latent_significance.svg](/Users/siyu/Documents/GitHub/biiigProject/Results/regression/drift_rate_latent_significance.svg), [Results/regression/drift_rate_latent_significance.pdf](/Users/siyu/Documents/GitHub/biiigProject/Results/regression/drift_rate_latent_significance.pdf), and [Results/regression/drift_rate_latent_significance.csv](/Users/siyu/Documents/GitHub/biiigProject/Results/regression/drift_rate_latent_significance.csv).
- In the current local run, the strongest drift-rate-related additions come from response-proximal windows, especially `z3`, followed by `z2` and `z5`, whereas the early window is comparatively weak.
- **What is still not unified yet** is the main low-rank / dual-prior pipeline: its active processed metadata do not yet carry a standard drift-rate column in the same way they carry RT and trial descriptors.

So, at the moment, the repository contains **both RT validation and a higher-priority drift-rate branch**, but the drift-rate branch is still not fully unified into the canonical low-rank metadata workflow.

## Supplementary Branches

These are useful side branches, but they should not replace the main low-rank entry path.

- [Scripts/s5_regression](/Users/siyu/Documents/GitHub/biiigProject/Scripts/s5_regression)  
  Behavioural follow-up notebooks, including the current higher-priority latent-to-drift-rate `z` to `v` route.
- [Data/joint-modeling](/Users/siyu/Documents/GitHub/biiigProject/Data/joint-modeling)  
  Additional data prepared for joint modelling style follow-up work.
- [Data/model_traces](/Users/siyu/Documents/GitHub/biiigProject/Data/model_traces)  
  Saved traces that support side analyses and inspection.

Local exploratory scripts that are not yet part of the stable main entry route may exist in the repository, but they should be treated as supplemental checks rather than canonical starting points.

## Legacy Context

The repository still contains older GRU-era materials, legacy notebooks, and archived outputs. They remain useful for provenance and comparison, but they are no longer the default way to understand or run the active project.

For historical milestones and older workflow decisions, see [logs.md](/Users/siyu/Documents/GitHub/biiigProject/logs.md). Do not start from `archive/` unless you specifically need old provenance.

## Selected References

- [Nature Neuroscience article `s41593-022-01088-4`](https://www.nature.com/articles/s41593-022-01088-4)  
  Useful here because it supports the broader idea that low-dimensional neural population dynamics can carry interpretable computational structure.
- [Mastrogiuseppe and Ostojic, Neuron 2018](https://www.cell.com/neuron/fulltext/S0896-6273(18)30173-5)  
  Useful here because it gives the clearest theoretical foundation for why low-rank recurrent networks can produce structured, interpretable low-dimensional dynamics.

## Quick Orientation

If you only need the shortest possible route:

1. Open [Scripts/low_rank_rnn_rank5_pipeline.ipynb](/Users/siyu/Documents/GitHub/biiigProject/Scripts/low_rank_rnn_rank5_pipeline.ipynb).
2. Check [Results/low_rank_full_training_rank5_20260626_1427](/Users/siyu/Documents/GitHub/biiigProject/Results/low_rank_full_training_rank5_20260626_1427).
3. Compare against [Results/rank5_dual_prior_comparison](/Users/siyu/Documents/GitHub/biiigProject/Results/rank5_dual_prior_comparison).
4. Use [Results/regression](/Users/siyu/Documents/GitHub/biiigProject/Results/regression) first for RT validation, then for the exported drift-rate significance figure.
