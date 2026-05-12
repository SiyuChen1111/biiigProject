# Stage2 Modeling Execution Decisions

## First unit of inference

- **Selected option**: **Option A**
- **Meaning**: pooled-subject first-pass training with subject-aware normalization and a subject-held-out evaluation path prepared.

## Why this default was applied

- It matches the recommendation in `stage2-modeling-roadmap.md`.
- It exposes data-contract and normalization issues earlier than a single-subject-only pilot.
- It keeps the first milestone aligned with the proposal's goal of generalizable latent structure rather than a narrowly tuned subject-specific proof of concept.

## Operational constraints for the first milestone

- Training remains **self-supervised first**.
- Only `CP1`, `CP2`, and `CPz` are accepted for the primary pipeline.
- The main model remains a **causal forward GRU**.
- Primary loss is computed only on the protected pre-response window `t <= RT - 50 ms`.
- Subject-level retained-trial summaries are required before model training is considered valid.
