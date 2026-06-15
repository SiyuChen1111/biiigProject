# biiigProject — CPP Latent Dynamics

**Scientific goal:** Model CPP (Centro-Parietal Positivity) related EEG latent dynamics
using a causal forward GRU, and connect the learned latent representations to human
decision behaviour (reaction time, evidence accumulation).

**Status:** Model training ✅ · Neural validation ✅ · Ridge RT regression ✅ · DDM regression 🔲

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone https://github.com/siyu/biiigProject
cd biiigProject
pip install torch numpy pandas scikit-learn matplotlib

# 2. Run the full pipeline (interactive, step-by-step)
jupyter notebook Scripts/master_pipeline.ipynb

# 3. Run tests
python -m pytest tests/ -v
```

---

## Project Structure

See [`Scripts/pipeline_overview.md`](Scripts/pipeline_overview.md) for the full
annotated directory tree and per-stage commands.

```
biiigProject/
├── Data/               ← InputData / ProcessedData / IntermediateData
├── Scripts/
│   ├── master_pipeline.ipynb   ← Master Script (TIER 4.0)
│   ├── pipeline_overview.md    ← Stage-by-stage command reference
│   ├── s0_preprocessing/       ← EEG → trial arrays
│   ├── s1_modeling/            ← model library (config, model, dataset)
│   ├── s2_training/            ← training, sweep, controls, CLI
│   ├── s3_validation/          ← validation routing notes / legacy reference
│   └── s4_analysis/            ← Ridge regression, PCA, figures
├── Results/            ← checkpoints / validation / regression / figures
├── tests/              ← pytest suite
├── logs.md             ← experiment log (all dated entries)
└── AGENTS.md           ← AI agent operating instructions
```

---

## Key Results (as of 2026-06-10)

| Metric | Value |
|--------|-------|
| Neural reconstruction R² | 0.88 |
| CPP ERP waveform correlation | 0.99 |
| Hidden → CPP amplitude (delta R²) | 0.95 |
| Best RT window (−600 to −300 ms) R² | 0.30 (vs 0.197 baseline) |
| Choice / condition decoding | ~chance |

---

## Code Architecture

The `modeling` library (`Scripts/s1_modeling/`) uses a three-level config hierarchy:

```python
TrainingConfig(
    model = ModelConfig(hidden_dim=32, projection_dim=16),
    loss  = LossWeights(lambda_recon=1.0, lambda_cpp_prior=0.1, ...),
    ...   # training loop params
)
```

The `CPPForwardGRU` model produces `ForwardOutputs(reconstructed, predicted, latents)`.
The loss function `masked_self_supervised_loss(outputs, ..., weights: LossWeights)` is
decomposed into three private sub-functions for readability.

The active validation entrypoints now live under `Results/validation/`, which also
stores the generated audit tables, summaries, and publication-style figures.

---

## Experiment Log

See [`logs.md`](logs.md) for the full dated experiment log including all key findings,
intermediate results, and next-step decisions.

---

## Reference

- Dataset: Kosciessa et al. (2021); van den Berg et al. (2019)  
- Protocol: [TIER Protocol 4.0](https://www.projecttier.org/tier-protocol/protocol-4-0/)
