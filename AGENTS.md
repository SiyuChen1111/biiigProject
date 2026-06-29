# PROJECT KNOWLEDGE BASE

**Generated:** 2026-05-12 Asia/Shanghai
**Updated:** 2026-06-29 Asia/Shanghai
**Status:** active repository root workflow

## OVERVIEW
This repository's active project files currently live at the repository root and
under the main working folders listed below.

Use the current repository structure rather than assuming a separate nested
project folder.

## DEFAULT SESSION RULE
- **Do not read `archive/` by default in new conversations.**
- Treat `archive/` as inactive historical material that should be ignored unless the user explicitly asks for provenance, recovery, comparison, or reuse of old work.
- This rule exists to avoid wasting context window / memory on retired project content.

## ACTIVE STRUCTURE
```text
biiigProject/
├── Data/            # active data inputs, processed data, and intermediate exports
├── Results/         # active model outputs, figures, tables, and checkpoints
├── Scripts/         # active preprocessing, modeling, training, and analysis code
├── tests/           # active automated checks
├── archive/         # retired repository history; do not read by default
├── AGENTS.md        # current routing and context rules
├── README.md        # project overview and current workflow notes
├── logs.md          # project-history notes
└── low_rank_rnn_drift_rate_followup_plan.md
```

## WHERE TO START
For normal work, start in this order:

1. `README.md`
2. `low_rank_rnn_drift_rate_followup_plan.md` for the current low-rank RNN / z-variable main analysis
3. `Scripts/low_rank_rnn_rank5_pipeline.ipynb` and `Scripts/low_rank_rnn_rank5_pipeline.executed.ipynb` for the current Rank-5 workflow
4. The relevant files under `Scripts/`, especially `Scripts/s1_modeling/low_rank_model.py` and `Scripts/s2_training/low_rank_smoke.py`
5. `tests/` when checking expected behavior

## CURRENT SCIENTIFIC DIRECTION
The current main line is the low-rank RNN analysis of CPP-related latent dynamics
from single-trial EEG. The Rank-5 low-rank model and its five learned `z`
variables should be treated as the priority workflow for normal analysis,
interpretation, and follow-up planning.

Current intended model direction from the active documents:
- response-locked single-trial EEG input
- channels focused on `CP1`, `CP2`, `CPz`
- compact Rank-5 low-rank RNN as the primary model path
- five learned low-rank latent variables, `z1` through `z5`, at every trial and time point
- self-supervised future prediction plus reconstruction
- downstream z-space analyses for response time, CPP dynamics, drift-rate-like effects, and evidence-accumulation behavior
- the earlier GRU latent-state model remains useful as background or comparison, but it is not the default main line

## INACTIVE / ARCHIVED MATERIAL
- `archive/` contains the former repo root content, including legacy pipeline code, outputs, documents, and plans.
- Do not assume archived scripts, paths, or conclusions remain current.
- If archived material must be used, explicitly say why it is being consulted and keep the read scope narrow.

## WORKING CONVENTIONS
- Run normal reads, planning, and implementation against the active root-level project folders unless instructed otherwise.
- If you need code scaffolding for the current phase, place it under the appropriate active folder such as `Scripts/`, `tests/`, `Data/`, or `Results/`, not under `archive/`.
- Hidden repo-control paths such as `.git/` and `.sisyphus/` are infrastructure, not project content.
