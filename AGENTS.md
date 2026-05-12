# PROJECT KNOWLEDGE BASE

**Generated:** 2026-05-12 Asia/Shanghai
**Status:** repository reset around `stage2_YuYNet`

## OVERVIEW
This repository has been intentionally re-centered around `stage2_YuYNet/`.

The only active project area for normal work is:

```text
/Users/siyu/Documents/GitHub/biiigProject/stage2_YuYNet/
```

Everything else from the earlier project phase has been preserved under `archive/` for reference only.

## DEFAULT SESSION RULE
- **Do not read `archive/` by default in new conversations.**
- Treat `archive/` as inactive historical material that should be ignored unless the user explicitly asks for provenance, recovery, comparison, or reuse of old work.
- This rule exists to avoid wasting context window / memory on retired project content.

## ACTIVE STRUCTURE
```text
biiigProject/
├── stage2_YuYNet/   # the only active project area
├── archive/         # retired repository history; do not read by default
├── AGENTS.md        # current routing and context rules
└── logs.md          # reset log and project-history notes
```

## WHERE TO START
For normal work, start in this order:

1. `stage2_YuYNet/CPP_latent_dynamics_scientific_proposal.md`
2. `stage2_YuYNet/EEG_preprocessing_request_for_partner.md`
3. `stage2_YuYNet/script_pre_EEG/` only if preprocessing seed materials are needed

## CURRENT SCIENTIFIC DIRECTION
The active goal is to build a neural-network model for CPP-related latent dynamics from single-trial EEG.

Current intended model direction from the active documents:
- stimulus-locked single-trial EEG input
- channels focused on `CP1`, `CP2`, `CPz`
- causal forward GRU latent-state model
- self-supervised future prediction plus reconstruction
- downstream latent-space analyses for evidence-accumulation behavior

## INACTIVE / ARCHIVED MATERIAL
- `archive/` contains the former repo root content, including legacy pipeline code, outputs, documents, and plans.
- Do not assume archived scripts, paths, or conclusions remain current.
- If archived material must be used, explicitly say why it is being consulted and keep the read scope narrow.

## WORKING CONVENTIONS
- Run normal reads, planning, and implementation only against `stage2_YuYNet/` unless instructed otherwise.
- If you need code scaffolding for the new phase, place it under `stage2_YuYNet/`, not under `archive/`.
- Hidden repo-control paths such as `.git/` and `.sisyphus/` are infrastructure, not project content.
