# Project Reset Log

## 2026-05-12 — Repository reset around `stage2_YuYNet`

### What changed
- Moved the previous root-level project materials into `archive/`.
- Kept `stage2_YuYNet/` at the repository root as the only active project area.
- Replaced the root `AGENTS.md` so future sessions default to `stage2_YuYNet/` and do not read `archive/` unless explicitly requested.

### What was archived
Archived former root-level project content including:
- `1_Data/`
- `2_Analysis/`
- previous `AGENTS.md`
- `docs/`
- `low-level_1.pdf`
- `outputs/`
- `README.md`
- `requirements.txt`
- `scripts/`
- `src/`
- `初步介绍.md`

### Why this reset was made
The project focus has changed.

The old repository structure represented an earlier CPP/EEG workflow with broader analysis code, legacy plans, and retained outputs. That material is still preserved for historical reference, but it is no longer the active working surface.

The active direction is now the stage-2 latent-dynamics program centered on:
- `stage2_YuYNet/EEG_preprocessing_request_for_partner.md`
- `stage2_YuYNet/CPP_latent_dynamics_scientific_proposal.md`

The main goal is to build a neural-network model for CPP-related single-trial EEG latent dynamics.

### Prior-plan status
- Prior repository-wide active plans are considered **retired / superseded**.
- Older planning documents, workflows, and assumptions should be treated as historical context only.
- Any future implementation planning should be based on the stage-2 documents, not on the archived project structure.

### Operational rule going forward
- Default all new conversations to `stage2_YuYNet/`.
- Avoid reading `archive/` during normal work to prevent unnecessary context and memory usage.
- Use archived material only when explicitly needed for comparison, recovery, or provenance.
