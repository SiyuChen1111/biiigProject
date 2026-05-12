<!--
This document is a normative contract for ROI-only conditional generation outputs.
Downstream smoke/evidence scripts should rely on these exact paths and filenames.
-->

# ROI Conditional Generator — Retained Output Contract

This repository retains selected experiment outputs under `outputs/` as evidence.
To prevent manual file moves and ad-hoc naming, ROI-only conditional generation runs MUST follow
the layout below.

## Canonical output roots

All ROI conditional generator artifacts live under **one of** the following roots:

- **Smoke** runs: `outputs/roi_conditional_generator_smoke/`
- **Evidence** runs: `outputs/roi_conditional_generator_evidence/`

These names are intentionally stable (no timestamps) so they can be referenced directly in docs and
scripts.

## Contract version

- `contract_id`: `roi_conditional_generator_v1`

If the structure must change, bump the contract ID and update the scripts to write both the new
layout and the ID.

## Directory layout (must match exactly)

Both output roots share the same internal structure:

```text
outputs/roi_conditional_generator_{smoke|evidence}/
├── run_manifest.json
├── train/
│   ├── train_run_config.json
│   ├── fold_losses.json
│   ├── summary_losses.json
│   ├── fold_1/
│   │   └── conditional_samples.npy
│   ├── fold_2/
│   │   └── conditional_samples.npy
│   ├── ...
│   └── fold_5/
│       └── conditional_samples.npy
└── analysis/
    ├── analysis_run_config.json
    ├── fold_generation_summary.json
    ├── fold_generation_summary.csv
    ├── aggregate_generation_summary.json
    ├── aggregate_generation_summary.csv
    ├── cpp_style_roi_summary.png
    ├── cpp_style_roi_summary.pdf
    ├── real_only_cpp_target.png
    ├── real_only_cpp_target.pdf
    ├── fold_1/
    │   └── real_vs_generated_roi_erp.png
    ├── fold_2/
    │   └── real_vs_generated_roi_erp.png
    ├── ...
    └── fold_5/
        └── real_vs_generated_roi_erp.png
```

## File meanings

### `run_manifest.json` (root)

Single, root-level metadata blob for later provenance checks.

Minimum required keys:

```json
{
  "contract_id": "roi_conditional_generator_v1",
  "created_utc": "2026-04-22T00:00:00Z",
  "train_dir": "train",
  "analysis_dir": "analysis"
}
```

Scripts may include additional keys (e.g. git commit hash) but MUST preserve the minimum keys.

### `train/`

- `train_run_config.json`: JSON dump of the training script CLI arguments (exact flags used).
- `fold_losses.json`: fold-level loss parts for each fold (list of dicts, one per fold).
- `summary_losses.json`: mean/std summary derived from `fold_losses.json`.
- `fold_{k}/conditional_samples.npy`: conditional sample tensor saved per fold.

### `analysis/`

- `analysis_run_config.json`: JSON dump of the analysis script CLI arguments (exact flags used).
- `fold_generation_summary.json`: fold-level comparison summary (e.g. real-vs-generated gap).
- `fold_generation_summary.csv`: CSV export of the per-fold comparison summary.
- `aggregate_generation_summary.json`: aggregate across-fold summary with mean/std and per-condition metrics.
- `aggregate_generation_summary.csv`: single-row CSV export of the aggregate summary.
- `cpp_style_roi_summary.png` / `.pdf`: publication-style comparison figure showing real vs generated ROI waveforms, CPP difference wave, and mismatch panel.
- `real_only_cpp_target.png` / `.pdf`: classic held-out real CPP target figure used to show what the real ROI CPP morphology should look like.
- `fold_{k}/real_vs_generated_roi_erp.png`: per-fold ROI ERP comparison plot.

## Implementation rules (non-negotiable)

- Do not write ROI generator artifacts into `outputs/phase1_*` directories.
- Keep fold artifacts inside `fold_{k}/` directories.
- Keep summary artifacts at the `train/` and `analysis/` roots with the exact filenames above.
- A smoke/evidence run should be verifiable by a simple `ls -R` tree listing against this contract.
