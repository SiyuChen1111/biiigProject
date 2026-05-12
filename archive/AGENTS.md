# PROJECT KNOWLEDGE BASE

**Generated:** 2026-04-22 Asia/Shanghai
**Commit:** 389c88f
**Branch:** main

## OVERVIEW
Python-first EEG research repository for CPP-related analysis on the Kosciessa `task-dynamic` dataset. Current active workflow is a TIER-inspired transitional layout: reusable code in `src/`, runnable experiment CLIs in `scripts/`, retained protocol docs in `docs/`, and a legacy notebook pipeline in `2_Analysis/`.

## STRUCTURE
```text
biiigProject/
├── src/                 # reusable loaders, preprocessing, feature logic, evaluation, models
├── scripts/             # canonical runnable experiment entrypoints
├── docs/                # notes + protocol/blueprint docs
├── 2_Analysis/          # legacy notebook-era pipeline and outputs
├── 1_Data/              # local raw dataset mirror; not source code
├── outputs/             # retained experiment evidence; mostly generated artifacts
├── README.md            # repo navigation + current workflow summary
└── requirements.txt     # minimal Python dependency baseline
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Start here | `README.md` | repo layout, current scientific status, canonical script examples |
| Main command router | `scripts/run_master.py` | dispatches supported phase-1/phase-2 discriminative runs |
| Dataset path logic | `src/data/data_kosciessa.py` | prefers `data/raw/...`, falls back to `1_Data/...` |
| Epoch extraction | `src/preprocessing/epoching.py` | response-locked epochs, filtering, resampling |
| CPP features / labels | `src/features/labels_cpp.py` | ROI channels, AMS/PAMS/SLPS, quartile thresholds |
| Shared metrics / plots | `src/evaluation/evaluate_phase1.py` | fold metrics, subject metrics, ROI ERP plots |
| Main discriminative model | `src/models/EEGModels_PyTorch.py` | EEGNet backbone + feature API |
| Conditional generation | `src/models/conditional_eeg_vae.py` | VAE branch built on EEGNet features |
| Active protocol docs | `docs/protocol/` | stage summary, retained-results index, implementation blueprints |
| Legacy pipeline | `2_Analysis/CPP_PIPELINE_README.md` | explains old notebook/module workflow |

## CODE MAP
| Symbol | Location | Role |
|--------|----------|------|
| `resolve_dataset_dir` | `src/data/data_kosciessa.py` | raw-data path selection |
| `extract_response_locked_epochs` | `src/preprocessing/epoching.py` | shared epoching hub used by all modern phase scripts |
| `compute_cpp_features` | `src/features/labels_cpp.py` | shared CPP feature extraction |
| `fit_cpp_label_transform` | `src/features/labels_cpp.py` | train-fold-only label statistics + thresholds |
| `compute_binary_metrics` | `src/evaluation/evaluate_phase1.py` | shared evaluation primitive |
| `ConditionalEEGVAE` | `src/models/conditional_eeg_vae.py` | conditional EEG generation model |
| `main` | `scripts/run_master.py` | CLI dispatcher |

## CONVENTIONS
- Run everything from the **repository root**.
- Modern executable surface lives in `scripts/`; reusable code lives in `src/`.
- `src` itself is the import root. Scripts commonly inject repo root into `sys.path` so `from src...` works without packaging.
- Raw data is local-only. Preferred path: `data/raw/CPP_low-level-2_Kosciessa_et_al_2021/`; fallback path: `1_Data/CPP_low-level-2_Kosciessa_et_al_2021/`.
- `outputs/` is intentionally retained evidence, not disposable scratch output.
- `2_Analysis/` is legacy/reference workflow. Useful for context; avoid adding new core logic there unless you are explicitly maintaining the legacy path.

## ANTI-PATTERNS (THIS PROJECT)
- Do **not** split train/val/test randomly by trial. Split by **subject**.
- Do **not** fit CPP score thresholds on validation/test/full data. Fit transforms on **training fold only**.
- Do **not** treat `events.tsv` as the authoritative trial definition. Behavior table is primary; event files are consistency checks.
- Do **not** put new source logic inside `1_Data/`, `outputs/`, or per-fold/per-subject directories.
- Do **not** assume `output/` and `outputs/` are the same. This repo uses tracked `outputs/`; `.gitignore` still ignores legacy `output/`.
- Do **not** load arbitrary checkpoints unsafely. Prefer `torch.load(..., weights_only=True)` patterns already used in active scripts.

## UNIQUE STYLES
- Research decisions are documented unusually well in `docs/protocol/`; read protocol docs before changing pipeline semantics.
- Phase naming is meaningful: `phase1_*`, `phase2_*`, and `conditional_vae_*` correspond to distinct scientific stages.
- Active pipeline favors deterministic subject ordering and deterministic fold assignment over randomized convenience.

## COMMANDS
```bash
pip install -r requirements.txt

python3 scripts/run_master.py phase1 -- --max-subjects 15 --epochs 40 --output-dir outputs/phase1_midrun
python3 scripts/run_master.py phase1-baseline -- --max-subjects 15 --output-dir outputs/phase1_baseline_midrun
python3 scripts/run_master.py phase2-cue -- --max-subjects 10 --epochs 10 --output-dir outputs/phase2_quickcheck
python3 scripts/run_master.py phase2-rt -- --max-subjects 5 --epochs 2 --output-dir outputs/phase2_rt_smoketest

python3 scripts/train_phase1_conditional_vae.py --output-dir outputs/phase1_conditional_vae_auxquick
python3 scripts/analyze_conditional_vae_samples.py --vae-output-dir outputs/phase1_conditional_vae_auxquick --analysis-output-dir outputs/phase1_conditional_vae_auxquick_analysis
```

## NOTES
- No conventional `tests/` harness, linter config, CI workflow, or packaging metadata is present. Validation is currently script-driven.
- Deep trees under `1_Data/` and `outputs/` are storage mirrors (`sub-*`, `fold_*`), not documentation boundaries.
- `phase1-eegnet-implementation-blueprint.md` exists both at repo root and under `docs/protocol/`; prefer the `docs/protocol/` copy as the canonical documentation location.
