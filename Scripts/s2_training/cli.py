from __future__ import annotations

import argparse
import sys
from pathlib import Path
import importlib

# ---------------------------------------------------------------------------
# Path bootstrap: make s1_modeling, s2_training, and s4_analysis importable
# as "modeling", "training", and "analysis" when this CLI is invoked directly.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent   # Scripts/
_p = str(_ROOT)
if _p not in sys.path:
    sys.path.insert(0, _p)

sys.modules.setdefault("modeling", importlib.import_module("s1_modeling"))
sys.modules.setdefault("training", importlib.import_module("s2_training"))
sys.modules.setdefault("analysis", importlib.import_module("s4_analysis"))

from modeling.config import (
    AnalysisConfig,
    DataContractConfig,
    TrainingConfig,
    default_evidence_dir,
)
from modeling.data_contract import validate_stage2_dataset
from modeling.preparation import (
    audit_preliminary_stage2_dataset,
    prepare_stage2_dataset_package,
)
from training.controls import run_minimal_controls
from training.sweep import run_small_cpp_prior_sweep
from training.train import export_full_latents_from_checkpoint, train_model
from analysis.rt_ridge import run_ridge_rt_analysis


# =============================================================================
# § 1  Argument Parser
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="stage2",
        description=(
            "Stage 2 response-locked CPP latent-dynamics pipeline.\n\n"
            "Commands follow the analysis flow:\n"
            "  s0: prepare  →  s1: validate  →  s2: train  →  s2: sweep\n"
            "      extract-latents  →  s4: ridge-rt  →  s4: controls"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "validate",
            "train",
            "controls",
            "sweep",
            "extract-latents",
            "ridge-rt",
        ),
        help="Pipeline step to run.",
    )
    parser.add_argument(
        "--dataset-dir", type=Path, required=True,
        help="Path to the processed dataset directory (contains eeg_cpp_trials.npy etc.).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Root output directory. Defaults to Results/stage2_modeling next to dataset-dir.",
    )
    parser.add_argument(
        "--checkpoint-path", type=Path, default=None,
        help="Path to best_model.pt (required for extract-latents).",
    )
    parser.add_argument(
        "--latent-path", type=Path, default=None,
        help="Path to latents_full.npz (required for controls and ridge-rt).",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Torch device string, e.g. 'cpu', 'cuda', or 'auto'.",
    )
    return parser


# =============================================================================
# § 2  Command Dispatch
# =============================================================================

def main() -> None:
    """Command-line entry point for the Stage 2 pipeline."""
    parser = _build_parser()
    args = parser.parse_args()
    output_dir = args.output_dir or default_evidence_dir(args.dataset_dir.parent)

    # ── s0: prepare ───────────────────────────────────────────────────────
    if args.command == "prepare":
        prepare_stage2_dataset_package(args.dataset_dir, output_dir / "s0_prepare")
        audit_preliminary_stage2_dataset(args.dataset_dir, output_dir / "s0_prepare")

    # ── s1: validate ──────────────────────────────────────────────────────
    elif args.command == "validate":
        validate_stage2_dataset(
            args.dataset_dir, output_dir / "s1_validate", DataContractConfig()
        )

    # ── s2: train ─────────────────────────────────────────────────────────
    elif args.command == "train":
        train_model(
            dataset_dir=args.dataset_dir,
            output_dir=output_dir / "s2_training",
            config=TrainingConfig(),
        )

    # ── s2: sweep ─────────────────────────────────────────────────────────
    elif args.command == "sweep":
        run_small_cpp_prior_sweep(
            dataset_dir=args.dataset_dir,
            output_dir=output_dir / "s2_sweep",
            base_config=TrainingConfig(),
        )

    # ── s2: extract-latents ───────────────────────────────────────────────
    elif args.command == "extract-latents":
        if args.checkpoint_path is None:
            parser.error("--checkpoint-path is required for extract-latents")
        export_full_latents_from_checkpoint(
            checkpoint_path=args.checkpoint_path,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir or (args.dataset_dir / "latents_full"),
        )

    # ── s4: ridge-rt ──────────────────────────────────────────────────────
    elif args.command == "ridge-rt":
        if args.latent_path is None:
            parser.error("--latent-path is required for ridge-rt")
        run_ridge_rt_analysis(
            latent_npz=args.latent_path,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir or (args.dataset_dir / "ridge_rt"),
        )

    # ── s4: controls ──────────────────────────────────────────────────────
    elif args.command == "controls":
        run_minimal_controls(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir or (args.dataset_dir / "controls"),
            config=TrainingConfig(),
        )


if __name__ == "__main__":
    main()
