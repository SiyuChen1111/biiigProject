from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import run_core_latent_analysis, run_pooled_latent_analysis
from .config import AnalysisConfig, DataContractConfig, TrainingConfig, default_evidence_dir
from .controls import run_minimal_controls
from .data_contract import validate_stage2_dataset
from .prepare_contract import audit_preliminary_stage2_dataset
from .preparation import prepare_stage2_dataset_package
from .sweep import run_cpp_prior_sweep
from .train import train_stage2_pipeline


def main() -> None:
    """Command-line entry point for the stage2 pipeline."""
    parser = argparse.ArgumentParser(description="Stage2 response-locked CPP latent-dynamics baseline pipeline")
    parser.add_argument("command", choices=("prepare", "validate", "train", "analyze", "controls", "sweep"))
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--latent-path", type=Path)
    parser.add_argument("--latent-paths", type=Path, nargs="+")
    parser.add_argument("--analysis-label", default="single_split")
    args = parser.parse_args()

    output_dir = args.output_dir or default_evidence_dir(args.dataset_dir.parent)

    if args.command == "prepare":
        prepare_stage2_dataset_package(args.dataset_dir, output_dir / "stage0")
        audit_preliminary_stage2_dataset(args.dataset_dir, output_dir / "stage0")
    elif args.command == "validate":
        validate_stage2_dataset(args.dataset_dir, output_dir / "stage1", DataContractConfig())
    elif args.command == "train":
        train_stage2_pipeline(args.dataset_dir, output_dir / "stage2", TrainingConfig())
    elif args.command == "analyze":
        if args.latent_paths:
            stage_dir = output_dir / ("stage3_pooled" if args.analysis_label == "pooled_train_val_test" else f"stage3_{args.analysis_label}")
            run_pooled_latent_analysis(
                args.latent_paths,
                stage_dir,
                AnalysisConfig(),
                dataset_dir=args.dataset_dir,
                analysis_label=args.analysis_label,
            )
        else:
            if args.latent_path is None:
                raise SystemExit("--latent-path or --latent-paths is required for analyze")
            run_core_latent_analysis(args.latent_path, output_dir / "stage3", AnalysisConfig(), dataset_dir=args.dataset_dir)
    elif args.command == "controls":
        if args.latent_path is None:
            raise SystemExit("--latent-path is required for controls")
        run_minimal_controls(args.latent_path, output_dir / "stage4")
    elif args.command == "sweep":
        run_cpp_prior_sweep(args.dataset_dir, output_dir / "sweep_cpp_prior", TrainingConfig())


if __name__ == "__main__":
    main()
