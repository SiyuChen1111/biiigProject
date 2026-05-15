from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class DataContractConfig:
    """Stage 1 contract for the active CPP pipeline."""

    expected_files: Tuple[str, ...] = (
        "eeg_cpp_trials.npy",
        "metadata.csv",
        "times_ms.npy",
        "channel_names.txt",
        "preprocessing_notes.md",
    )
    expected_channel_order: Tuple[str, ...] = ("CP1", "CP2", "CPz")
    required_metadata_columns: Tuple[str, ...] = (
        "trial_id",
        "alignment",
    )
    optional_aliases: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingConfig:
    """Minimal Stage 2 pooled-subject training configuration."""

    seed: int = 42
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    max_epochs: int = 100
    early_stopping_patience: int = 15
    projection_dim: int = 16
    hidden_dim: int = 32
    num_layers: int = 1
    future_horizon_ms: int = 50
    lambda_future: float = 0.2
    lambda_recon: float = 1.0
    lambda_derivative: float = 0.5
    lambda_variance: float = 0.5
    lambda_cpp_mean: float = 0.5
    lambda_cpp_prior: float = 0.1
    lambda_monotonic: float = 1.0
    lambda_slope_floor: float = 0.5
    lambda_late_amplitude: float = 1.0
    lambda_cpp_mean_alignment: float = 0.05
    lambda_smooth: float = 0.001
    future_weight_scale: float = 0.75
    slope_floor_ratio: float = 0.5
    enable_cpp_shape_prior: bool = True
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    analysis_window_ms: Tuple[float, float] = (-600.0, -50.0)
    early_window_ms: Tuple[float, float] = (-600.0, -300.0)
    mid_window_ms: Tuple[float, float] = (-300.0, -120.0)
    late_window_ms: Tuple[float, float] = (-120.0, -50.0)


@dataclass(frozen=True)
class AnalysisConfig:
    """Stage 3 and 4 latent readout settings."""

    response_locked_window_ms: Tuple[int, int] = (-600, -50)
    contaminated_window_ms: Tuple[int, int] = (-50, 100)
    pca_components: int = 3
    rt_bin_quantiles: Tuple[float, float] = (0.33, 0.66)
    evidence_bin_quantiles: Tuple[float, ...] = (0.50,)


def default_evidence_dir(root: Path) -> Path:
    return root / "evidence" / "stage2_modeling"
