from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


# =============================================================================
# § 1  Data Contract Configuration
# =============================================================================

@dataclass(frozen=True)
class DataContractConfig:
    """Stage 1 contract: expected files, channels, and required metadata columns.

    Encodes the structural requirements that the processed EEG dataset directory
    must satisfy before any model training can begin.
    """

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


# =============================================================================
# § 2  Model Architecture Configuration
# =============================================================================

@dataclass(frozen=True)
class ModelConfig:
    """Architectural hyperparameters for CPPForwardGRU.

    Separating architecture from training and loss concerns makes it easy to
    swap encoder depth or projection size without touching loss weights.
    """

    projection_dim: int = 16
    """Dimensionality of the linear input-projection layer (before LayerNorm)."""

    hidden_dim: int = 32
    """GRU hidden-state dimensionality; also the latent-space dimensionality."""

    num_layers: int = 1
    """Number of stacked GRU layers (set > 1 for deeper recurrent encoding)."""


# =============================================================================
# § 3  Loss Weights & Shape-Prior Configuration
# =============================================================================

@dataclass(frozen=True)
class LossWeights:
    """Weights and windows that control the self-supervised composite loss.

    Each ``lambda_*`` field scales one loss term.  Setting a weight to 0.0
    effectively disables that term without touching the model or training loop.
    The CPP shape-prior group (monotonic / slope_floor / late_amplitude /
    cpp_mean_alignment) can be disabled wholesale via ``enable_cpp_shape_prior``.

    Loss term overview
    ------------------
    lambda_recon             : MSE between reconstructed and real EEG (per-channel).
    lambda_future            : MSE between predicted and real future EEG.
    lambda_derivative        : MSE between reconstructed and real first-order slope.
    lambda_variance          : Channel-level variance alignment between recon and real.
    lambda_cpp_mean          : MSE on the 3-channel CPP proxy (mean across channels).
    lambda_cpp_prior         : Global scale for the CPP shape-prior sub-group.
    lambda_monotonic         : Penalises downward steps in the reconstructed CPP.
    lambda_slope_floor       : Penalises recon slope falling below a fraction of target.
    lambda_late_amplitude    : Penalises under-shooting the late CPP amplitude.
    lambda_cpp_mean_alignment: Alignment loss on CPP proxy over the analysis window.
    lambda_smooth            : Penalises large frame-to-frame jumps in latent state.
    """

    lambda_recon: float = 1.0
    lambda_future: float = 0.2
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
    """Scaling factor applied to future-prediction mask weights."""

    slope_floor_ratio: float = 0.5
    """Fraction of the target slope used as the floor for slope_floor_loss."""

    enable_cpp_shape_prior: bool = True
    """Toggle the entire CPP shape-prior sub-group on or off."""

    # Time windows (ms, response-locked) used inside the loss computation.
    analysis_window_ms: Tuple[float, float] = (-600.0, -50.0)
    late_window_ms: Tuple[float, float] = (-120.0, -50.0)


# =============================================================================
# § 4  Training Loop & Data-Pipeline Configuration
# =============================================================================

@dataclass(frozen=True)
class TrainingConfig:
    """Training loop, data-pipeline, and split configuration.

    Architecture hyperparameters live in the nested ``model`` field
    (a :class:`ModelConfig` instance), and all loss weights live in the
    nested ``loss`` field (a :class:`LossWeights` instance).

    Example
    -------
    >>> cfg = TrainingConfig(
    ...     max_epochs=50,
    ...     model=ModelConfig(hidden_dim=64),
    ...     loss=LossWeights(lambda_smooth=0.01, enable_cpp_shape_prior=False),
    ... )
    """

    # --- Reproducibility --------------------------------------------------
    seed: int = 42

    # --- Data pipeline ----------------------------------------------------
    batch_size: int = 64
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    future_horizon_ms: int = 50
    """Length of the causal prediction horizon in milliseconds."""

    # --- Optimiser --------------------------------------------------------
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0

    # --- Training schedule ------------------------------------------------
    max_epochs: int = 100
    early_stopping_patience: int = 15

    # --- Time-window helpers (used by dataset weighting and analysis) ------
    analysis_window_ms: Tuple[float, float] = (-600.0, -50.0)
    early_window_ms: Tuple[float, float] = (-600.0, -300.0)
    mid_window_ms: Tuple[float, float] = (-300.0, -120.0)
    late_window_ms: Tuple[float, float] = (-120.0, -50.0)

    # --- Nested sub-configurations ----------------------------------------
    model: ModelConfig = field(default_factory=ModelConfig)
    """Architecture hyperparameters (projection_dim, hidden_dim, num_layers)."""

    loss: LossWeights = field(default_factory=LossWeights)
    """All loss weights, shape-prior flags, and loss-related time windows."""

    # ------------------------------------------------------------------
    # Backwards-compatibility shims
    # ------------------------------------------------------------------
    # The properties below expose the most-used sub-fields directly on
    # TrainingConfig so that existing call sites (model instantiation,
    # sweep parameter replacement) continue to work without modification.
    # New code should prefer ``config.model.*`` and ``config.loss.*``.

    @property
    def projection_dim(self) -> int:
        """Shortcut → config.model.projection_dim."""
        return self.model.projection_dim

    @property
    def hidden_dim(self) -> int:
        """Shortcut → config.model.hidden_dim."""
        return self.model.hidden_dim

    @property
    def num_layers(self) -> int:
        """Shortcut → config.model.num_layers."""
        return self.model.num_layers

    @property
    def lambda_recon(self) -> float:
        return self.loss.lambda_recon

    @property
    def lambda_future(self) -> float:
        return self.loss.lambda_future

    @property
    def lambda_derivative(self) -> float:
        return self.loss.lambda_derivative

    @property
    def lambda_variance(self) -> float:
        return self.loss.lambda_variance

    @property
    def lambda_cpp_mean(self) -> float:
        return self.loss.lambda_cpp_mean

    @property
    def lambda_cpp_prior(self) -> float:
        return self.loss.lambda_cpp_prior

    @property
    def lambda_monotonic(self) -> float:
        return self.loss.lambda_monotonic

    @property
    def lambda_slope_floor(self) -> float:
        return self.loss.lambda_slope_floor

    @property
    def lambda_late_amplitude(self) -> float:
        return self.loss.lambda_late_amplitude

    @property
    def lambda_cpp_mean_alignment(self) -> float:
        return self.loss.lambda_cpp_mean_alignment

    @property
    def lambda_smooth(self) -> float:
        return self.loss.lambda_smooth

    @property
    def future_weight_scale(self) -> float:
        return self.loss.future_weight_scale

    @property
    def slope_floor_ratio(self) -> float:
        return self.loss.slope_floor_ratio

    @property
    def enable_cpp_shape_prior(self) -> bool:
        return self.loss.enable_cpp_shape_prior


# =============================================================================
# § 5  Analysis / Readout Configuration
# =============================================================================

@dataclass(frozen=True)
class AnalysisConfig:
    """Stage 3 & 4 latent-readout and decoding settings.

    Governs PCA decomposition, RT-bin boundaries, and the time windows used
    for latent-state analyses after training.
    """

    response_locked_window_ms: Tuple[int, int] = (-600, -50)
    """Analysis window used for response-locked latent extraction."""

    contaminated_window_ms: Tuple[int, int] = (-50, 100)
    """Post-response window excluded from causal analyses."""

    pca_components: int = 3
    """Number of principal components retained in latent PCA."""

    rt_bin_quantiles: Tuple[float, float] = (0.33, 0.66)
    """Quantile boundaries for fast / medium / slow RT tertile binning."""

    evidence_bin_quantiles: Tuple[float, ...] = (0.50,)
    """Quantile boundaries for low / high evidence median split."""


# =============================================================================
# § 6  Path Helpers
# =============================================================================

def default_evidence_dir(root: Path) -> Path:
    """Return the canonical evidence output directory for a project root."""
    return root / "Results" / "stage2_modeling"
