from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .config import LossWeights, ModelConfig


# =============================================================================
# § 1  Output Container
# =============================================================================

@dataclass
class ForwardOutputs:
    """Named container for all outputs produced by a single CPPForwardGRU forward pass.

    Attributes
    ----------
    reconstructed : (B, T, C) tensor — per-channel EEG reconstruction at each time step.
    predicted     : (B, T, C) tensor — one-step-ahead causal prediction at each time step.
    latents       : (B, T, H) tensor — GRU hidden states (the learned latent representation).
    """

    reconstructed: torch.Tensor
    predicted: torch.Tensor
    latents: torch.Tensor


# =============================================================================
# § 2  Model Definition
# =============================================================================

class CPPForwardGRU(nn.Module):
    """Causal GRU encoder for CPP-related EEG latent dynamics.

    Architecture (left-to-right)
    ----------------------------
    Input  (B, T, C)
      └─ Linear projection  C → projection_dim
      └─ LayerNorm
      └─ GRU (causal, num_layers stacked)   hidden = hidden_dim
      └─ Reconstruction head  hidden_dim → C   (what the model saw)
      └─ Prediction head      hidden_dim → C   (what comes next)

    Parameters
    ----------
    n_channels   : Number of EEG channels in the input (typically 3: CP1, CP2, CPz).
    model_config : :class:`ModelConfig` instance carrying projection_dim, hidden_dim,
                   and num_layers.  Pass ``cfg.model`` from a :class:`TrainingConfig`.
    """

    def __init__(self, n_channels: int, model_config: ModelConfig) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.cfg = model_config

        # --- Input projection -------------------------------------------------
        self.input_proj = nn.Linear(n_channels, model_config.projection_dim)
        self.input_norm = nn.LayerNorm(model_config.projection_dim)

        # --- Causal recurrent encoder ----------------------------------------
        # batch_first=True keeps the (B, T, *) convention throughout.
        self.gru = nn.GRU(
            input_size=model_config.projection_dim,
            hidden_size=model_config.hidden_dim,
            num_layers=model_config.num_layers,
            batch_first=True,
        )

        # --- Output heads -----------------------------------------------------
        self.recon_head = nn.Sequential(
            nn.Linear(model_config.hidden_dim, model_config.hidden_dim),
            nn.ReLU(),
            nn.Linear(model_config.hidden_dim, n_channels),
        )
        self.pred_head = nn.Sequential(
            nn.Linear(model_config.hidden_dim, model_config.hidden_dim),
            nn.ReLU(),
            nn.Linear(model_config.hidden_dim, n_channels),
        )

    def forward(self, x: torch.Tensor) -> ForwardOutputs:
        """Run the causal forward pass.

        Parameters
        ----------
        x : (B, T, C) float tensor — normalised EEG input.

        Returns
        -------
        ForwardOutputs
            reconstructed : (B, T, C)
            predicted     : (B, T, C)
            latents       : (B, T, H)
        """
        projected = self.input_norm(self.input_proj(x))   # (B, T, proj_dim)
        hidden, _ = self.gru(projected)                    # (B, T, H)
        reconstructed = self.recon_head(hidden)            # (B, T, C)
        predicted = self.pred_head(hidden)                 # (B, T, C)
        return ForwardOutputs(
            reconstructed=reconstructed,
            predicted=predicted,
            latents=hidden,
        )


# =============================================================================
# § 3  Loss Sub-functions
# =============================================================================

def _compute_reconstruction_losses(
    outputs: ForwardOutputs,
    target_current: torch.Tensor,
    target_future: torch.Tensor,
    mask: torch.Tensor,
    weights: LossWeights,
) -> Dict[str, torch.Tensor]:
    """Compute reconstruction, future-prediction, derivative, and variance losses.

    Parameters
    ----------
    outputs        : ForwardOutputs from the model forward pass.
    target_current : (B, T, C) — EEG signal the model should reconstruct.
    target_future  : (B, T, C) — one-step-ahead EEG targets.
    mask           : (B, T)    — float/bool mask (1 = valid time step).
    weights        : LossWeights controlling lambda values.

    Returns
    -------
    Dict mapping loss-name → scalar tensor.
    """
    mask_f = mask.float()
    n_valid = mask_f.sum().clamp(min=1.0)

    # --- 1a. Reconstruction loss (MSE, masked) --------------------------------
    recon_err = ((outputs.reconstructed - target_current) ** 2).mean(dim=-1)
    recon_loss = (recon_err * mask_f).sum() / n_valid

    # --- 1b. Future-prediction loss (MSE, down-weighted near window edges) ---
    pred_mask = mask_f * weights.future_weight_scale
    pred_err = ((outputs.predicted - target_future) ** 2).mean(dim=-1)
    future_loss = (pred_err * pred_mask).sum() / pred_mask.sum().clamp(min=1.0)

    # --- 1c. Derivative matching loss ----------------------------------------
    # Encourages the reconstructed waveform to have the same local slope as target.
    if target_current.shape[1] > 1:
        recon_diff = outputs.reconstructed[:, 1:, :] - outputs.reconstructed[:, :-1, :]
        target_diff = target_current[:, 1:, :] - target_current[:, :-1, :]
        deriv_err = ((recon_diff - target_diff) ** 2).mean(dim=-1)
        deriv_mask = mask_f[:, 1:]
        derivative_loss = (deriv_err * deriv_mask).sum() / deriv_mask.sum().clamp(min=1.0)
    else:
        derivative_loss = recon_loss.new_zeros(())

    # --- 1d. Variance alignment loss -----------------------------------------
    # Prevents the model from learning a flat mean and ignoring trial variance.
    recon_var = outputs.reconstructed.var(dim=1).mean()
    target_var = target_current.var(dim=1).mean()
    variance_loss = (recon_var - target_var).abs()

    return {
        "recon_loss": recon_loss,
        "future_loss": future_loss,
        "derivative_loss": derivative_loss,
        "variance_loss": variance_loss,
    }


def _compute_cpp_shape_prior_losses(
    outputs: ForwardOutputs,
    target_current: torch.Tensor,
    mask: torch.Tensor,
    times_ms: torch.Tensor,
    weights: LossWeights,
) -> Dict[str, torch.Tensor]:
    """Compute the CPP shape-prior sub-group losses.

    These losses encode soft inductive biases about the CPP waveform shape:
    monotonic build-up, non-zero slope, and late-window amplitude.
    All four terms are scaled by ``weights.lambda_cpp_prior`` in addition to
    their individual lambdas, so setting that to 0.0 disables the whole group.

    The CPP proxy is defined as the mean across the three CPP channels (CP1/CP2/CPz),
    which approximates the classic CPP scoring used in the literature.

    Parameters
    ----------
    outputs        : ForwardOutputs from the model forward pass.
    target_current : (B, T, C) — EEG signal used to compute target CPP proxy.
    mask           : (B, T)    — float/bool mask (1 = valid time step).
    times_ms       : (T,)      — time axis in milliseconds (response-locked).
    weights        : LossWeights carrying shape-prior lambdas and time windows.

    Returns
    -------
    Dict mapping loss-name → scalar tensor (all zero when prior is disabled).
    """
    zero = target_current.new_zeros(())

    if not weights.enable_cpp_shape_prior:
        return {
            "cpp_mean_loss": zero,
            "monotonic_loss": zero,
            "slope_floor_loss": zero,
            "late_amplitude_loss": zero,
            "cpp_mean_alignment_loss": zero,
        }

    # CPP proxy = mean across channels (shape B, T)
    recon_cpp: torch.Tensor = outputs.reconstructed.mean(dim=-1)
    target_cpp: torch.Tensor = target_current.mean(dim=-1)

    # Analysis-window 1-D mask
    analysis_mask_1d = (
        (times_ms >= weights.analysis_window_ms[0])
        & (times_ms <= weights.analysis_window_ms[1])
    )  # (T,)

    # Late-window 1-D mask
    late_mask_1d = (
        (times_ms >= weights.late_window_ms[0])
        & (times_ms <= weights.late_window_ms[1])
    )  # (T,)

    mask_f = mask.float()  # (B, T)
    analysis_mask_2d = analysis_mask_1d.float().unsqueeze(0) * mask_f  # (B, T)

    # --- 3a. CPP mean MSE loss ------------------------------------------------
    cpp_mean_err = (recon_cpp - target_cpp) ** 2
    n_ana = analysis_mask_2d.sum().clamp(min=1.0)
    cpp_mean_loss = (cpp_mean_err * analysis_mask_2d).sum() / n_ana

    # --- 3b. Monotonicity loss ------------------------------------------------
    # Penalise downward steps in the reconstructed CPP within the analysis window.
    if recon_cpp.shape[1] > 1:
        steps = recon_cpp[:, 1:] - recon_cpp[:, :-1]  # (B, T-1)
        ana_step_mask = analysis_mask_2d[:, 1:]
        monotonic_loss = (torch.clamp(-steps, min=0.0) * ana_step_mask).sum() / ana_step_mask.sum().clamp(min=1.0)
    else:
        monotonic_loss = zero

    # --- 3c. Slope floor loss -------------------------------------------------
    # Penalise the recon slope being below a fraction of the target slope.
    if target_cpp.shape[1] > 1:
        recon_slope = recon_cpp[:, 1:] - recon_cpp[:, :-1]
        target_slope = target_cpp[:, 1:] - target_cpp[:, :-1]
        slope_floor = weights.slope_floor_ratio * target_slope
        ana_step_mask = analysis_mask_2d[:, 1:]
        slope_deficit = torch.clamp(slope_floor - recon_slope, min=0.0)
        slope_floor_loss = (slope_deficit * ana_step_mask).sum() / ana_step_mask.sum().clamp(min=1.0)
    else:
        slope_floor_loss = zero

    # --- 3d. Late amplitude loss ----------------------------------------------
    # Penalise under-shooting the late CPP amplitude (last 70 ms before response).
    late_mask_2d = late_mask_1d.float().unsqueeze(0) * mask_f  # (B, T)
    n_late = late_mask_2d.sum().clamp(min=1.0)
    recon_late_mean = (recon_cpp * late_mask_2d).sum(dim=1) / late_mask_2d.sum(dim=1).clamp(min=1.0)
    target_late_mean = (target_cpp * late_mask_2d).sum(dim=1) / late_mask_2d.sum(dim=1).clamp(min=1.0)
    late_amplitude_loss = torch.clamp(target_late_mean - recon_late_mean, min=0.0).mean()

    # --- 3e. CPP mean alignment loss -----------------------------------------
    # Coarser global alignment across the full analysis window.
    recon_ana_mean = (recon_cpp * analysis_mask_2d).sum(dim=1) / analysis_mask_2d.sum(dim=1).clamp(min=1.0)
    target_ana_mean = (target_cpp * analysis_mask_2d).sum(dim=1) / analysis_mask_2d.sum(dim=1).clamp(min=1.0)
    cpp_mean_alignment_loss = ((recon_ana_mean - target_ana_mean) ** 2).mean()

    return {
        "cpp_mean_loss": cpp_mean_loss,
        "monotonic_loss": monotonic_loss,
        "slope_floor_loss": slope_floor_loss,
        "late_amplitude_loss": late_amplitude_loss,
        "cpp_mean_alignment_loss": cpp_mean_alignment_loss,
    }


def _compute_smoothness_loss(outputs: ForwardOutputs) -> torch.Tensor:
    """Penalise large frame-to-frame jumps in the GRU latent state.

    A smooth latent trajectory is a weak prior that encourages the model to
    learn slowly-varying dynamics rather than frame-by-frame noise fitting.

    Parameters
    ----------
    outputs : ForwardOutputs — latents shape (B, T, H).

    Returns
    -------
    Scalar tensor (0.0 when T <= 1).
    """
    if outputs.latents.shape[1] <= 1:
        return outputs.latents.new_zeros(())
    delta = outputs.latents[:, 1:, :] - outputs.latents[:, :-1, :]  # (B, T-1, H)
    return (delta ** 2).mean()


# =============================================================================
# § 4  Composite Loss
# =============================================================================

def masked_self_supervised_loss(
    outputs: ForwardOutputs,
    target_current: torch.Tensor,
    target_future: torch.Tensor,
    mask: torch.Tensor,
    times_ms: torch.Tensor,
    weights: LossWeights,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the full composite self-supervised loss and return a metrics dict.

    This function combines four loss groups into a single differentiable scalar:
      1. Reconstruction & prediction  (recon, future, derivative, variance)
      2. CPP proxy alignment          (cpp_mean)
      3. CPP shape prior              (monotonic, slope_floor, late_amplitude,
                                       cpp_mean_alignment) — can be disabled
      4. Latent smoothness            (smooth)

    Parameters
    ----------
    outputs        : ForwardOutputs produced by CPPForwardGRU.forward().
    target_current : (B, T, C) — EEG signal to reconstruct.
    target_future  : (B, T, C) — one-step-ahead EEG prediction targets.
    mask           : (B, T) bool/float — 1 at valid (non-horizon) time steps.
    times_ms       : (T,) — response-locked time axis in milliseconds.
    weights        : LossWeights carrying all lambda values and toggle flags.

    Returns
    -------
    total_loss : scalar tensor (differentiable).
    metrics    : Dict[str, float] with individual loss values for logging.
                 Always contains ``"total_loss"``.  Shape-prior terms are
                 present but zero when ``weights.enable_cpp_shape_prior`` is
                 ``False``.
    """
    # --- Group 1: Reconstruction & prediction --------------------------------
    recon_losses = _compute_reconstruction_losses(
        outputs, target_current, target_future, mask, weights
    )

    # --- Group 2: CPP proxy mean loss ----------------------------------------
    # (Simpler than shape prior; always active.)
    cpp_proxy = outputs.reconstructed.mean(dim=-1)          # (B, T)
    target_proxy = target_current.mean(dim=-1)               # (B, T)
    mask_f = mask.float()
    n_valid = mask_f.sum().clamp(min=1.0)
    cpp_mean_loss = (((cpp_proxy - target_proxy) ** 2) * mask_f).sum() / n_valid

    # --- Group 3: CPP shape prior --------------------------------------------
    prior_losses = _compute_cpp_shape_prior_losses(
        outputs, target_current, mask, times_ms, weights
    )

    # --- Group 4: Latent smoothness ------------------------------------------
    smooth_loss = _compute_smoothness_loss(outputs)

    # --- Weighted sum --------------------------------------------------------
    total_loss = (
        weights.lambda_recon       * recon_losses["recon_loss"]
        + weights.lambda_future    * recon_losses["future_loss"]
        + weights.lambda_derivative * recon_losses["derivative_loss"]
        + weights.lambda_variance  * recon_losses["variance_loss"]
        + weights.lambda_cpp_mean  * cpp_mean_loss
        + weights.lambda_cpp_prior * (
            weights.lambda_monotonic          * prior_losses["monotonic_loss"]
            + weights.lambda_slope_floor      * prior_losses["slope_floor_loss"]
            + weights.lambda_late_amplitude   * prior_losses["late_amplitude_loss"]
            + weights.lambda_cpp_mean_alignment * prior_losses["cpp_mean_alignment_loss"]
        )
        + weights.lambda_smooth    * smooth_loss
    )

    metrics: Dict[str, float] = {
        "total_loss":              total_loss.item(),
        "recon_loss":              recon_losses["recon_loss"].item(),
        "future_loss":             recon_losses["future_loss"].item(),
        "derivative_loss":         recon_losses["derivative_loss"].item(),
        "variance_loss":           recon_losses["variance_loss"].item(),
        "cpp_mean_loss":           cpp_mean_loss.item(),
        "monotonic_loss":          prior_losses["monotonic_loss"].item(),
        "slope_floor_loss":        prior_losses["slope_floor_loss"].item(),
        "late_amplitude_loss":     prior_losses["late_amplitude_loss"].item(),
        "cpp_mean_alignment_loss": prior_losses["cpp_mean_alignment_loss"].item(),
        "smooth_loss":             smooth_loss.item(),
    }
    return total_loss, metrics
