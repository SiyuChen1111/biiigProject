from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import TrainingConfig


@dataclass
class ForwardOutputs:
    latents: torch.Tensor
    reconstruction: torch.Tensor
    future_prediction: torch.Tensor


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CPPForwardGRU(nn.Module):
    """Causal CPP latent-dynamics model.

    The encoder is strictly forward (causal) so each latent state only uses
    information available up to the current time point.
    """

    def __init__(self, config: TrainingConfig, input_channels: int = 3) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(input_channels, config.projection_dim)
        self.layer_norm = nn.LayerNorm(config.projection_dim)
        self.encoder = nn.GRU(
            input_size=config.projection_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.reconstruction_head = MLPHead(config.hidden_dim, config.hidden_dim, input_channels)
        self.future_hidden_dim = 64
        self.future_head = MLPHead(config.hidden_dim, self.future_hidden_dim, input_channels)
        self.input_channels = input_channels
        self.horizon_steps = 1

    def set_horizon(self, horizon_steps: int) -> None:
        self.horizon_steps = int(horizon_steps)
        self.future_head = MLPHead(
            self.config.hidden_dim,
            self.future_hidden_dim,
            self.horizon_steps * self.input_channels,
        )

    def forward(self, eeg: torch.Tensor) -> ForwardOutputs:
        projected = self.layer_norm(self.input_projection(eeg))
        latents, _ = self.encoder(projected)
        reconstruction = self.reconstruction_head(latents)
        future = self.future_head(latents).view(
            eeg.shape[0], eeg.shape[1], self.horizon_steps, self.input_channels
        )
        return ForwardOutputs(latents=latents, reconstruction=reconstruction, future_prediction=future)


def masked_self_supervised_loss(
    outputs: ForwardOutputs,
    target_current: torch.Tensor,
    target_future: torch.Tensor,
    mask: torch.Tensor,
    future_weight_scale: float,
    lambda_future: float,
    lambda_recon: float,
    lambda_derivative: float,
    lambda_variance: float,
    lambda_cpp_mean: float,
    lambda_cpp_prior: float,
    lambda_monotonic: float,
    lambda_slope_floor: float,
    lambda_late_amplitude: float,
    lambda_cpp_mean_alignment: float,
    enable_cpp_shape_prior: bool,
    analysis_window: tuple[float, float],
    late_window: tuple[float, float],
    slope_floor_ratio: float,
    times_ms: torch.Tensor,
    lambda_smooth: float,
) -> tuple[torch.Tensor, dict]:
    """Combine reconstruction, prediction, and temporal smoothness losses."""
    mask = mask.float()
    denom = mask.sum().clamp_min(1.0)
    recon_error = ((outputs.reconstruction - target_current) ** 2).mean(dim=-1)
    future_error = ((outputs.future_prediction - target_future) ** 2).mean(dim=(-1, -2))
    recon_loss = (recon_error * mask).sum() / denom
    future_loss = (future_error * (mask * future_weight_scale)).sum() / (mask * future_weight_scale).sum().clamp_min(1.0)
    recon_delta = outputs.reconstruction[:, 1:, :] - outputs.reconstruction[:, :-1, :]
    target_delta = target_current[:, 1:, :] - target_current[:, :-1, :]
    derivative_error = ((recon_delta - target_delta) ** 2).mean(dim=-1)
    derivative_mask = mask[:, 1:] * 0.5 + mask[:, :-1] * 0.5
    derivative_loss = (derivative_error * derivative_mask).sum() / derivative_mask.sum().clamp_min(1.0)
    recon_centered = outputs.reconstruction - outputs.reconstruction.mean(dim=-1, keepdim=True)
    target_centered = target_current - target_current.mean(dim=-1, keepdim=True)
    variance_loss = torch.abs(recon_centered.std(dim=-1) - target_centered.std(dim=-1))
    variance_loss = (variance_loss * mask).sum() / denom
    recon_cpp = outputs.reconstruction.mean(dim=-1)
    target_cpp = target_current.mean(dim=-1)
    cpp_mean_loss = ((recon_cpp - target_cpp) ** 2 * mask).sum() / denom

    times = times_ms.to(target_current.device)
    analysis_mask_1d = ((times >= analysis_window[0]) & (times <= analysis_window[1])).float()
    late_mask_1d = ((times >= late_window[0]) & (times <= late_window[1])).float()
    analysis_mask = analysis_mask_1d[None, :]
    late_mask = late_mask_1d[None, :]
    recon_cpp_delta = recon_cpp[:, 1:] - recon_cpp[:, :-1]
    target_cpp_delta = target_cpp[:, 1:] - target_cpp[:, :-1]
    analysis_delta_mask = analysis_mask[:, 1:] * 0.5 + analysis_mask[:, :-1] * 0.5
    negative_delta = torch.relu(-recon_cpp_delta)
    monotonic_loss = (negative_delta * analysis_delta_mask).sum() / analysis_delta_mask.sum().clamp_min(1.0)

    target_slope = (target_cpp_delta * analysis_delta_mask).sum(dim=1) / analysis_delta_mask.sum(dim=1).clamp_min(1.0)
    recon_slope = (recon_cpp_delta * analysis_delta_mask).sum(dim=1) / analysis_delta_mask.sum(dim=1).clamp_min(1.0)
    slope_floor = slope_floor_ratio * target_slope
    slope_floor_loss = torch.relu(slope_floor - recon_slope).mean()

    target_late_amplitude = (target_cpp * late_mask).sum(dim=1) / late_mask.sum(dim=1).clamp_min(1.0)
    recon_late_amplitude = (recon_cpp * late_mask).sum(dim=1) / late_mask.sum(dim=1).clamp_min(1.0)
    late_amplitude_loss = torch.relu(target_late_amplitude - recon_late_amplitude).mean()

    cpp_mean_alignment_loss = ((recon_cpp - target_cpp) ** 2 * analysis_mask).sum() / analysis_mask.sum().clamp_min(1.0)
    cpp_prior_total_loss = (
        lambda_monotonic * monotonic_loss
        + lambda_slope_floor * slope_floor_loss
        + lambda_late_amplitude * late_amplitude_loss
        + lambda_cpp_mean_alignment * cpp_mean_alignment_loss
    )
    if not enable_cpp_shape_prior:
        cpp_prior_total_loss = torch.zeros((), device=target_current.device)
        monotonic_loss = torch.zeros((), device=target_current.device)
        slope_floor_loss = torch.zeros((), device=target_current.device)
        late_amplitude_loss = torch.zeros((), device=target_current.device)
        cpp_mean_alignment_loss = torch.zeros((), device=target_current.device)

    smooth_delta = outputs.latents[:, 1:, :] - outputs.latents[:, :-1, :]
    smooth_loss = (smooth_delta ** 2).mean()
    total_loss = (
        lambda_future * future_loss
        + lambda_recon * recon_loss
        + lambda_derivative * derivative_loss
        + lambda_variance * variance_loss
        + lambda_cpp_mean * cpp_mean_loss
        + lambda_cpp_prior * cpp_prior_total_loss
        + lambda_smooth * smooth_loss
    )
    metrics = {
        "total_loss": float(total_loss.detach().cpu()),
        "future_loss": float(future_loss.detach().cpu()),
        "recon_loss": float(recon_loss.detach().cpu()),
        "derivative_loss": float(derivative_loss.detach().cpu()),
        "variance_loss": float(variance_loss.detach().cpu()),
        "cpp_mean_loss": float(cpp_mean_loss.detach().cpu()),
        "monotonic_loss": float(monotonic_loss.detach().cpu()),
        "slope_floor_loss": float(slope_floor_loss.detach().cpu()),
        "late_amplitude_loss": float(late_amplitude_loss.detach().cpu()),
        "cpp_mean_alignment_loss": float(cpp_mean_alignment_loss.detach().cpu()),
        "cpp_prior_total_loss": float(cpp_prior_total_loss.detach().cpu()),
        "smooth_loss": float(smooth_loss.detach().cpu()),
    }
    return total_loss, metrics
