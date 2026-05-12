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
    lambda_recon: float,
    lambda_smooth: float,
) -> tuple[torch.Tensor, dict]:
    """Combine reconstruction, prediction, and temporal smoothness losses."""
    mask = mask.float()
    denom = mask.sum().clamp_min(1.0)
    recon_error = ((outputs.reconstruction - target_current) ** 2).mean(dim=-1)
    future_error = ((outputs.future_prediction - target_future) ** 2).mean(dim=(-1, -2))
    recon_loss = (recon_error * mask).sum() / denom
    future_loss = (future_error * mask).sum() / denom
    smooth_delta = outputs.latents[:, 1:, :] - outputs.latents[:, :-1, :]
    smooth_loss = (smooth_delta ** 2).mean()
    total_loss = future_loss + lambda_recon * recon_loss + lambda_smooth * smooth_loss
    metrics = {
        "total_loss": float(total_loss.detach().cpu()),
        "future_loss": float(future_loss.detach().cpu()),
        "recon_loss": float(recon_loss.detach().cpu()),
        "smooth_loss": float(smooth_loss.detach().cpu()),
    }
    return total_loss, metrics
