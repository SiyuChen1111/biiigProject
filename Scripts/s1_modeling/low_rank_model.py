from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .config import LossWeights
from .model import ForwardOutputs, masked_self_supervised_loss


@dataclass(frozen=True)
class LowRankRNNConfig:
    """Small low-rank recurrent model configuration for exploratory smoke tests."""

    rank: int = 3
    population_dim: int = 64
    input_scale: float = 1.0
    recurrent_scale: float = 1.0
    state_leak: float = 0.25


class CPPLowRankRNN(nn.Module):
    """Lightweight low-rank recurrent baseline for response-locked CPP EEG.

    The recurrent state exposed for analysis is rank-dimensional. A larger
    nonlinear population expansion is used internally, but the next state and
    EEG reconstruction both pass through the low-rank state.
    """

    def __init__(self, n_channels: int, config: LowRankRNNConfig) -> None:
        super().__init__()
        if config.rank < 1:
            raise ValueError("rank must be >= 1")
        if config.population_dim < config.rank:
            raise ValueError("population_dim must be >= rank")
        if not 0.0 < config.state_leak <= 1.0:
            raise ValueError("state_leak must be in (0, 1]")

        self.n_channels = n_channels
        self.cfg = config

        self.input_to_population = nn.Linear(n_channels, config.population_dim)
        self.m_factor = nn.Parameter(torch.randn(config.population_dim, config.rank) * 0.15)
        self.n_factor = nn.Parameter(torch.randn(config.population_dim, config.rank) * 0.15)
        self.state_bias = nn.Parameter(torch.zeros(config.rank))

        self.recon_head = nn.Sequential(
            nn.LayerNorm(config.rank),
            nn.Linear(config.rank, config.rank),
            nn.Tanh(),
            nn.Linear(config.rank, n_channels),
        )
        self.pred_head = nn.Sequential(
            nn.LayerNorm(config.rank),
            nn.Linear(config.rank, config.rank),
            nn.Tanh(),
            nn.Linear(config.rank, n_channels),
        )

    def forward(self, x: torch.Tensor) -> ForwardOutputs:
        """Run a causal low-rank recurrent pass over ``x``.

        Parameters
        ----------
        x : (B, T, C) tensor

        Returns
        -------
        ForwardOutputs with rank-dimensional ``latents``.
        """
        batch_size, n_time, _ = x.shape
        z = x.new_zeros(batch_size, self.cfg.rank)
        states = []
        scale = self.cfg.recurrent_scale / (self.cfg.population_dim ** 0.5)

        for t in range(n_time):
            population_drive = (
                self.cfg.input_scale * self.input_to_population(x[:, t, :])
                + self.cfg.recurrent_scale * (z @ self.m_factor.T)
            )
            population = torch.tanh(population_drive)
            proposed_z = scale * (population @ self.n_factor) + self.state_bias
            z = (1.0 - self.cfg.state_leak) * z + self.cfg.state_leak * proposed_z
            states.append(z)

        latents = torch.stack(states, dim=1)
        reconstructed = self.recon_head(latents)
        predicted = self.pred_head(latents)
        return ForwardOutputs(
            reconstructed=reconstructed,
            predicted=predicted,
            latents=latents,
        )


def low_rank_self_supervised_loss(
    outputs: ForwardOutputs,
    target_current: torch.Tensor,
    target_future: torch.Tensor,
    mask: torch.Tensor,
    times_ms: torch.Tensor,
    weights: LossWeights,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Use the project composite CPP loss for the low-rank smoke model."""
    return masked_self_supervised_loss(
        outputs=outputs,
        target_current=target_current,
        target_future=target_future,
        mask=mask,
        times_ms=times_ms,
        weights=weights,
    )
