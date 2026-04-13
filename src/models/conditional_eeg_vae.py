from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.EEGModels_PyTorch import EEGNet


class ConditionalEEGVAE(nn.Module):
    def __init__(
        self,
        chans: int,
        samples: int,
        num_classes: int = 2,
        latent_dim: int = 32,
        dropout_rate: float = 0.5,
        kern_length: int = 32,
        f1: int = 8,
        d: int = 2,
        f2: int = 16,
    ) -> None:
        super().__init__()
        self.chans = chans
        self.samples = samples
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        self.encoder_backbone = EEGNet(
            nb_classes=num_classes,
            Chans=chans,
            Samples=samples,
            dropoutRate=dropout_rate,
            kernLength=kern_length,
            F1=f1,
            D=d,
            F2=f2,
            dropoutType="Dropout",
        )
        feature_dim = self.encoder_backbone.feature_dim
        self.fc_mu = nn.Linear(feature_dim + num_classes, latent_dim)
        self.fc_logvar = nn.Linear(feature_dim + num_classes, latent_dim)
        self.label_head = nn.Linear(latent_dim, num_classes)
        self.decoder_input = nn.Linear(latent_dim + num_classes, 64 * (samples // 4))
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv1d(16, chans, kernel_size=7, padding=3),
        )

    def _prepare_condition(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.dim() == 1:
            return F.one_hot(labels, num_classes=self.num_classes).float()
        return labels.float()

    def encode(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder_backbone.forward_features(x)
        cond = self._prepare_condition(labels).to(features.device)
        joint = torch.cat([features, cond], dim=1)
        mu = self.fc_mu(joint)
        logvar = self.fc_logvar(joint)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cond = self._prepare_condition(labels).to(z.device)
        joint = torch.cat([z, cond], dim=1)
        recon = self.decoder_input(joint)
        recon = recon.view(-1, 64, self.samples // 4)
        recon = self.decoder(recon)
        recon = F.interpolate(recon, size=self.samples, mode="linear", align_corners=False)
        return recon.unsqueeze(-1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, labels)
        label_logits = self.label_head(mu)
        return recon, mu, logvar, label_logits

    @staticmethod
    def loss_function(
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        label_logits: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        beta: float = 1.0,
        feature_consistency_loss: torch.Tensor | None = None,
        smoothness_weight: float = 0.0,
        feature_weight: float = 0.0,
        roi_recon_loss: torch.Tensor | None = None,
        roi_weight: float = 0.0,
        label_weight: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        smoothness_loss = torch.mean((recon_x[..., 1:, :] - recon_x[..., :-1, :]) ** 2)
        feature_loss = feature_consistency_loss if feature_consistency_loss is not None else torch.tensor(0.0, device=recon_x.device)
        roi_loss = roi_recon_loss if roi_recon_loss is not None else torch.tensor(0.0, device=recon_x.device)
        if label_logits is not None and labels is not None:
            label_loss = F.cross_entropy(label_logits, labels)
        else:
            label_loss = torch.tensor(0.0, device=recon_x.device)
        loss = (
            recon_loss
            + beta * kl
            + smoothness_weight * smoothness_loss
            + feature_weight * feature_loss
            + roi_weight * roi_loss
            + label_weight * label_loss
        )
        return loss, {
            "recon_loss": float(recon_loss.detach().cpu().item()),
            "kl_loss": float(kl.detach().cpu().item()),
            "smoothness_loss": float(smoothness_loss.detach().cpu().item()),
            "feature_loss": float(feature_loss.detach().cpu().item()),
            "roi_recon_loss": float(roi_loss.detach().cpu().item()),
            "label_loss": float(label_loss.detach().cpu().item()),
            "total_loss": float(loss.detach().cpu().item()),
        }

    def sample(self, labels: torch.Tensor, n_samples: int | None = None, device: torch.device | None = None) -> torch.Tensor:
        cond = self._prepare_condition(labels)
        if n_samples is not None and cond.shape[0] != n_samples:
            raise ValueError("Condition batch size does not match requested sample count.")
        current_device = device or next(self.parameters()).device
        z = torch.randn(cond.shape[0], self.latent_dim, device=current_device)
        cond = cond.to(current_device)
        return self.decode(z, cond)
