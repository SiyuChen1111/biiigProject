from __future__ import annotations

import json
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modeling.config import TrainingConfig
from modeling.dataset import load_stage2_dataset, make_dataloaders
from modeling.model import CPPForwardGRU, ForwardOutputs, masked_self_supervised_loss
from modeling.utils import set_global_seed


# =============================================================================
# § 1  Training Utilities
# =============================================================================

def _run_epoch(
    model: CPPForwardGRU,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    config: TrainingConfig,
    device: torch.device,
    *,
    train: bool,
) -> Dict[str, float]:
    """Run one full epoch (training or evaluation) and return averaged metrics.

    Parameters
    ----------
    model     : CPPForwardGRU instance.
    loader    : DataLoader yielding batches from EEGWindowDataset.
    optimizer : Adam optimiser (None when ``train=False``).
    config    : TrainingConfig carrying gradient clip and loss weights.
    device    : Torch device to run on.
    train     : If True, compute gradients and update parameters.

    Returns
    -------
    Dict[str, float] with averaged loss metrics for the epoch.
    """
    model.train(train)
    accum: Dict[str, float] = {}
    n_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            x         = batch["eeg"].to(device)
            x_future  = batch["target_future"].to(device)
            mask      = batch["mask"].to(device)
            times_ms  = batch["times_ms"][0].to(device)  # shared across batch

            outputs: ForwardOutputs = model(x)
            loss, metrics = masked_self_supervised_loss(
                outputs, x, x_future, mask, times_ms, config.loss
            )

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()

            for k, v in metrics.items():
                accum[k] = accum.get(k, 0.0) + v
            n_batches += 1

    if n_batches == 0:
        return accum
    return {k: v / n_batches for k, v in accum.items()}


# =============================================================================
# § 2  Visualisation Helpers
# =============================================================================

def _save_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    output_dir: Path,
) -> None:
    """Save a training / validation loss curve plot to *output_dir*.

    Parameters
    ----------
    train_losses : Per-epoch training total losses.
    val_losses   : Per-epoch validation total losses.
    output_dir   : Directory to write ``loss_curve.png`` into.
    """
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(train_losses, label="train")
        ax.plot(val_losses,   label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total loss")
        ax.set_title("Training curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "loss_curve.png", dpi=120)
        plt.close(fig)
    except Exception:
        pass  # Non-critical; skip silently if matplotlib is unavailable.


def _save_reconstruction_examples(
    model: CPPForwardGRU,
    loader: DataLoader,
    output_dir: Path,
    device: torch.device,
    n_examples: int = 8,
) -> None:
    """Save a grid of per-trial reconstruction vs. input waveforms.

    Parameters
    ----------
    model      : Trained CPPForwardGRU.
    loader     : DataLoader to sample trials from (first batch used).
    output_dir : Directory to write ``reconstruction_examples.png`` into.
    device     : Torch device.
    n_examples : Number of trials to display.
    """
    try:
        import matplotlib.pyplot as plt
        model.eval()
        batch = next(iter(loader))
        x = batch["eeg"][:n_examples].to(device)
        times_ms = batch["times_ms"][0].cpu().numpy()
        with torch.no_grad():
            out = model(x)
        x_np   = x.cpu().numpy()
        rec_np = out.reconstructed.cpu().numpy()

        fig, axes = plt.subplots(n_examples, 1, figsize=(10, 2 * n_examples), sharex=True)
        for i, ax in enumerate(axes):
            ax.plot(times_ms, x_np[i, :, :].mean(axis=-1), label="input",  lw=1.2)
            ax.plot(times_ms, rec_np[i, :, :].mean(axis=-1), label="recon", lw=1.2, ls="--")
            if i == 0:
                ax.legend(fontsize=8)
        axes[-1].set_xlabel("Time (ms)")
        fig.suptitle("Reconstruction examples (CPP proxy)")
        fig.tight_layout()
        fig.savefig(output_dir / "reconstruction_examples.png", dpi=120)
        plt.close(fig)
    except Exception:
        pass


def _save_cpp_average_plot(
    model: CPPForwardGRU,
    loader: DataLoader,
    output_dir: Path,
    device: torch.device,
) -> None:
    """Compute and save the grand-average CPP waveform comparison.

    Saves both a PNG figure and a ``.npz`` file (``cpp_average_sanity.npz``)
    with keys ``recon_mean``, ``target_mean``, and ``times_ms``.

    .. note::
        ``_save_cpp_comparison_overlay`` reads the ``.npz`` written here, so
        this function **must** run before that one.

    Parameters
    ----------
    model      : Trained CPPForwardGRU.
    loader     : DataLoader (full split, no shuffle preferred).
    output_dir : Directory for output files.
    device     : Torch device.
    """
    try:
        import matplotlib.pyplot as plt
        model.eval()
        all_recon, all_input, times_np = [], [], None
        with torch.no_grad():
            for batch in loader:
                x = batch["eeg"].to(device)
                times_ms = batch["times_ms"][0].cpu().numpy()
                out = model(x)
                all_recon.append(out.reconstructed.cpu().numpy().mean(axis=-1))
                all_input.append(x.cpu().numpy().mean(axis=-1))
                times_np = times_ms

        recon_mean  = np.concatenate(all_recon,  axis=0).mean(axis=0)
        target_mean = np.concatenate(all_input, axis=0).mean(axis=0)

        # Persist for downstream overlay plot.
        np.savez(output_dir / "cpp_average_sanity.npz",
                 recon_mean=recon_mean, target_mean=target_mean, times_ms=times_np)

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(times_np, target_mean, label="target CPP", lw=1.5)
        ax.plot(times_np, recon_mean,  label="recon CPP",  lw=1.5, ls="--")
        ax.axvline(0, color="k", lw=0.8, ls=":")
        ax.set_xlabel("Time from response (ms)")
        ax.legend()
        ax.set_title("Grand-average CPP: target vs reconstruction")
        fig.tight_layout()
        fig.savefig(output_dir / "cpp_average_comparison.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass


def _save_cpp_comparison_overlay(output_dir: Path) -> None:
    """Overlay derivative comparison on the grand-average CPP figure.

    Prerequisite: ``_save_cpp_average_plot`` must have written
    ``cpp_average_sanity.npz`` to *output_dir*.

    Parameters
    ----------
    output_dir : Directory containing ``cpp_average_sanity.npz``.
    """
    try:
        import matplotlib.pyplot as plt
        npz_path = output_dir / "cpp_average_sanity.npz"
        if not npz_path.exists():
            return
        data = np.load(npz_path)
        recon_mean  = data["recon_mean"]
        target_mean = data["target_mean"]
        times_np    = data["times_ms"]

        recon_deriv  = np.gradient(recon_mean,  times_np)
        target_deriv = np.gradient(target_mean, times_np)

        fig, axes = plt.subplots(1, 2, figsize=(12, 3))
        axes[0].plot(times_np, target_mean, label="target"); axes[0].plot(times_np, recon_mean, ls="--", label="recon")
        axes[0].set_title("CPP amplitude"); axes[0].legend()
        axes[1].plot(times_np, target_deriv, label="target slope"); axes[1].plot(times_np, recon_deriv, ls="--", label="recon slope")
        axes[1].set_title("CPP slope (d/dt)"); axes[1].legend()
        for ax in axes:
            ax.axvline(0, color="k", lw=0.8, ls=":")
            ax.set_xlabel("Time (ms)")
        fig.tight_layout()
        fig.savefig(output_dir / "cpp_derivative_comparison.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass


# =============================================================================
# § 3  Main Training Loop
# =============================================================================

def train_model(
    dataset_dir: Path,
    output_dir: Path,
    config: TrainingConfig,
) -> Dict[str, object]:
    """Train CPPForwardGRU with early stopping and save the best checkpoint.

    Parameters
    ----------
    dataset_dir : Directory containing the processed EEG dataset
                  (output of the preprocessing pipeline).
    output_dir  : Directory to write the checkpoint, loss curve, and
                  diagnostic figures into.
    config      : Full TrainingConfig (architecture + training + loss weights).

    Returns
    -------
    Dict with keys:
      ``"best_val_loss"``    : float — best validation total loss achieved.
      ``"checkpoint_path"``  : Path  — absolute path to ``best_model.pt``.
      ``"n_epochs_trained"`` : int   — total epochs run (including patience).
      ``"train_losses"``     : list[float].
      ``"val_losses"``       : list[float].
    """
    set_global_seed(config.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data ----------------------------------------------------------------
    eeg, targets, mask, times_ms, metadata = load_stage2_dataset(dataset_dir, config)
    train_loader, val_loader, test_loader, split_indices = make_dataloaders(
        eeg, targets, mask, times_ms, config
    )
    n_channels = eeg.shape[-1]

    # --- Model & optimiser ---------------------------------------------------
    model = CPPForwardGRU(n_channels=n_channels, model_config=config.model).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # --- Training loop -------------------------------------------------------
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses: list[float] = []
    val_losses:   list[float] = []
    checkpoint_path = output_dir / "best_model.pt"

    for epoch in range(config.max_epochs):
        train_metrics = _run_epoch(model, train_loader, optimizer, config, device, train=True)
        val_metrics   = _run_epoch(model, val_loader,   None,      config, device, train=False)

        train_losses.append(train_metrics["total_loss"])
        val_losses.append(val_metrics["total_loss"])

        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "split_indices": split_indices,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                break

    # --- Diagnostic figures --------------------------------------------------
    _save_loss_curves(train_losses, val_losses, output_dir)
    # Reload best weights for figure generation.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _save_reconstruction_examples(model, val_loader, output_dir, device)
    _save_cpp_average_plot(model, val_loader, output_dir, device)
    _save_cpp_comparison_overlay(output_dir)

    return {
        "best_val_loss":   best_val_loss,
        "checkpoint_path": checkpoint_path,
        "n_epochs_trained": len(train_losses),
        "train_losses": train_losses,
        "val_losses":   val_losses,
    }


# =============================================================================
# § 4  Latent Export
# =============================================================================

def export_full_latents_from_checkpoint(
    checkpoint_path: Path,
    dataset_dir: Path,
    output_dir: Path,
) -> Path:
    """Export full-dataset GRU hidden states from a saved checkpoint.

    Runs the model in eval mode over *all* trials (train + val + test) and
    saves the resulting latent tensor to a ``.npz`` file.

    Parameters
    ----------
    checkpoint_path : Path to ``best_model.pt`` (written by :func:`train_model`).
    dataset_dir     : Dataset directory (same as used during training).
    output_dir      : Directory to write ``latents_full.npz`` into.

    Returns
    -------
    Path to the saved ``latents_full.npz`` file.

    Output ``.npz`` keys
    --------------------
    ``"latents"``  : (N, T, H) float32 — hidden states for all N trials.
    ``"times_ms"`` : (T,)      float32 — shared response-locked time axis.
    """
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load checkpoint & config --------------------------------------------
    ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config: TrainingConfig = ckpt["config"]

    # --- Reload dataset (all trials, no split filtering) --------------------
    eeg, targets, mask, times_ms, metadata = load_stage2_dataset(dataset_dir, config)
    n_channels = eeg.shape[-1]

    # --- Rebuild model & load weights ----------------------------------------
    model = CPPForwardGRU(n_channels=n_channels, model_config=config.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # --- Inference in mini-batches (avoids OOM on large datasets) ------------
    batch_size = 256
    all_latents: list[np.ndarray] = []
    eeg_tensor = torch.as_tensor(eeg, dtype=torch.float32)

    with torch.no_grad():
        for start in range(0, len(eeg), batch_size):
            x_batch = eeg_tensor[start : start + batch_size].to(device)
            out     = model(x_batch)
            all_latents.append(out.latents.cpu().numpy())

    latents_full = np.concatenate(all_latents, axis=0)  # (N, T, H)
    out_path = output_dir / "latents_full.npz"
    np.savez(out_path, latents=latents_full, times_ms=times_ms)

    return out_path
