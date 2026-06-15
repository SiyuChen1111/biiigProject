from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from modeling.config import LossWeights, TrainingConfig
from modeling.dataset import load_stage2_dataset, make_dataloaders
from modeling.model import CPPForwardGRU, ForwardOutputs, masked_self_supervised_loss
from modeling.utils import set_global_seed


# =============================================================================
# § 1  Internal Evaluation Helpers
# =============================================================================

def _eval_loss(
    model: CPPForwardGRU,
    loader,
    config: TrainingConfig,
    device,
) -> Dict[str, float]:
    """Return average loss metrics over one full pass of *loader* (no grad).

    Parameters
    ----------
    model  : CPPForwardGRU in eval mode.
    loader : DataLoader to iterate over.
    config : TrainingConfig carrying loss weights.
    device : Torch device.

    Returns
    -------
    Dict[str, float] with averaged metric values.
    """
    import torch
    model.eval()
    accum: Dict[str, float] = {}
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            x        = batch["eeg"].to(device)
            x_future = batch["target_future"].to(device)
            mask     = batch["mask"].to(device)
            times_ms = batch["times_ms"][0].to(device)
            out: ForwardOutputs = model(x)
            _, metrics = masked_self_supervised_loss(
                out, x, x_future, mask, times_ms, config.loss
            )
            for k, v in metrics.items():
                accum[k] = accum.get(k, 0.0) + v
            n_batches += 1
    if n_batches == 0:
        return accum
    return {k: v / n_batches for k, v in accum.items()}


def _waveform_metrics(
    model: CPPForwardGRU,
    loader,
    device,
) -> Dict[str, float]:
    """Compute grand-average CPP waveform quality metrics.

    Metrics returned
    ----------------
    ``cpp_corr``  : Pearson r between grand-average reconstructed and target CPP.
    ``cpp_rmse``  : RMSE between grand-average waveforms.

    Parameters
    ----------
    model  : CPPForwardGRU in eval mode.
    loader : DataLoader (validation or test split).
    device : Torch device.

    Returns
    -------
    Dict[str, float].
    """
    import torch
    model.eval()
    all_recon, all_input = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["eeg"].to(device)
            out = model(x)
            all_recon.append(out.reconstructed.cpu().numpy().mean(axis=-1))
            all_input.append(x.cpu().numpy().mean(axis=-1))
    recon_mean  = np.concatenate(all_recon,  axis=0).mean(axis=0)
    target_mean = np.concatenate(all_input, axis=0).mean(axis=0)
    corr = float(np.corrcoef(recon_mean, target_mean)[0, 1])
    rmse = float(np.sqrt(np.mean((recon_mean - target_mean) ** 2)))
    return {"cpp_corr": corr, "cpp_rmse": rmse}


# =============================================================================
# § 2  Quick-Train Helper
# =============================================================================

def _quick_train(
    dataset_dir: Path,
    config: TrainingConfig,
    n_epochs: int = 20,
) -> CPPForwardGRU:
    """Train a CPPForwardGRU for *n_epochs* and return the model.

    Used inside sweep loops to avoid importing the full train.py workflow.

    Parameters
    ----------
    dataset_dir : Processed dataset directory.
    config      : TrainingConfig (including loss weights to sweep over).
    n_epochs    : Number of training epochs (kept low for sweep speed).

    Returns
    -------
    Trained CPPForwardGRU on CPU.
    """
    import torch
    import torch.nn as nn

    set_global_seed(config.seed)
    device = torch.device("cpu")  # Sweeps run on CPU for portability.

    eeg, targets, mask, times_ms, _ = load_stage2_dataset(dataset_dir, config)
    train_loader, val_loader, _, _ = make_dataloaders(eeg, targets, mask, times_ms, config)
    n_channels = eeg.shape[-1]

    model     = CPPForwardGRU(n_channels=n_channels, model_config=config.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)

    for _ in range(n_epochs):
        model.train()
        for batch in train_loader:
            x        = batch["eeg"].to(device)
            x_future = batch["target_future"].to(device)
            mask_b   = batch["mask"].to(device)
            times_b  = batch["times_ms"][0].to(device)
            out      = model(x)
            loss, _  = masked_self_supervised_loss(out, x, x_future, mask_b, times_b, config.loss)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()

    return model


# =============================================================================
# § 3  CPP Shape-Prior Sweep
# =============================================================================

def run_small_cpp_prior_sweep(
    dataset_dir: Path,
    output_dir: Path,
    config: TrainingConfig,
    lambda_cpp_prior_grid: Sequence[float] = (0.0, 0.05, 0.1, 0.5, 1.0),
    n_epochs: int = 20,
) -> Dict[str, Any]:
    """Sweep over ``lambda_cpp_prior`` values and record CPP waveform quality.

    For each candidate value, a fresh model is trained for *n_epochs* and
    evaluated on the validation split.  Results are written to
    ``sweep_results.csv`` in *output_dir*.

    Parameters
    ----------
    dataset_dir            : Processed dataset directory.
    output_dir             : Directory to write ``sweep_results.csv`` into.
    config                 : Base TrainingConfig (lambda_cpp_prior overridden).
    lambda_cpp_prior_grid  : Sequence of lambda values to evaluate.
    n_epochs               : Training epochs per candidate (default: 20).

    Returns
    -------
    Dict with key ``"score"`` — the ``cpp_corr`` of the best candidate —
    and ``"results"`` — a list of per-candidate result dicts.
    """
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device    = torch.device("cpu")
    rows: List[Dict[str, Any]] = []
    best_corr = -np.inf

    # Pre-load data once to avoid repeated disk reads.
    eeg, targets, mask, times_ms, _ = load_stage2_dataset(dataset_dir, config)
    _, val_loader, _, _ = make_dataloaders(eeg, targets, mask, times_ms, config)

    for lam in lambda_cpp_prior_grid:
        sweep_cfg = replace(config, loss=replace(config.loss, lambda_cpp_prior=float(lam)))
        model = _quick_train(dataset_dir, sweep_cfg, n_epochs=n_epochs)

        val_metrics  = _eval_loss(model, val_loader, sweep_cfg, device)
        wave_metrics = _waveform_metrics(model, val_loader, device)

        row = {
            "lambda_cpp_prior": lam,
            "val_total_loss":   val_metrics.get("total_loss", float("nan")),
            "cpp_corr":         wave_metrics["cpp_corr"],
            "cpp_rmse":         wave_metrics["cpp_rmse"],
        }
        rows.append(row)

        if wave_metrics["cpp_corr"] > best_corr:
            best_corr = wave_metrics["cpp_corr"]

    # --- Persist results ------------------------------------------------------
    if rows:
        csv_path = output_dir / "sweep_results.csv"
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return {"score": best_corr, "results": rows}
