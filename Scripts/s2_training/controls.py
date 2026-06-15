from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from modeling.config import TrainingConfig
from modeling.dataset import load_stage2_dataset, make_dataloaders
from modeling.model import CPPForwardGRU, ForwardOutputs, masked_self_supervised_loss
from modeling.utils import set_global_seed


# =============================================================================
# § 1  Internal Helpers
# =============================================================================

def _compute_mean_loss(
    model: CPPForwardGRU,
    loader,
    config: TrainingConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Return mean loss metrics for *loader* under *model* (no gradient).

    Parameters
    ----------
    model  : CPPForwardGRU (untrained or trained, eval mode enforced).
    loader : DataLoader to evaluate on.
    config : TrainingConfig carrying loss weights.
    device : Torch device.

    Returns
    -------
    Dict[str, float] with averaged metrics (same keys as ``masked_self_supervised_loss``).
    """
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


# =============================================================================
# § 2  Control Conditions
# =============================================================================

def run_minimal_controls(
    dataset_dir: Path,
    output_dir: Path,
    config: TrainingConfig,
) -> Dict[str, Dict[str, float]]:
    """Evaluate two baseline control conditions against the same data.

    Control conditions
    ------------------
    ``"untrained"``
        A freshly initialised (random-weight) CPPForwardGRU evaluated on the
        validation set.  Establishes the loss floor for an uninformed model.

    ``"shuffled"``
        The same random model evaluated on time-shuffled EEG (each trial's
        time dimension is independently permuted).  Verifies that any structure
        the model learns is time-ordered rather than purely distributional.

    Parameters
    ----------
    dataset_dir : Processed dataset directory.
    output_dir  : Directory to write ``control_results.json`` into (created
                  if it does not exist).
    config      : TrainingConfig used to build the dataset and model.

    Returns
    -------
    Dict with keys ``"untrained"`` and ``"shuffled"``, each mapping to a
    Dict[str, float] of averaged loss metrics.
    """
    import json

    set_global_seed(config.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    # --- Load data ------------------------------------------------------------
    eeg, targets, mask, times_ms, _ = load_stage2_dataset(dataset_dir, config)
    _, val_loader, _, _ = make_dataloaders(eeg, targets, mask, times_ms, config)
    n_channels = eeg.shape[-1]

    # --- Untrained control ---------------------------------------------------
    model_untrained = CPPForwardGRU(n_channels=n_channels, model_config=config.model).to(device)
    untrained_metrics = _compute_mean_loss(model_untrained, val_loader, config, device)

    # --- Time-shuffled control -----------------------------------------------
    # Build a shuffled version of the DataLoader by permuting each trial's
    # time axis independently before wrapping into a new Dataset.
    rng = np.random.default_rng(config.seed + 1)
    eeg_shuffled = eeg.copy()
    for i in range(eeg_shuffled.shape[0]):
        perm = rng.permutation(eeg_shuffled.shape[1])
        eeg_shuffled[i] = eeg_shuffled[i, perm, :]

    from modeling.dataset import EEGWindowDataset
    from torch.utils.data import DataLoader as DL

    # Reuse split indices from the standard split for a fair comparison.
    from modeling.dataset import _random_trial_split, _build_trial_mask
    from pathlib import Path as P
    import numpy as _np

    train_idx, val_idx, _ = _random_trial_split(eeg.shape[0], config)
    fs_hz = 1000.0 / float(np.mean(np.diff(times_ms)))
    horizon_steps = max(1, int(round(config.future_horizon_ms * fs_hz / 1000.0)))
    mask_shuf = _build_trial_mask(times_ms, eeg.shape[0], config, horizon_steps)
    targets_shuf = np.zeros_like(targets)
    if eeg_shuffled.shape[1] > 1:
        targets_shuf[:, :-1, :] = eeg_shuffled[:, 1:, :]

    ds_shuf = EEGWindowDataset(eeg_shuffled, targets_shuf, mask_shuf, times_ms, val_idx)
    loader_shuf = DL(ds_shuf, batch_size=config.batch_size, shuffle=False)

    model_shuffled = CPPForwardGRU(n_channels=n_channels, model_config=config.model).to(device)
    shuffled_metrics = _compute_mean_loss(model_shuffled, loader_shuf, config, device)

    result = {
        "untrained": untrained_metrics,
        "shuffled":  shuffled_metrics,
    }

    # --- Persist results ------------------------------------------------------
    with open(output_dir / "control_results.json", "w") as fh:
        json.dump(result, fh, indent=2)

    return result
