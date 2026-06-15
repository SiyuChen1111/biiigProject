from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import DataContractConfig, TrainingConfig
from .data_contract import _resolve_required_columns, _read_channel_names
from .utils import set_global_seed


# =============================================================================
# § 1  Split Metadata Container
# =============================================================================

@dataclass
class Stage2SplitArtifacts:
    """Holds the trial indices and normalisation statistics for a data split.

    Attributes
    ----------
    train_indices : 1-D array of trial indices assigned to the training set.
    val_indices   : 1-D array of trial indices assigned to the validation set.
    test_indices  : 1-D array of trial indices assigned to the test set.
    train_mean    : (C,) per-channel mean computed on the training set only.
    train_std     : (C,) per-channel std  computed on the training set only.
    horizon_steps : Number of time steps in the causal prediction horizon.
    """

    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    train_mean: np.ndarray
    train_std: np.ndarray
    horizon_steps: int


# =============================================================================
# § 2  Mask Construction
# =============================================================================

def _build_trial_mask(
    times_ms: np.ndarray,
    n_trials: int,
    config: TrainingConfig,
    horizon_steps: int,
) -> np.ndarray:
    """Build a (n_trials, T) boolean mask for valid loss-computation time steps.

    A time step is considered valid when:
      1. It falls within ``config.analysis_window_ms`` (the pre-response EEG
         window of interest), AND
      2. It is not among the last ``horizon_steps`` steps of the epoch.
         Those final steps lack complete causal future targets and must be
         excluded to avoid training on zero-padded targets.

    Parameters
    ----------
    times_ms      : (T,) time axis in milliseconds (response-locked).
    n_trials      : Number of trials (first axis of the mask).
    config        : TrainingConfig carrying ``analysis_window_ms``.
    horizon_steps : Number of future steps in the prediction target; the last
                    ``horizon_steps`` time steps are excluded.

    Returns
    -------
    mask : (n_trials, T) bool array.
    """
    in_window = (
        (times_ms >= config.analysis_window_ms[0])
        & (times_ms <= config.analysis_window_ms[1])
    )  # (T,)

    # Exclude the last horizon_steps time steps (no complete future targets).
    has_future = np.arange(len(times_ms)) <= (len(times_ms) - horizon_steps - 1)  # (T,)

    row_mask = in_window & has_future  # (T,)
    return np.broadcast_to(row_mask[None, :], (n_trials, len(times_ms))).copy()


def build_pre_response_mask(
    times_ms: np.ndarray,
    window_end_ms: np.ndarray | float,
    min_mask_lead_ms: int,
) -> np.ndarray:
    """Build a variable-endpoint pre-response validity mask.

    Used in analyses where each trial has a different response time and we
    want to mask out post-response contamination per trial.

    Parameters
    ----------
    times_ms        : (T,) shared time axis.
    window_end_ms   : Scalar or (N,) array of per-trial window end times.
    min_mask_lead_ms: Safety margin in ms subtracted from each end time.

    Returns
    -------
    mask : (1, T) or (N, T) bool array.
    """
    times_ms = np.asarray(times_ms, dtype=np.float32)
    if np.isscalar(window_end_ms):
        threshold = (
            np.asarray([float(window_end_ms)], dtype=np.float32)
            - float(min_mask_lead_ms)
        )
    else:
        threshold = np.asarray(window_end_ms, dtype=np.float32) - float(min_mask_lead_ms)
    return (times_ms[None, :] >= 0.0) & (times_ms[None, :] <= threshold[:, None])


# =============================================================================
# § 3  Split & Normalisation Helpers
# =============================================================================

def _random_trial_split(
    n_trials: int,
    config: TrainingConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly partition trials into train / val / test sets.

    Parameters
    ----------
    n_trials : Total number of trials.
    config   : TrainingConfig carrying seed and fraction fields.

    Returns
    -------
    train_indices, val_indices, test_indices : non-overlapping index arrays.
    """
    indices = np.arange(n_trials)
    rng = np.random.default_rng(config.seed)
    rng.shuffle(indices)
    n_train = max(1, int(round(n_trials * config.train_fraction)))
    n_val   = max(1, int(round(n_trials * config.val_fraction)))
    n_train = min(n_train, max(1, n_trials - 2))
    n_val   = min(n_val,   max(1, n_trials - n_train - 1))
    train_indices = indices[:n_train]
    val_indices   = indices[n_train : n_train + n_val]
    test_indices  = indices[n_train + n_val :]
    if len(test_indices) == 0:
        test_indices = val_indices[-1:]
        val_indices  = val_indices[:-1]
    return train_indices, val_indices, test_indices


def _compute_channel_stats(
    eeg: np.ndarray,
    indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std from the training subset only.

    Parameters
    ----------
    eeg     : (N, T, C) raw EEG array.
    indices : Trial indices belonging to the training set.

    Returns
    -------
    mean : (C,) float32 array.
    std  : (C,) float32 array (zero channels replaced with 1.0).
    """
    train_data = eeg[indices]
    mean = train_data.mean(axis=(0, 1))
    std  = train_data.std(axis=(0, 1))
    std  = np.where(std == 0.0, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _normalize_with_stats(
    eeg: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply z-score normalisation using pre-computed mean and std.

    Parameters
    ----------
    eeg  : (N, T, C) array to normalise.
    mean : (C,) per-channel mean.
    std  : (C,) per-channel std.

    Returns
    -------
    Normalised (N, T, C) float32 array.
    """
    return ((eeg - mean[None, None, :]) / std[None, None, :]).astype(np.float32)


# =============================================================================
# § 4  Future-Target & Time-Weight Construction
# =============================================================================

def _build_future_targets(
    eeg: np.ndarray,
    horizon_steps: int,
) -> np.ndarray:
    """Build single-step-ahead causal prediction targets.

    For each time step t, the target is the EEG at time t+1
    (i.e. a 1-step horizon).  The last ``horizon_steps`` positions are
    left as zero because no complete future window exists there.

    Parameters
    ----------
    eeg           : (N, T, C) normalised EEG array.
    horizon_steps : Number of look-ahead steps (typically 1 at 1-step horizon).

    Returns
    -------
    targets : (N, T, C) float32 array — EEG at t+1 for each valid t.
    """
    n_trials, n_timepoints, n_channels = eeg.shape
    targets = np.zeros((n_trials, n_timepoints, n_channels), dtype=np.float32)
    # Shift EEG forward by 1 step: target[t] = eeg[t+1]
    if n_timepoints > 1:
        targets[:, :-1, :] = eeg[:, 1:, :]
    return targets


def _build_time_weights(
    times_ms: np.ndarray,
    config: TrainingConfig,
) -> np.ndarray:
    """Assign per-time-step training weights that up-weight the late pre-response window.

    Weights increase from early (1.0) → mid (1.75) → late (2.5) to
    steer the model toward the CPP build-up region that matters most
    for behaviour.

    Parameters
    ----------
    times_ms : (T,) time axis.
    config   : TrainingConfig carrying early/mid/late window boundaries.

    Returns
    -------
    weights : (T,) float32 array.
    """
    weights = np.zeros_like(times_ms, dtype=np.float32)
    early_mask = (times_ms >= config.early_window_ms[0]) & (times_ms < config.early_window_ms[1])
    mid_mask   = (times_ms >= config.mid_window_ms[0])   & (times_ms < config.mid_window_ms[1])
    late_mask  = (times_ms >= config.late_window_ms[0])  & (times_ms <= config.late_window_ms[1])
    weights[early_mask] = 1.0
    weights[mid_mask]   = 1.75
    weights[late_mask]  = 2.5
    return weights


# =============================================================================
# § 5  PyTorch Dataset
# =============================================================================

class EEGWindowDataset(Dataset):
    """PyTorch Dataset wrapping a trial-level EEG array and its auxiliary tensors.

    Each item is a dict with keys:
      ``eeg``          : (T, C) float tensor — normalised EEG input.
      ``target_future``: (T, C) float tensor — one-step-ahead prediction target.
      ``mask``         : (T,)   float tensor — 1 at valid loss-computation steps.
      ``times_ms``     : (T,)   float tensor — shared time axis.
      ``trial_idx``    : ()     long tensor  — original trial index in the full dataset.

    Parameters
    ----------
    eeg      : (N, T, C) normalised EEG array.
    targets  : (N, T, C) future-target array.
    mask     : (N, T)    boolean/float mask array.
    times_ms : (T,)      time axis.
    indices  : 1-D array of trial indices to include in this split.
    """

    def __init__(
        self,
        eeg: np.ndarray,
        targets: np.ndarray,
        mask: np.ndarray,
        times_ms: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.eeg      = torch.as_tensor(eeg[indices],     dtype=torch.float32)
        self.targets  = torch.as_tensor(targets[indices], dtype=torch.float32)
        self.mask     = torch.as_tensor(mask[indices],    dtype=torch.float32)
        self.times_ms = torch.as_tensor(times_ms,         dtype=torch.float32)
        self.indices  = torch.as_tensor(indices,          dtype=torch.long)

    def __len__(self) -> int:
        return int(self.eeg.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "eeg":          self.eeg[idx],
            "target_future": self.targets[idx],
            "mask":         self.mask[idx],
            "times_ms":     self.times_ms,
            "trial_idx":    self.indices[idx],
        }


# =============================================================================
# § 6  Public API
# =============================================================================

def load_stage2_dataset(
    dataset_dir: Path,
    config: TrainingConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load, validate, normalise, and mask the Stage 2 EEG dataset.

    Parameters
    ----------
    dataset_dir : Directory containing ``eeg_cpp_trials.npy``, ``times_ms.npy``,
                  ``metadata.csv``, ``channel_names.txt``, and
                  ``preprocessing_notes.md`` (as required by DataContractConfig).
    config      : TrainingConfig driving split fractions, normalisation, and
                  mask boundaries.

    Returns
    -------
    eeg      : (N, T, C) float32 — z-scored EEG (training stats).
    targets  : (N, T, C) float32 — one-step-ahead prediction targets.
    mask     : (N, T)    bool    — valid time-step mask (analysis window,
                                   horizon boundary excluded).
    times_ms : (T,)      float32 — shared time axis.
    metadata : pd.DataFrame      — per-trial behavioural metadata.
    """
    # --- Load raw arrays ------------------------------------------------------
    eeg = np.load(dataset_dir / "eeg_cpp_trials.npy").astype(np.float32)
    if not np.isfinite(eeg).all():
        eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0)
    times_ms = np.load(dataset_dir / "times_ms.npy").astype(np.float32)
    metadata = pd.read_csv(dataset_dir / "metadata.csv")

    # --- Contract validation --------------------------------------------------
    metadata, missing_columns = _resolve_required_columns(metadata, DataContractConfig())
    if missing_columns:
        raise ValueError(
            f"Missing required metadata columns after alias resolution: {missing_columns}"
        )
    channels = _read_channel_names(dataset_dir / "channel_names.txt")
    if tuple(channels) != DataContractConfig().expected_channel_order:
        raise ValueError(f"Unexpected channel order: {channels}")

    # --- Split & normalise ----------------------------------------------------
    train_idx, _, _ = _random_trial_split(len(metadata), config)
    train_mean, train_std = _compute_channel_stats(eeg, train_idx)
    eeg_norm = _normalize_with_stats(eeg, train_mean, train_std)

    # --- Causal horizon & targets ---------------------------------------------
    fs = 1000.0 / float(np.mean(np.diff(times_ms)))
    horizon_steps = max(1, int(round(config.future_horizon_ms * fs / 1000.0)))
    targets = _build_future_targets(eeg_norm, horizon_steps)

    # --- Mask (single shared call) -------------------------------------------
    n_trials = eeg_norm.shape[0]
    mask = _build_trial_mask(times_ms, n_trials, config, horizon_steps)

    return eeg_norm, targets, mask, times_ms, metadata


def make_dataloaders(
    eeg: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    times_ms: np.ndarray,
    config: TrainingConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, np.ndarray]]:
    """Wrap normalised EEG arrays into train / val / test DataLoaders.

    Parameters
    ----------
    eeg      : (N, T, C) normalised EEG (output of load_stage2_dataset).
    targets  : (N, T, C) future-prediction targets.
    mask     : (N, T)    valid-step boolean mask.
    times_ms : (T,)      time axis.
    config   : TrainingConfig driving batch size, seed, and time weights.

    Returns
    -------
    train_loader : DataLoader shuffled over training trials.
    val_loader   : DataLoader over validation trials (no shuffle).
    test_loader  : DataLoader over test trials (no shuffle).
    split_indices: Dict with keys ``"train"``, ``"val"``, ``"test"`` mapping
                   to 1-D index arrays.
    """
    set_global_seed(config.seed)
    train_idx, val_idx, test_idx = _random_trial_split(eeg.shape[0], config)
    time_weights = _build_time_weights(times_ms, config)

    # Scale mask by per-time-step weights (upweights the late CPP window).
    weighted_mask = mask * time_weights[None, :]  # (N, T) float

    def _make_loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        ds = EEGWindowDataset(eeg, targets, weighted_mask, times_ms, indices)
        return DataLoader(ds, batch_size=config.batch_size, shuffle=shuffle)

    train_loader = _make_loader(train_idx, shuffle=True)
    val_loader   = _make_loader(val_idx,   shuffle=False)
    test_loader  = _make_loader(test_idx,  shuffle=False)

    split_indices: Dict[str, np.ndarray] = {
        "train": train_idx,
        "val":   val_idx,
        "test":  test_idx,
    }
    return train_loader, val_loader, test_loader, split_indices
