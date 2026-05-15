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


@dataclass
class Stage2SplitArtifacts:
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    train_mean: np.ndarray
    train_std: np.ndarray
    horizon_steps: int


def build_pre_response_mask(times_ms: np.ndarray, window_end_ms: np.ndarray | float, min_mask_lead_ms: int) -> np.ndarray:
    times_ms = np.asarray(times_ms, dtype=np.float32)
    if np.isscalar(window_end_ms):
        threshold = np.asarray([float(window_end_ms)], dtype=np.float32) - float(min_mask_lead_ms)
        return (times_ms[None, :] >= 0.0) & (times_ms[None, :] <= threshold[:, None])
    else:
        threshold = np.asarray(window_end_ms, dtype=np.float32) - float(min_mask_lead_ms)
        return (times_ms[None, :] >= 0.0) & (times_ms[None, :] <= threshold[:, None])


def _random_trial_split(n_trials: int, config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(n_trials)
    rng = np.random.default_rng(config.seed)
    rng.shuffle(indices)
    n_train = max(1, int(round(n_trials * config.train_fraction)))
    n_val = max(1, int(round(n_trials * config.val_fraction)))
    n_train = min(n_train, max(1, n_trials - 2))
    n_val = min(n_val, max(1, n_trials - n_train - 1))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    if len(test_indices) == 0:
        test_indices = val_indices[-1:]
        val_indices = val_indices[:-1]
    return train_indices, val_indices, test_indices


def _compute_channel_stats(eeg: np.ndarray, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    train_data = eeg[indices]
    mean = train_data.mean(axis=(0, 1))
    std = train_data.std(axis=(0, 1))
    std = np.where(std == 0.0, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _normalize_with_stats(eeg: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((eeg - mean[None, None, :]) / std[None, None, :]).astype(np.float32)


class EEGWindowDataset(Dataset):
    def __init__(
        self,
        eeg: np.ndarray,
        future_targets: np.ndarray,
        mask: np.ndarray,
        metadata: pd.DataFrame,
        indices: np.ndarray,
    ) -> None:
        self.eeg = torch.as_tensor(eeg[indices], dtype=torch.float32)
        self.future_targets = torch.as_tensor(future_targets[indices], dtype=torch.float32)
        self.mask = torch.as_tensor(mask[indices], dtype=torch.float32)
        self.metadata = metadata.iloc[indices].reset_index(drop=True)

    def __len__(self) -> int:
        return int(self.eeg.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "eeg": self.eeg[idx],
            "future_targets": self.future_targets[idx],
            "mask": self.mask[idx],
        }


def _build_future_targets(eeg: np.ndarray, horizon_steps: int) -> np.ndarray:
    """Build short-horizon causal prediction targets."""
    trials, timepoints, channels = eeg.shape
    targets = np.zeros((trials, timepoints, horizon_steps, channels), dtype=np.float32)
    for offset in range(1, horizon_steps + 1):
        if offset >= timepoints:
            continue
        targets[:, :-offset, offset - 1, :] = eeg[:, offset:, :]
    return targets


def _build_time_weights(times_ms: np.ndarray, config: TrainingConfig) -> np.ndarray:
    weights = np.zeros_like(times_ms, dtype=np.float32)
    early_mask = (times_ms >= config.early_window_ms[0]) & (times_ms < config.early_window_ms[1])
    mid_mask = (times_ms >= config.mid_window_ms[0]) & (times_ms < config.mid_window_ms[1])
    late_mask = (times_ms >= config.late_window_ms[0]) & (times_ms <= config.late_window_ms[1])
    weights[early_mask] = 1.0
    weights[mid_mask] = 1.75
    weights[late_mask] = 2.5
    return weights


def load_stage2_dataset(dataset_dir: Path, config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Stage2SplitArtifacts]:
    eeg = np.load(dataset_dir / "eeg_cpp_trials.npy").astype(np.float32)
    if not np.isfinite(eeg).all():
        eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0)
    times_ms = np.load(dataset_dir / "times_ms.npy").astype(np.float32)
    metadata = pd.read_csv(dataset_dir / "metadata.csv")
    metadata, missing_columns = _resolve_required_columns(metadata, DataContractConfig())
    if missing_columns:
        raise ValueError(f"Missing required metadata columns after alias resolution: {missing_columns}")
    channels = _read_channel_names(dataset_dir / "channel_names.txt")
    if tuple(channels) != DataContractConfig().expected_channel_order:
        raise ValueError(f"Unexpected channel order: {channels}")

    train_indices, val_indices, test_indices = _random_trial_split(len(metadata), config)
    train_mean, train_std = _compute_channel_stats(eeg, train_indices)
    eeg_normalized = _normalize_with_stats(eeg, train_mean, train_std)

    fs = 1000.0 / float(np.mean(np.diff(times_ms)))
    horizon_steps = max(1, int(round(config.future_horizon_ms * fs / 1000.0)))
    future_targets = _build_future_targets(eeg_normalized, horizon_steps)
    mask = (times_ms[None, :] >= config.analysis_window_ms[0]) & (times_ms[None, :] <= config.analysis_window_ms[1])
    valid_time = np.arange(len(times_ms)) <= (len(times_ms) - horizon_steps - 1)
    mask = mask & valid_time[None, :]
    time_weights = _build_time_weights(times_ms, config)

    artifacts = Stage2SplitArtifacts(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        train_mean=train_mean,
        train_std=train_std,
        horizon_steps=horizon_steps,
    )
    return eeg_normalized, future_targets, metadata, artifacts, time_weights


def make_dataloaders(dataset_dir: Path, config: TrainingConfig) -> Tuple[Dict[str, DataLoader], np.ndarray, pd.DataFrame, Stage2SplitArtifacts]:
    """Return train/val/test loaders plus the shared time axis."""
    set_global_seed(config.seed)
    eeg, future_targets, metadata, artifacts, time_weights = load_stage2_dataset(dataset_dir, config)
    times_ms = np.load(dataset_dir / "times_ms.npy").astype(np.float32)
    mask = (times_ms[None, :] >= config.analysis_window_ms[0]) & (times_ms[None, :] <= config.analysis_window_ms[1])
    mask = np.repeat(mask, eeg.shape[0], axis=0)
    valid_time = np.arange(len(times_ms)) <= (len(times_ms) - artifacts.horizon_steps - 1)
    mask = mask & valid_time[None, :]

    datasets = {
        "train": EEGWindowDataset(eeg, future_targets, mask * time_weights[None, :], metadata, artifacts.train_indices),
        "val": EEGWindowDataset(eeg, future_targets, mask * time_weights[None, :], metadata, artifacts.val_indices),
        "test": EEGWindowDataset(eeg, future_targets, mask * time_weights[None, :], metadata, artifacts.test_indices),
    }
    loaders = {
        split: DataLoader(dataset, batch_size=config.batch_size, shuffle=(split == "train"))
        for split, dataset in datasets.items()
    }
    return loaders, times_ms, metadata, artifacts
