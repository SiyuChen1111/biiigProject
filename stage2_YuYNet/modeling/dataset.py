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
from .utils import safe_float, set_global_seed


@dataclass
class Stage2SplitArtifacts:
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    subject_stats: Dict[str, Dict[str, List[float]]]
    horizon_steps: int


def build_pre_response_mask(times_ms: np.ndarray, rt_ms: np.ndarray, min_mask_lead_ms: int) -> np.ndarray:
    threshold = rt_ms[:, None] - float(min_mask_lead_ms)
    return (times_ms[None, :] >= 0.0) & (times_ms[None, :] <= threshold)


def _assign_subject_splits(subject_ids: np.ndarray, config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_subjects = np.array(sorted(pd.unique(subject_ids)))
    rng = np.random.default_rng(config.seed)
    rng.shuffle(unique_subjects)
    total = len(unique_subjects)
    n_train = max(1, int(round(total * config.train_fraction)))
    n_val = max(1, int(round(total * config.val_fraction))) if total >= 3 else 1
    n_train = min(n_train, max(1, total - 2)) if total >= 3 else max(1, total - 1)
    n_val = min(n_val, max(1, total - n_train - 1)) if total >= 3 else 0
    train_subjects = unique_subjects[:n_train]
    val_subjects = unique_subjects[n_train:n_train + n_val]
    test_subjects = unique_subjects[n_train + n_val:]
    if len(test_subjects) == 0:
        test_subjects = val_subjects[-1:]
        val_subjects = val_subjects[:-1]
    return train_subjects, val_subjects, test_subjects


def _compute_subject_channel_stats(eeg: np.ndarray, metadata: pd.DataFrame, train_indices: np.ndarray) -> Dict[str, Dict[str, List[float]]]:
    stats: Dict[str, Dict[str, List[float]]] = {}
    train_metadata = metadata.iloc[train_indices]
    for subject_id, subject_rows in train_metadata.groupby("subject_id"):
        subject_frame = subject_rows.reset_index()
        subject_idx = subject_frame["index"].to_numpy(dtype=int)
        subject_data = eeg[subject_idx]
        mean = subject_data.mean(axis=(0, 1))
        std = subject_data.std(axis=(0, 1))
        std = np.where(std == 0.0, 1.0, std)
        stats[str(subject_id)] = {"mean": mean.tolist(), "std": std.tolist()}
    return stats


def _normalize_subjectwise(eeg: np.ndarray, metadata: pd.DataFrame, stats: Dict[str, Dict[str, List[float]]]) -> np.ndarray:
    normalized = np.empty_like(eeg, dtype=np.float32)
    subject_id_series = metadata.reset_index(drop=True)["subject_id"].astype(str)
    for index, subject_id in enumerate(subject_id_series.tolist()):
        if subject_id not in stats:
            global_mean = np.mean([item["mean"] for item in stats.values()], axis=0)
            global_std = np.mean([item["std"] for item in stats.values()], axis=0)
            mean = np.asarray(global_mean)
            std = np.asarray(global_std)
        else:
            mean = np.asarray(stats[subject_id]["mean"])
            std = np.asarray(stats[subject_id]["std"])
        normalized[index] = ((eeg[index] - mean) / std).astype(np.float32)
    return normalized


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


def load_stage2_dataset(dataset_dir: Path, config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Stage2SplitArtifacts]:
    eeg = np.load(dataset_dir / "eeg_cpp_trials.npy").astype(np.float32)
    times_ms = np.load(dataset_dir / "times_ms.npy").astype(np.float32)
    metadata = pd.read_csv(dataset_dir / "metadata.csv")
    metadata, missing_columns = _resolve_required_columns(metadata, DataContractConfig())
    if missing_columns:
        raise ValueError(f"Missing required metadata columns after alias resolution: {missing_columns}")
    channels = _read_channel_names(dataset_dir / "channel_names.txt")
    if tuple(channels) != DataContractConfig().expected_channel_order:
        raise ValueError(f"Unexpected channel order: {channels}")

    rt_ms = metadata["RT_ms"].map(safe_float).to_numpy(dtype=np.float32)
    subject_ids = metadata["subject_id"].astype(str).to_numpy()
    train_subjects, val_subjects, test_subjects = _assign_subject_splits(subject_ids, config)
    train_indices = np.where(np.isin(subject_ids, train_subjects))[0]
    val_indices = np.where(np.isin(subject_ids, val_subjects))[0]
    test_indices = np.where(np.isin(subject_ids, test_subjects))[0]

    subject_stats = _compute_subject_channel_stats(eeg, metadata, train_indices)
    eeg_normalized = _normalize_subjectwise(eeg, metadata, subject_stats)

    fs = 1000.0 / float(np.mean(np.diff(times_ms)))
    horizon_steps = max(1, int(round(config.future_horizon_ms * fs / 1000.0)))
    future_targets = _build_future_targets(eeg_normalized, horizon_steps)
    mask = build_pre_response_mask(times_ms, rt_ms, config.min_mask_lead_ms)
    valid_time = np.arange(len(times_ms)) <= (len(times_ms) - horizon_steps - 1)
    mask = mask & valid_time[None, :]

    artifacts = Stage2SplitArtifacts(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        subject_stats=subject_stats,
        horizon_steps=horizon_steps,
    )
    return eeg_normalized, future_targets, metadata, artifacts


def make_dataloaders(dataset_dir: Path, config: TrainingConfig) -> Tuple[Dict[str, DataLoader], np.ndarray, pd.DataFrame, Stage2SplitArtifacts]:
    """Return train/val/test loaders plus the shared time axis."""
    set_global_seed(config.seed)
    eeg, future_targets, metadata, artifacts = load_stage2_dataset(dataset_dir, config)
    rt_ms = metadata["RT_ms"].map(safe_float).to_numpy(dtype=np.float32)
    times_ms = np.load(dataset_dir / "times_ms.npy").astype(np.float32)
    mask = build_pre_response_mask(times_ms, rt_ms, config.min_mask_lead_ms)
    valid_time = np.arange(len(times_ms)) <= (len(times_ms) - artifacts.horizon_steps - 1)
    mask = mask & valid_time[None, :]

    datasets = {
        "train": EEGWindowDataset(eeg, future_targets, mask, metadata, artifacts.train_indices),
        "val": EEGWindowDataset(eeg, future_targets, mask, metadata, artifacts.val_indices),
        "test": EEGWindowDataset(eeg, future_targets, mask, metadata, artifacts.test_indices),
    }
    loaders = {
        split: DataLoader(dataset, batch_size=config.batch_size, shuffle=(split == "train"))
        for split, dataset in datasets.items()
    }
    return loaders, times_ms, metadata, artifacts
