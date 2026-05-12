from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from src.data.data_kosciessa import DATASET_DIR, get_subject_paths
from src.features.labels_cpp import CPP_ROI_CHANNELS


PHASE1_EEG_CHANNEL_COUNT = 60
PHASE1_TMIN_SECONDS = -0.6
PHASE1_TMAX_SECONDS = 0.2
PHASE1_BASELINE = (-0.6, -0.4)
PHASE1_TARGET_SFREQ = 128.0


@dataclass(frozen=True)
class EpochedSubjectData:
    subject_id: str
    epochs: np.ndarray
    times: np.ndarray
    channel_names: list[str]
    trial_table: pd.DataFrame
    original_sfreq: float
    target_sfreq: float


@dataclass(frozen=True)
class RoiEpochedSubjectData:
    """ROI-only view of response-locked epochs.

    Contract:
    - epochs: (trials, 3, samples)
    - channel_names: exactly CPP_ROI_CHANNELS (order-preserving)
    - times and trial_table are preserved from the full-head epoched input
    """

    subject_id: str
    epochs: np.ndarray
    times: np.ndarray
    channel_names: list[str]
    trial_table: pd.DataFrame
    original_sfreq: float
    target_sfreq: float


def roi_channel_indices(channel_names: list[str], roi_channels: list[str] | None = None) -> list[int]:
    """Return indices for ROI channels in the requested order.

    Raises:
        ValueError: if any ROI channel is missing.
    """

    roi = roi_channels or CPP_ROI_CHANNELS
    index_by_name = {name: idx for idx, name in enumerate(channel_names)}
    missing = [name for name in roi if name not in index_by_name]
    if missing:
        raise ValueError(
            "Missing ROI channels: "
            + ", ".join(missing)
            + f". Available channels: {channel_names}"
        )
    return [index_by_name[name] for name in roi]


def roi_only_epochs(
    epochs: np.ndarray,
    channel_names: list[str],
    roi_channels: list[str] | None = None,
) -> np.ndarray:
    """Select ROI-only epochs from full-head epochs.

    Args:
        epochs: (trials, channels, samples)
        channel_names: names aligned to the channel axis.
        roi_channels: optional override; defaults to CPP_ROI_CHANNELS.

    Returns:
        ROI-only epochs with shape (trials, len(roi_channels), samples)
        and channel order matching roi_channels.
    """

    if epochs.ndim != 3:
        raise ValueError(f"Expected epochs with 3 dimensions (trials, channels, samples); got shape {epochs.shape}.")

    roi = roi_channels or CPP_ROI_CHANNELS
    roi_idx = roi_channel_indices(channel_names=channel_names, roi_channels=roi)
    roi_epochs = epochs[:, roi_idx, :]
    if roi_epochs.shape[1] != len(roi):
        raise ValueError(
            f"ROI selection produced {roi_epochs.shape[1]} channels; expected {len(roi)}. "
            f"ROI channels={roi}"
        )
    return roi_epochs


def to_roi_epoched_subject_data(
    epoched: EpochedSubjectData,
    roi_channels: list[str] | None = None,
) -> RoiEpochedSubjectData:
    """Convert a full-head EpochedSubjectData into an ROI-only contract."""

    roi = roi_channels or CPP_ROI_CHANNELS
    roi_epochs = roi_only_epochs(epochs=epoched.epochs, channel_names=epoched.channel_names, roi_channels=roi)
    return RoiEpochedSubjectData(
        subject_id=epoched.subject_id,
        epochs=roi_epochs.astype(np.float32, copy=False),
        times=epoched.times,
        channel_names=list(roi),
        trial_table=epoched.trial_table,
        original_sfreq=epoched.original_sfreq,
        target_sfreq=epoched.target_sfreq,
    )


def load_raw_eeg(subject_id: str, dataset_dir: Path = DATASET_DIR) -> mne.io.BaseRaw:
    paths = get_subject_paths(subject_id=subject_id, dataset_dir=dataset_dir)
    raw = mne.io.read_raw_brainvision(paths.eeg_vhdr, preload=True, verbose="ERROR")
    eeg_channel_names = [name for name in raw.ch_names if name not in {"A1", "VEOG", "HEOGL", "HEOGR", "ECG"}]
    raw.pick(eeg_channel_names)
    raw.set_eeg_reference("average", verbose="ERROR")
    raw.filter(l_freq=1.0, h_freq=30.0, verbose="ERROR")
    return raw


def _baseline_correct(epoch: np.ndarray, times: np.ndarray, baseline: tuple[float, float]) -> np.ndarray:
    baseline_mask = (times >= baseline[0]) & (times <= baseline[1])
    if not baseline_mask.any():
        raise ValueError("Baseline window produced no samples.")
    baseline_mean = epoch[:, baseline_mask].mean(axis=1, keepdims=True)
    return epoch - baseline_mean


def _extract_epoch(
    signal: np.ndarray,
    anchor_sample: int,
    sfreq: float,
    tmin: float,
    tmax: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    start_offset = int(round(tmin * sfreq))
    end_offset = int(round(tmax * sfreq))
    start = anchor_sample + start_offset
    stop = anchor_sample + end_offset
    if start < 0 or stop >= signal.shape[1]:
        return None
    epoch = signal[:, start : stop + 1]
    times = np.arange(epoch.shape[1], dtype=float) / sfreq + tmin
    return epoch, times


def _resample_epoch(epoch: np.ndarray, up: float, down: float) -> np.ndarray:
    return mne.filter.resample(epoch, up=up, down=down, axis=-1, verbose="ERROR")


def extract_response_locked_epochs(
    subject_id: str,
    trial_table: pd.DataFrame,
    dataset_dir: Path = DATASET_DIR,
    tmin: float = PHASE1_TMIN_SECONDS,
    tmax: float = PHASE1_TMAX_SECONDS,
    baseline: tuple[float, float] = PHASE1_BASELINE,
    target_sfreq: float = PHASE1_TARGET_SFREQ,
) -> EpochedSubjectData:
    raw = load_raw_eeg(subject_id=subject_id, dataset_dir=dataset_dir)
    signal = np.asarray(raw.get_data(), dtype=np.float64)
    channel_names = list(raw.ch_names)
    sfreq = float(raw.info["sfreq"])

    kept_epochs: list[np.ndarray] = []
    kept_rows: list[pd.Series] = []
    resampled_times: np.ndarray | None = None

    for _, row in trial_table.iterrows():
        extracted = _extract_epoch(
            signal=signal,
            anchor_sample=int(row["resp_onset_sample"]),
            sfreq=sfreq,
            tmin=tmin,
            tmax=tmax,
        )
        if extracted is None:
            continue

        epoch, times = extracted
        epoch = _baseline_correct(epoch=epoch, times=times, baseline=baseline)
        epoch = _resample_epoch(epoch=epoch, up=target_sfreq, down=sfreq)

        if resampled_times is None:
            resampled_times = np.linspace(tmin, tmax, epoch.shape[-1], endpoint=True)

        kept_epochs.append(epoch.astype(np.float32))
        kept_rows.append(row)

    if not kept_epochs:
        raise ValueError(f"No valid epochs extracted for subject {subject_id}.")

    epochs = np.stack(kept_epochs, axis=0)
    kept_trial_table = pd.DataFrame(kept_rows).reset_index(drop=True)

    if epochs.shape[1] != PHASE1_EEG_CHANNEL_COUNT:
        raise ValueError(
            f"Expected {PHASE1_EEG_CHANNEL_COUNT} EEG channels, got {epochs.shape[1]} for {subject_id}."
        )

    return EpochedSubjectData(
        subject_id=subject_id,
        epochs=epochs,
        times=resampled_times if resampled_times is not None else np.array([], dtype=float),
        channel_names=channel_names,
        trial_table=kept_trial_table,
        original_sfreq=sfreq,
        target_sfreq=target_sfreq,
    )
