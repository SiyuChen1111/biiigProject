from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from src.data.data_kosciessa import DATASET_DIR, get_subject_paths


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
