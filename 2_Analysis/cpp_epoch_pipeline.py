from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
import pandas as pd

from cpp_labeling import add_cpp_score_and_labels


DEFAULT_CPP_CHANNELS = ["CP1", "CPz", "CP2", "Pz"]


@dataclass(frozen=True)
class EpochConfig:
    tmin: float = -0.6
    tmax: float = 0.2
    baseline_start: float = -0.6
    baseline_end: float = -0.4
    slope_start: float = -0.25
    slope_end: float = -0.10
    amplitude_start: float = -0.05
    amplitude_end: float = 0.05
    post_start: float = 0.05
    post_end: float = 0.15
    monotonicity_start: float = -0.50
    monotonicity_end: float = -0.05


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    lowered = series.astype(str).str.strip().str.lower()
    mapped = lowered.map({"true": True, "false": False})
    return mapped.fillna(False)


def discover_subject_records(dataset_root: Path) -> list[dict[str, Path | str]]:
    records: list[dict[str, Path | str]] = []
    for subject_dir in sorted(dataset_root.glob("sub-*")):
        eeg_candidates = sorted(subject_dir.glob("eeg/*_eeg.vhdr"))
        beh_candidates = sorted(subject_dir.glob("beh/*_beh.csv"))
        if not eeg_candidates or not beh_candidates:
            continue
        records.append(
            {
                "subject_id": subject_dir.name,
                "eeg_path": eeg_candidates[0],
                "behavior_path": beh_candidates[0],
            }
        )
    if not records:
        raise FileNotFoundError(f"No subject records found under {dataset_root}")
    return records


def _prepare_behavior(behavior_path: Path, sfreq: float) -> pd.DataFrame:
    behavior = pd.read_csv(behavior_path)

    valid_trial_series = behavior["is_valid_trial"] if "is_valid_trial" in behavior.columns else None
    missing_response_series = behavior["is_missing_response"] if "is_missing_response" in behavior.columns else None

    is_valid = (
        _coerce_bool(pd.Series(valid_trial_series, index=behavior.index))
        if valid_trial_series is not None
        else pd.Series(True, index=behavior.index, dtype=bool)
    )
    has_response = (
        ~_coerce_bool(pd.Series(missing_response_series, index=behavior.index))
        if missing_response_series is not None
        else pd.Series(True, index=behavior.index, dtype=bool)
    )

    behavior = behavior.loc[is_valid & has_response].copy().reset_index(drop=True)
    behavior["probe_rt"] = pd.to_numeric(behavior["probe_rt"], errors="coerce")
    behavior["stim_onset"] = pd.to_numeric(behavior["stim_onset"], errors="coerce")
    behavior = behavior.loc[behavior["probe_rt"].notna() & behavior["stim_onset"].notna()].copy().reset_index(drop=True)

    if "resp_onset_sample" in behavior.columns:
        candidate_response = pd.Series(
            pd.to_numeric(behavior["resp_onset_sample"], errors="coerce"),
            index=behavior.index,
            dtype="float64",
        )
    else:
        candidate_response = pd.Series(np.nan, index=behavior.index, dtype="float64")

    derived_response = pd.Series(
        behavior["stim_onset"].to_numpy(dtype=float) + np.rint(behavior["probe_rt"].to_numpy(dtype=float) * sfreq),
        index=behavior.index,
        dtype="float64",
    )
    candidate_int = candidate_response.round().astype("Int64")
    stim_int = behavior["stim_onset"].round().astype("Int64")
    use_candidate = candidate_response.notna() & (candidate_int != stim_int)

    response_series = pd.Series(
        np.where(use_candidate.to_numpy(dtype=bool), candidate_response.to_numpy(dtype=float), derived_response.to_numpy(dtype=float)),
        index=behavior.index,
        dtype="float64",
    )
    behavior["response_sample"] = response_series
    behavior["response_sample"] = np.rint(behavior["response_sample"]).astype(int)
    behavior["response_sample_source"] = np.where(
        use_candidate.to_numpy(dtype=bool),
        "behavior_resp_onset_sample",
        "stim_onset_plus_probe_rt",
    )
    behavior["trial_index"] = np.arange(len(behavior), dtype=int)
    return behavior


def _pick_cpp_channels(raw: mne.io.BaseRaw, candidates: Iterable[str]) -> list[str]:
    available = [channel for channel in candidates if channel in raw.ch_names]
    if len(available) < 3:
        raise ValueError(
            "Not enough canonical CPP channels available. "
            f"Found {available!r} from candidates {list(candidates)!r}."
        )
    return available


def _window_mask(times: np.ndarray, start: float, end: float) -> np.ndarray:
    return (times >= start) & (times <= end)


def _linear_slope(values: np.ndarray, times: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = times[mask]
    selected = values[:, mask]
    return np.array([np.polyfit(x, row, deg=1)[0] for row in selected], dtype=float)


def _mean_amplitude(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return values[:, mask].mean(axis=1)


def _monotonicity(values: np.ndarray, times: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = times[mask]
    x_centered = x - x.mean()
    x_std = x_centered.std(ddof=0)
    if x_std == 0:
        return np.zeros(values.shape[0], dtype=float)
    x_norm = x_centered / x_std
    selected = values[:, mask]
    correlations = []
    for row in selected:
        row_centered = row - row.mean()
        row_std = row_centered.std(ddof=0)
        if row_std == 0:
            correlations.append(0.0)
        else:
            correlations.append(float(np.mean((row_centered / row_std) * x_norm)))
    return np.array(correlations, dtype=float)


def process_subject(
    subject_record: dict[str, Path | str],
    *,
    config: EpochConfig,
    cpp_channel_candidates: Iterable[str] = DEFAULT_CPP_CHANNELS,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    subject_id = str(subject_record["subject_id"])
    eeg_path = Path(subject_record["eeg_path"])
    behavior_path = Path(subject_record["behavior_path"])

    raw = mne.io.read_raw_brainvision(eeg_path, preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    behavior = _prepare_behavior(behavior_path, sfreq=sfreq)

    cpp_channels = _pick_cpp_channels(raw, cpp_channel_candidates)

    response_samples = behavior["response_sample"].to_numpy(dtype=int)
    lower_bound = int(round(abs(config.tmin) * sfreq))
    upper_bound = raw.n_times - int(round(config.tmax * sfreq)) - 1
    in_bounds = (response_samples >= lower_bound) & (response_samples <= upper_bound)
    behavior = behavior.loc[in_bounds].copy().reset_index(drop=True)
    response_samples = behavior["response_sample"].to_numpy(dtype=int)

    events = np.column_stack(
        [
            response_samples,
            np.zeros(len(response_samples), dtype=int),
            np.ones(len(response_samples), dtype=int),
        ]
    )
    event_id = {"derived_response": 1}

    epochs = mne.Epochs(
        raw=raw,
        events=events,
        event_id=event_id,
        tmin=config.tmin,
        tmax=config.tmax,
        baseline=(config.baseline_start, config.baseline_end),
        preload=True,
        event_repeated="drop",
        verbose="ERROR",
    )

    epoch_tensor = epochs.get_data(picks=cpp_channels).astype(np.float32)
    cpp_waveform = epoch_tensor.mean(axis=1)
    times = epochs.times

    slope_mask = _window_mask(times, config.slope_start, config.slope_end)
    amplitude_mask = _window_mask(times, config.amplitude_start, config.amplitude_end)
    post_mask = _window_mask(times, config.post_start, config.post_end)
    monotonicity_mask = _window_mask(times, config.monotonicity_start, config.monotonicity_end)

    metadata = behavior.copy()
    metadata["subject_id"] = subject_id
    metadata["cpp_channels"] = ",".join(cpp_channels)
    metadata["sampling_frequency"] = sfreq
    metadata["pre_response_slope"] = _linear_slope(cpp_waveform, times, slope_mask)
    metadata["late_amplitude"] = _mean_amplitude(cpp_waveform, amplitude_mask)
    metadata["post_response_amplitude"] = _mean_amplitude(cpp_waveform, post_mask)
    metadata["post_response_drop"] = metadata["late_amplitude"] - metadata["post_response_amplitude"]
    metadata["pre_response_monotonicity"] = _monotonicity(cpp_waveform, times, monotonicity_mask)
    metadata["epoch_n_times"] = cpp_waveform.shape[1]

    return metadata, epoch_tensor, cpp_waveform.astype(np.float32), times.astype(np.float32), cpp_channels


def export_cpp_epoch_dataset(
    dataset_root: Path,
    output_root: Path,
    *,
    config: EpochConfig | None = None,
    cpp_channel_candidates: Iterable[str] = DEFAULT_CPP_CHANNELS,
) -> dict[str, Path]:
    config = config or EpochConfig()
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_frames: list[pd.DataFrame] = []
    tensors: list[np.ndarray] = []
    waveforms: list[np.ndarray] = []
    channel_sets: set[tuple[str, ...]] = set()
    times_reference: np.ndarray | None = None

    records = discover_subject_records(dataset_root)
    tensor_offset = 0
    for record in records:
        metadata, epoch_tensor, cpp_waveform, times, cpp_channels = process_subject(
            record,
            config=config,
            cpp_channel_candidates=cpp_channel_candidates,
        )
        metadata = metadata.copy()
        metadata["tensor_row_index"] = np.arange(tensor_offset, tensor_offset + len(metadata), dtype=int)
        tensor_offset += len(metadata)

        metadata_frames.append(metadata)
        tensors.append(epoch_tensor)
        waveforms.append(cpp_waveform)
        channel_sets.add(tuple(cpp_channels))

        if times_reference is None:
            times_reference = times
        elif not np.allclose(times_reference, times):
            raise ValueError("Epoch times differ across subjects; cannot concatenate safely.")

    if not metadata_frames:
        raise RuntimeError("No epochs were exported from the dataset.")
    if len(channel_sets) != 1:
        raise ValueError(f"CPP channel selection was inconsistent across subjects: {channel_sets!r}")
    if times_reference is None:
        raise RuntimeError("Epoch time reference was never initialized.")

    metadata_all = pd.concat(metadata_frames, ignore_index=True)
    epoch_tensor_all = np.concatenate(tensors, axis=0).astype(np.float32)
    cpp_waveform_all = np.concatenate(waveforms, axis=0).astype(np.float32)
    metadata_all = add_cpp_score_and_labels(metadata_all)

    if len(metadata_all) != epoch_tensor_all.shape[0] or len(metadata_all) != cpp_waveform_all.shape[0]:
        raise ValueError("Metadata and tensor shapes are misaligned.")

    metadata_path = output_root / "epoch_metadata.csv"
    tensor_path = output_root / "epoch_tensor.npy"
    waveform_path = output_root / "epoch_cpp_waveform.npy"
    score_path = output_root / "epoch_score.npy"
    label_path = output_root / "epoch_label.npy"
    times_path = output_root / "epoch_times.npy"

    metadata_all.to_csv(metadata_path, index=False)
    np.save(tensor_path, epoch_tensor_all)
    np.save(waveform_path, cpp_waveform_all)
    np.save(score_path, metadata_all["cpp_score"].to_numpy(dtype=np.float32))
    np.save(label_path, metadata_all["cpp_label"].to_numpy(dtype=np.int64))
    np.save(times_path, np.asarray(times_reference, dtype=np.float32))

    return {
        "metadata_csv": metadata_path,
        "epoch_tensor_npy": tensor_path,
        "epoch_cpp_waveform_npy": waveform_path,
        "epoch_score_npy": score_path,
        "epoch_label_npy": label_path,
        "epoch_times_npy": times_path,
    }
