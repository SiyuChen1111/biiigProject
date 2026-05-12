from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CPP_ROI_CHANNELS = ["CPz", "CP1", "CP2"]
AMS_WINDOW = (-0.18, -0.08)
PAMS_WINDOW = (-0.05, 0.05)
SLPS_WINDOW = (-0.25, -0.05)


@dataclass(frozen=True)
class CPPLabelTransform:
    ams_mean: float
    ams_std: float
    pams_mean: float
    pams_std: float
    slps_mean: float
    slps_std: float
    lower_threshold: float
    upper_threshold: float


def _window_mask(times: np.ndarray, start: float, end: float) -> np.ndarray:
    mask = (times >= start) & (times <= end)
    if not mask.any():
        raise ValueError(f"No samples found in window [{start}, {end}].")
    return mask


def _roi_indices(channel_names: list[str], roi_channels: list[str] | None = None) -> list[int]:
    roi = roi_channels or CPP_ROI_CHANNELS
    index_by_name = {name: idx for idx, name in enumerate(channel_names)}
    missing = [name for name in roi if name not in index_by_name]
    if missing:
        raise ValueError(f"Missing ROI channels: {missing}")
    return [index_by_name[name] for name in roi]


def compute_cpp_features(
    epochs: np.ndarray,
    times: np.ndarray,
    channel_names: list[str],
    roi_channels: list[str] | None = None,
) -> dict[str, np.ndarray]:
    roi_idx = _roi_indices(channel_names=channel_names, roi_channels=roi_channels)
    roi_signal = epochs[:, roi_idx, :].mean(axis=1)

    ams_mask = _window_mask(times, *AMS_WINDOW)
    pams_mask = _window_mask(times, *PAMS_WINDOW)
    slps_mask = _window_mask(times, *SLPS_WINDOW)

    ams = roi_signal[:, ams_mask].mean(axis=1)
    pams = roi_signal[:, pams_mask].max(axis=1)

    slope_times = times[slps_mask].astype(np.float64)
    x_mean = slope_times.mean()
    x_centered = slope_times - x_mean
    denominator = np.sum(x_centered**2)
    if denominator == 0:
        raise ValueError("Degenerate SLPS time window produced zero denominator.")
    slope_signal = roi_signal[:, slps_mask].astype(np.float64)
    slope_y_mean = slope_signal.mean(axis=1, keepdims=True)
    slps = np.sum(x_centered * (slope_signal - slope_y_mean), axis=1) / denominator

    return {
        "ams": ams.astype(np.float32),
        "pams": pams.astype(np.float32),
        "slps": slps.astype(np.float32),
    }


def _safe_zscore(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    denom = std if std > 1e-8 else 1.0
    return (values - mean) / denom


def fit_cpp_label_transform(features: dict[str, np.ndarray]) -> CPPLabelTransform:
    ams = features["ams"].astype(np.float64)
    pams = features["pams"].astype(np.float64)
    slps = features["slps"].astype(np.float64)

    ams_mean, ams_std = float(ams.mean()), float(ams.std(ddof=0))
    pams_mean, pams_std = float(pams.mean()), float(pams.std(ddof=0))
    slps_mean, slps_std = float(slps.mean()), float(slps.std(ddof=0))

    score = (
        _safe_zscore(ams, ams_mean, ams_std)
        + _safe_zscore(pams, pams_mean, pams_std)
        + _safe_zscore(slps, slps_mean, slps_std)
    )
    lower = float(np.percentile(score, 25.0))
    upper = float(np.percentile(score, 75.0))

    return CPPLabelTransform(
        ams_mean=ams_mean,
        ams_std=ams_std,
        pams_mean=pams_mean,
        pams_std=pams_std,
        slps_mean=slps_mean,
        slps_std=slps_std,
        lower_threshold=lower,
        upper_threshold=upper,
    )


def transform_cpp_scores(features: dict[str, np.ndarray], transform: CPPLabelTransform) -> np.ndarray:
    ams_z = _safe_zscore(features["ams"], transform.ams_mean, transform.ams_std)
    pams_z = _safe_zscore(features["pams"], transform.pams_mean, transform.pams_std)
    slps_z = _safe_zscore(features["slps"], transform.slps_mean, transform.slps_std)
    return (ams_z + pams_z + slps_z).astype(np.float32)


def threshold_cpp_scores(scores: np.ndarray, transform: CPPLabelTransform) -> tuple[np.ndarray, np.ndarray]:
    labels = np.full(scores.shape[0], fill_value=-1, dtype=np.int64)
    labels[scores <= transform.lower_threshold] = 0
    labels[scores >= transform.upper_threshold] = 1
    keep_mask = labels >= 0
    return labels, keep_mask
