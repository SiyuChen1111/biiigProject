from __future__ import annotations

import torch
import torch.nn.functional as F

from src.features.labels_cpp import CPP_ROI_CHANNELS


def _ensure_batch_time(epochs: torch.Tensor) -> torch.Tensor:
    if epochs.dim() == 4:
        if epochs.shape[-1] != 1:
            raise ValueError(f"Expected singleton trailing dim for 4D epochs, got shape {tuple(epochs.shape)}.")
        epochs = epochs.squeeze(-1)
    if epochs.dim() != 3:
        raise ValueError(f"Expected epochs with shape (batch, channels, time) or (batch, channels, time, 1), got {tuple(epochs.shape)}.")
    if epochs.shape[1] != len(CPP_ROI_CHANNELS):
        raise ValueError(
            f"CPP morphology losses expect ROI-only tensors with {len(CPP_ROI_CHANNELS)} channels for {CPP_ROI_CHANNELS}, got shape {tuple(epochs.shape)}."
        )
    return epochs


def _ensure_binary_labels(labels: torch.Tensor) -> torch.Tensor:
    if labels.dim() != 1:
        raise ValueError(f"Expected labels with shape (batch,), got {tuple(labels.shape)}.")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"Expected integer binary labels, got dtype {labels.dtype}.")
    unique = torch.unique(labels)
    allowed = torch.tensor([0, 1], device=labels.device, dtype=labels.dtype)
    if not torch.isin(unique, allowed).all():
        raise ValueError(f"CPP morphology losses expect binary labels in {{0, 1}}, got values {unique.tolist()}.")
    return labels


def _window_mask(times: torch.Tensor, window: tuple[float, float]) -> torch.Tensor:
    start, end = window
    mask = (times >= start) & (times <= end)
    if not bool(mask.any()):
        raise ValueError(f"No samples found in window [{start}, {end}].")
    return mask


def extract_roi_cpp_waveform(
    epochs: torch.Tensor,
    baseline_window: tuple[float, float],
    times: torch.Tensor,
) -> torch.Tensor:
    """Return baseline-corrected ROI-averaged CPP waveforms in native training units.

    Args:
        epochs: (batch, channels=3, time) or (batch, channels=3, time, 1)
        baseline_window: time window in seconds for baseline correction
        times: (time,) tensor in seconds aligned to epoch samples
    Returns:
        (batch, time) tensor in the same units as the training input
    """

    epochs_3d = _ensure_batch_time(epochs)
    if times.dim() != 1:
        raise ValueError(f"Expected times with shape (time,), got {tuple(times.shape)}.")
    if epochs_3d.shape[-1] != times.shape[0]:
        raise ValueError(
            f"Epoch sample axis ({epochs_3d.shape[-1]}) does not match times length ({times.shape[0]})."
        )

    roi_mean = epochs_3d.mean(dim=1)
    mask = _window_mask(times, baseline_window)
    baseline = roi_mean[:, mask].mean(dim=1, keepdim=True)
    return roi_mean - baseline


def compute_condition_average(
    waveforms: torch.Tensor,
    labels: torch.Tensor,
    target_label: int,
) -> torch.Tensor | None:
    if waveforms.dim() != 2:
        raise ValueError(f"Expected waveforms with shape (batch, time), got {tuple(waveforms.shape)}.")
    labels = _ensure_binary_labels(labels)
    mask = labels == target_label
    if not bool(mask.any()):
        return None
    return waveforms[mask].mean(dim=0)


def cpp_waveform_loss(real_waveforms: torch.Tensor, recon_waveforms: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(recon_waveforms, real_waveforms, reduction="mean")


def cpp_peak_loss(
    real_waveforms: torch.Tensor,
    recon_waveforms: torch.Tensor,
    times: torch.Tensor,
    peak_window: tuple[float, float],
) -> torch.Tensor:
    mask = _window_mask(times, peak_window)
    real_peak = real_waveforms[:, mask].amax(dim=1)
    recon_peak = recon_waveforms[:, mask].amax(dim=1)
    return F.mse_loss(recon_peak, real_peak, reduction="mean")


def cpp_difference_wave_loss(
    real_waveforms: torch.Tensor,
    recon_waveforms: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    real_low = compute_condition_average(real_waveforms, labels, target_label=0)
    real_high = compute_condition_average(real_waveforms, labels, target_label=1)
    recon_low = compute_condition_average(recon_waveforms, labels, target_label=0)
    recon_high = compute_condition_average(recon_waveforms, labels, target_label=1)

    if real_low is None or real_high is None or recon_low is None or recon_high is None:
        raise ValueError("CPP difference-wave loss requires both binary classes to be present in the provided labels.")

    real_diff = real_high - real_low
    recon_diff = recon_high - recon_low
    return F.mse_loss(recon_diff, real_diff, reduction="mean")
