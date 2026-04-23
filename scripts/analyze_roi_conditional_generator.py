from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.data_kosciessa import DATASET_DIR, build_subject_trial_table, filter_subject_ids_with_paths
from src.features.labels_cpp import (
    CPP_ROI_CHANNELS,
    PAMS_WINDOW,
    compute_cpp_features,
    fit_cpp_label_transform,
    threshold_cpp_scores,
    transform_cpp_scores,
)
from src.preprocessing.epoching import PHASE1_BASELINE, extract_response_locked_epochs, to_roi_epoched_subject_data


@dataclass(frozen=True)
class SubjectDataset:
    subject_id: str
    epochs: np.ndarray
    times: np.ndarray
    channel_names: list[str]
    trial_table: pd.DataFrame
    features: dict[str, np.ndarray]


@dataclass(frozen=True)
class SubjectBlock:
    epochs: np.ndarray
    trial_table: pd.DataFrame
    features: dict[str, np.ndarray]
    times: np.ndarray
    channel_names: list[str]


@dataclass(frozen=True)
class LabeledBlock:
    epochs: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    trial_table: pd.DataFrame
    times: np.ndarray
    channel_names: list[str]


@dataclass(frozen=True)
class ConditionComparison:
    label_name: str
    real_waveform: np.ndarray
    generated_waveform: np.ndarray
    waveform_correlation: float
    peak_amplitude_real_uv: float
    peak_amplitude_generated_uv: float
    peak_amplitude_error_uv: float
    peak_latency_real_seconds: float
    peak_latency_generated_seconds: float
    peak_latency_error_seconds: float
    baseline_mean_real_uv: float
    baseline_mean_generated_uv: float
    baseline_mean_abs_error_uv: float
    distribution_distance: float
    real_trial_count: int
    generated_trial_count: int


@dataclass(frozen=True)
class FoldAnalysis:
    fold: int
    test_subject_ids: list[str]
    times: np.ndarray
    generated_times: np.ndarray
    positive: ConditionComparison
    negative: ConditionComparison

    @property
    def all_waveforms(self) -> list[np.ndarray]:
        return [
            self.positive.real_waveform,
            self.positive.generated_waveform,
            self.negative.real_waveform,
            self.negative.generated_waveform,
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze ROI-only conditional generator outputs against held-out real ROI epochs."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/roi_conditional_generator_smoke"),
        help=(
            "ROI generator contract root containing train/ and analysis/ directories. "
            "Defaults to outputs/roi_conditional_generator_smoke."
        ),
    )
    parser.add_argument(
        "--train-output-dir",
        type=Path,
        default=None,
        help="Optional override for the training artifacts directory. Defaults to <output-root>/train.",
    )
    parser.add_argument(
        "--analysis-output-dir",
        type=Path,
        default=None,
        help="Optional override for the analysis artifacts directory. Defaults to <output-root>/analysis.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Dataset root. If omitted, the script reuses train/train_run_config.json when available.",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Optional subject cap. If omitted, the script reuses train/train_run_config.json when available.",
    )
    return parser.parse_args()


def _resolve_dirs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    output_root = Path(args.output_root)
    train_output_dir = Path(args.train_output_dir) if args.train_output_dir is not None else output_root / "train"
    analysis_output_dir = (
        Path(args.analysis_output_dir) if args.analysis_output_dir is not None else output_root / "analysis"
    )
    return output_root, train_output_dir, analysis_output_dir


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_runtime_args(args: argparse.Namespace, train_output_dir: Path) -> tuple[Path, int | None]:
    train_config_path = train_output_dir / "train_run_config.json"
    train_config = _load_json(train_config_path) if train_config_path.exists() else {"args": {}}
    config_args = train_config.get("args", {})

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir is not None else Path(config_args.get("dataset_dir", DATASET_DIR))
    max_subjects = args.max_subjects if args.max_subjects is not None else config_args.get("max_subjects")
    return dataset_dir, max_subjects


def _write_run_manifest(root: Path) -> None:
    manifest_path = root / "run_manifest.json"
    if manifest_path.exists():
        return
    manifest = {
        "contract_id": "roi_conditional_generator_v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "train_dir": "train",
        "analysis_dir": "analysis",
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _build_fold_assignment(subject_ids: list[str], n_folds: int = 5) -> list[list[str]]:
    folds = [[] for _ in range(n_folds)]
    for idx, subject_id in enumerate(sorted(subject_ids)):
        folds[idx % n_folds].append(subject_id)
    return folds


def _split_train_val_subjects(subject_ids: list[str]) -> tuple[list[str], list[str]]:
    sorted_ids = sorted(subject_ids)
    n_val = max(1, round(0.2 * len(sorted_ids)))
    val_ids = sorted_ids[:n_val]
    train_ids = sorted_ids[n_val:]
    if not train_ids:
        raise ValueError("Validation split consumed all training subjects.")
    return train_ids, val_ids


def _load_subject_dataset(subject_id: str, dataset_dir: Path) -> SubjectDataset:
    trial_table = build_subject_trial_table(subject_id=subject_id, dataset_dir=dataset_dir)
    epoched = extract_response_locked_epochs(subject_id=subject_id, trial_table=trial_table, dataset_dir=dataset_dir)
    roi_epoched = to_roi_epoched_subject_data(epoched)
    features = compute_cpp_features(
        epochs=roi_epoched.epochs,
        times=roi_epoched.times,
        channel_names=roi_epoched.channel_names,
        roi_channels=CPP_ROI_CHANNELS,
    )
    return SubjectDataset(
        subject_id=subject_id,
        epochs=roi_epoched.epochs,
        times=roi_epoched.times,
        channel_names=roi_epoched.channel_names,
        trial_table=roi_epoched.trial_table,
        features=features,
    )


def _concatenate_subject_blocks(blocks: list[SubjectDataset]) -> SubjectBlock:
    return SubjectBlock(
        epochs=np.concatenate([block.epochs for block in blocks], axis=0),
        trial_table=pd.concat([block.trial_table for block in blocks], ignore_index=True),
        features={
            key: np.concatenate([block.features[key] for block in blocks], axis=0)
            for key in ["ams", "pams", "slps"]
        },
        times=blocks[0].times,
        channel_names=blocks[0].channel_names,
    )


def _apply_label_transform(block: SubjectBlock, transform) -> LabeledBlock:
    scores = transform_cpp_scores(block.features, transform)
    labels, keep_mask = threshold_cpp_scores(scores, transform)
    keep_mask = keep_mask.astype(bool)
    return LabeledBlock(
        epochs=block.epochs[keep_mask],
        labels=labels[keep_mask],
        scores=scores[keep_mask],
        trial_table=block.trial_table.iloc[keep_mask].reset_index(drop=True),
        times=block.times,
        channel_names=block.channel_names,
    )


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x_centered, y_centered) / denom)


def _window_mask(times: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    mask = (times >= window[0]) & (times <= window[1])
    if not mask.any():
        raise ValueError(f"No samples found in window [{window[0]}, {window[1]}].")
    return mask


def _peak_metrics(waveform: np.ndarray, times: np.ndarray, window: tuple[float, float]) -> tuple[float, float]:
    mask = _window_mask(times, window)
    window_waveform = waveform[mask]
    window_times = times[mask]
    peak_idx = int(np.argmax(window_waveform))
    return float(window_waveform[peak_idx]), float(window_times[peak_idx])


def _baseline_mean(waveform: np.ndarray, times: np.ndarray) -> float:
    mask = _window_mask(times, PHASE1_BASELINE)
    return float(np.mean(waveform[mask]))


def _quantile_wasserstein_distance(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        raise ValueError("Distribution comparison requires non-empty arrays.")
    n_quantiles = max(128, x.size, y.size)
    quantiles = np.linspace(0.0, 1.0, n_quantiles)
    x_quantiles = np.quantile(x, quantiles)
    y_quantiles = np.quantile(y, quantiles)
    return float(np.mean(np.abs(x_quantiles - y_quantiles)))


def _condition_name(label: int) -> str:
    return "positive" if label == 1 else "negative"


def _mean_roi_waveform(epochs: np.ndarray) -> np.ndarray:
    return np.mean(epochs, axis=1)


def _prepare_generated_samples(sample_path: Path, reference_times: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    generated = np.load(sample_path)
    if generated.ndim != 4:
        raise ValueError(f"Expected generated samples with 4 dims (samples, channels, times, 1); got {generated.shape}.")
    if generated.shape[1] != len(CPP_ROI_CHANNELS):
        raise ValueError(
            f"Expected generated ROI samples to have {len(CPP_ROI_CHANNELS)} channels; got {generated.shape[1]}."
        )
    if generated.shape[-1] != 1:
        raise ValueError(f"Expected generated samples trailing singleton dimension; got {generated.shape}.")
    if generated.shape[0] % 2 != 0:
        raise ValueError(f"Expected equal class counts in generated samples; got {generated.shape[0]} samples.")

    generated_epochs = generated.squeeze(-1).astype(np.float32, copy=False)
    generated_times = np.linspace(float(reference_times[0]), float(reference_times[-1]), generated_epochs.shape[-1], endpoint=True)
    samples_per_class = generated_epochs.shape[0] // 2
    return generated_epochs, generated_times, samples_per_class


def _generated_condition_block(
    generated_epochs: np.ndarray,
    generated_scores: np.ndarray,
    label: int,
    samples_per_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    if label == 0:
        start, stop = 0, samples_per_class
    else:
        start, stop = samples_per_class, 2 * samples_per_class
    return generated_epochs[start:stop], generated_scores[start:stop]


def _real_condition_block(real_epochs: np.ndarray, real_scores: np.ndarray, labels: np.ndarray, label: int) -> tuple[np.ndarray, np.ndarray]:
    mask = labels == label
    if not mask.any():
        raise ValueError(f"No held-out real trials found for label {label}.")
    return real_epochs[mask], real_scores[mask]


def _compare_condition(
    label: int,
    real_epochs: np.ndarray,
    real_scores: np.ndarray,
    real_labels: np.ndarray,
    generated_epochs: np.ndarray,
    generated_scores: np.ndarray,
    samples_per_class: int,
    times: np.ndarray,
    generated_times: np.ndarray,
) -> ConditionComparison:
    real_condition_epochs, real_condition_scores = _real_condition_block(real_epochs, real_scores, real_labels, label)
    generated_condition_epochs, generated_condition_scores = _generated_condition_block(
        generated_epochs=generated_epochs,
        generated_scores=generated_scores,
        label=label,
        samples_per_class=samples_per_class,
    )

    real_waveform = _mean_roi_waveform(real_condition_epochs).mean(axis=0)
    generated_waveform = _mean_roi_waveform(generated_condition_epochs).mean(axis=0)

    real_peak_amplitude, real_peak_latency = _peak_metrics(real_waveform, times, PAMS_WINDOW)
    generated_peak_amplitude, generated_peak_latency = _peak_metrics(generated_waveform, generated_times, PAMS_WINDOW)
    real_baseline = _baseline_mean(real_waveform, times)
    generated_baseline = _baseline_mean(generated_waveform, generated_times)

    return ConditionComparison(
        label_name=_condition_name(label),
        real_waveform=real_waveform,
        generated_waveform=generated_waveform,
        waveform_correlation=_pearson_correlation(real_waveform, generated_waveform),
        peak_amplitude_real_uv=real_peak_amplitude,
        peak_amplitude_generated_uv=generated_peak_amplitude,
        peak_amplitude_error_uv=abs(real_peak_amplitude - generated_peak_amplitude),
        peak_latency_real_seconds=real_peak_latency,
        peak_latency_generated_seconds=generated_peak_latency,
        peak_latency_error_seconds=abs(real_peak_latency - generated_peak_latency),
        baseline_mean_real_uv=real_baseline,
        baseline_mean_generated_uv=generated_baseline,
        baseline_mean_abs_error_uv=abs(real_baseline - generated_baseline),
        distribution_distance=_quantile_wasserstein_distance(real_condition_scores, generated_condition_scores),
        real_trial_count=int(real_condition_epochs.shape[0]),
        generated_trial_count=int(generated_condition_epochs.shape[0]),
    )


def _flatten_condition(prefix: str, comparison: ConditionComparison) -> dict[str, float | int | str]:
    return {
        f"{prefix}_label_name": comparison.label_name,
        f"{prefix}_waveform_correlation": comparison.waveform_correlation,
        f"{prefix}_peak_amplitude_real_uv": comparison.peak_amplitude_real_uv,
        f"{prefix}_peak_amplitude_generated_uv": comparison.peak_amplitude_generated_uv,
        f"{prefix}_peak_amplitude_error_uv": comparison.peak_amplitude_error_uv,
        f"{prefix}_peak_latency_real_seconds": comparison.peak_latency_real_seconds,
        f"{prefix}_peak_latency_generated_seconds": comparison.peak_latency_generated_seconds,
        f"{prefix}_peak_latency_error_seconds": comparison.peak_latency_error_seconds,
        f"{prefix}_baseline_mean_real_uv": comparison.baseline_mean_real_uv,
        f"{prefix}_baseline_mean_generated_uv": comparison.baseline_mean_generated_uv,
        f"{prefix}_baseline_mean_abs_error_uv": comparison.baseline_mean_abs_error_uv,
        f"{prefix}_distribution_distance": comparison.distribution_distance,
        f"{prefix}_real_trial_count": comparison.real_trial_count,
        f"{prefix}_generated_trial_count": comparison.generated_trial_count,
    }


def _fold_summary_row(fold_analysis: FoldAnalysis) -> dict[str, float | int | str]:
    return {
        "fold": fold_analysis.fold,
        "test_subject_ids": ",".join(fold_analysis.test_subject_ids),
        **_flatten_condition("positive", fold_analysis.positive),
        **_flatten_condition("negative", fold_analysis.negative),
        "waveform_correlation_mean": float(
            np.mean([
                fold_analysis.positive.waveform_correlation,
                fold_analysis.negative.waveform_correlation,
            ])
        ),
        "peak_amplitude_error_mean_uv": float(
            np.mean([
                fold_analysis.positive.peak_amplitude_error_uv,
                fold_analysis.negative.peak_amplitude_error_uv,
            ])
        ),
        "peak_latency_error_mean_seconds": float(
            np.mean([
                fold_analysis.positive.peak_latency_error_seconds,
                fold_analysis.negative.peak_latency_error_seconds,
            ])
        ),
        "baseline_mean_abs_error_mean_uv": float(
            np.mean([
                fold_analysis.positive.baseline_mean_abs_error_uv,
                fold_analysis.negative.baseline_mean_abs_error_uv,
            ])
        ),
        "distribution_distance_mean": float(
            np.mean([
                fold_analysis.positive.distribution_distance,
                fold_analysis.negative.distribution_distance,
            ])
        ),
    }


def _collect_global_ylim(fold_analyses: list[FoldAnalysis]) -> tuple[float, float]:
    all_values = np.concatenate([waveform for fold in fold_analyses for waveform in fold.all_waveforms])
    y_min = float(all_values.min())
    y_max = float(all_values.max())
    margin = 0.05 * max(1e-6, y_max - y_min)
    return y_min - margin, y_max + margin


def _plot_fold_overlay(fold_analysis: FoldAnalysis, output_path: Path, y_limits: tuple[float, float]) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(fold_analysis.times, fold_analysis.positive.real_waveform, label="Real positive", linewidth=2)
    plt.plot(fold_analysis.times, fold_analysis.negative.real_waveform, label="Real negative", linewidth=2)
    plt.plot(
        fold_analysis.generated_times,
        fold_analysis.positive.generated_waveform,
        label="Generated positive",
        linestyle="--",
        linewidth=2,
    )
    plt.plot(
        fold_analysis.generated_times,
        fold_analysis.negative.generated_waveform,
        label="Generated negative",
        linestyle="--",
        linewidth=2,
    )
    plt.axvline(0.0, color="k", linestyle="--", linewidth=1)
    plt.ylim(*y_limits)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.title(f"Fold {fold_analysis.fold}: ROI real vs generated ERP")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def _pooled_condition_comparison(fold_analyses: list[FoldAnalysis], label: str) -> ConditionComparison:
    comparisons = [fold.positive if label == "positive" else fold.negative for fold in fold_analyses]
    pooled_real_waveform = np.mean(np.stack([item.real_waveform for item in comparisons], axis=0), axis=0)
    pooled_generated_waveform = np.mean(np.stack([item.generated_waveform for item in comparisons], axis=0), axis=0)
    pooled_times = fold_analyses[0].times
    pooled_generated_times = fold_analyses[0].generated_times
    real_peak_amplitude, real_peak_latency = _peak_metrics(pooled_real_waveform, pooled_times, PAMS_WINDOW)
    generated_peak_amplitude, generated_peak_latency = _peak_metrics(
        pooled_generated_waveform,
        pooled_generated_times,
        PAMS_WINDOW,
    )
    real_baseline = _baseline_mean(pooled_real_waveform, pooled_times)
    generated_baseline = _baseline_mean(pooled_generated_waveform, pooled_generated_times)
    return ConditionComparison(
        label_name=label,
        real_waveform=pooled_real_waveform,
        generated_waveform=pooled_generated_waveform,
        waveform_correlation=_pearson_correlation(pooled_real_waveform, pooled_generated_waveform),
        peak_amplitude_real_uv=real_peak_amplitude,
        peak_amplitude_generated_uv=generated_peak_amplitude,
        peak_amplitude_error_uv=abs(real_peak_amplitude - generated_peak_amplitude),
        peak_latency_real_seconds=real_peak_latency,
        peak_latency_generated_seconds=generated_peak_latency,
        peak_latency_error_seconds=abs(real_peak_latency - generated_peak_latency),
        baseline_mean_real_uv=real_baseline,
        baseline_mean_generated_uv=generated_baseline,
        baseline_mean_abs_error_uv=abs(real_baseline - generated_baseline),
        distribution_distance=float(np.mean([item.distribution_distance for item in comparisons])),
        real_trial_count=int(sum(item.real_trial_count for item in comparisons)),
        generated_trial_count=int(sum(item.generated_trial_count for item in comparisons)),
    )


def _aggregate_summary(fold_analyses: list[FoldAnalysis], fold_rows: list[dict[str, float | int | str]]) -> dict[str, float | int | str | dict]:
    positive = _pooled_condition_comparison(fold_analyses, label="positive")
    negative = _pooled_condition_comparison(fold_analyses, label="negative")

    numeric_columns = [
        "waveform_correlation_mean",
        "peak_amplitude_error_mean_uv",
        "peak_latency_error_mean_seconds",
        "baseline_mean_abs_error_mean_uv",
        "distribution_distance_mean",
    ]
    fold_df = pd.DataFrame(fold_rows)
    per_fold_means = {column: float(fold_df[column].mean()) for column in numeric_columns}
    per_fold_stds = {column: float(fold_df[column].std(ddof=0)) for column in numeric_columns}
    return {
        "n_folds_analyzed": len(fold_analyses),
        "held_out_subject_count": int(sum(len(fold.test_subject_ids) for fold in fold_analyses)),
        "per_fold_metric_means": per_fold_means,
        "per_fold_metric_stds": per_fold_stds,
        "positive": _flatten_condition("positive", positive),
        "negative": _flatten_condition("negative", negative),
        "waveform_correlation_mean": float(np.mean([positive.waveform_correlation, negative.waveform_correlation])),
        "peak_amplitude_error_mean_uv": float(
            np.mean([positive.peak_amplitude_error_uv, negative.peak_amplitude_error_uv])
        ),
        "peak_latency_error_mean_seconds": float(
            np.mean([positive.peak_latency_error_seconds, negative.peak_latency_error_seconds])
        ),
        "baseline_mean_abs_error_mean_uv": float(
            np.mean([positive.baseline_mean_abs_error_uv, negative.baseline_mean_abs_error_uv])
        ),
        "distribution_distance_mean": float(np.mean([positive.distribution_distance, negative.distribution_distance])),
    }


def _aggregate_summary_row(summary: dict[str, float | int | str | dict]) -> dict[str, float | int | str]:
    n_folds_analyzed = summary["n_folds_analyzed"]
    held_out_subject_count = summary["held_out_subject_count"]
    waveform_correlation_mean = summary["waveform_correlation_mean"]
    peak_amplitude_error_mean_uv = summary["peak_amplitude_error_mean_uv"]
    peak_latency_error_mean_seconds = summary["peak_latency_error_mean_seconds"]
    baseline_mean_abs_error_mean_uv = summary["baseline_mean_abs_error_mean_uv"]
    distribution_distance_mean = summary["distribution_distance_mean"]
    row: dict[str, float | int | str] = {
        "n_folds_analyzed": int(n_folds_analyzed) if isinstance(n_folds_analyzed, (int, float)) else 0,
        "held_out_subject_count": int(held_out_subject_count) if isinstance(held_out_subject_count, (int, float)) else 0,
        "waveform_correlation_mean": float(waveform_correlation_mean)
        if isinstance(waveform_correlation_mean, (int, float))
        else float("nan"),
        "peak_amplitude_error_mean_uv": float(peak_amplitude_error_mean_uv)
        if isinstance(peak_amplitude_error_mean_uv, (int, float))
        else float("nan"),
        "peak_latency_error_mean_seconds": float(peak_latency_error_mean_seconds)
        if isinstance(peak_latency_error_mean_seconds, (int, float))
        else float("nan"),
        "baseline_mean_abs_error_mean_uv": float(baseline_mean_abs_error_mean_uv)
        if isinstance(baseline_mean_abs_error_mean_uv, (int, float))
        else float("nan"),
        "distribution_distance_mean": float(distribution_distance_mean)
        if isinstance(distribution_distance_mean, (int, float))
        else float("nan"),
    }
    positive = summary["positive"]
    negative = summary["negative"]
    if isinstance(positive, dict):
        row.update({key: value for key, value in positive.items() if not key.endswith("label_name")})
    if isinstance(negative, dict):
        row.update({key: value for key, value in negative.items() if not key.endswith("label_name")})
    return row


def main() -> None:
    args = parse_args()
    output_root, train_output_dir, analysis_output_dir = _resolve_dirs(args)
    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_run_manifest(output_root)

    dataset_dir, max_subjects = _resolve_runtime_args(args, train_output_dir)

    run_config = {
        "argv": sys.argv,
        "args": {
            "output_root": str(output_root),
            "train_output_dir": str(train_output_dir),
            "analysis_output_dir": str(analysis_output_dir),
            "dataset_dir": str(dataset_dir),
            "max_subjects": max_subjects,
        },
    }
    with open(analysis_output_dir / "analysis_run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    train_config = _load_json(train_output_dir / "train_run_config.json") if (train_output_dir / "train_run_config.json").exists() else {"args": {}}
    samples_per_class_from_train = train_config.get("args", {}).get("samples_per_class")

    subject_ids = filter_subject_ids_with_paths(dataset_dir=dataset_dir)
    if max_subjects is not None:
        subject_ids = subject_ids[:max_subjects]
    if len(subject_ids) < 5:
        raise ValueError("Need at least 5 subjects for grouped 5-fold ROI analysis.")

    subject_blocks = {
        subject_id: _load_subject_dataset(subject_id=subject_id, dataset_dir=dataset_dir) for subject_id in subject_ids
    }
    fold_assignments = _build_fold_assignment(subject_ids=subject_ids, n_folds=5)

    fold_analyses: list[FoldAnalysis] = []
    fold_rows: list[dict[str, float | int | str]] = []

    for fold_idx, test_subjects in enumerate(fold_assignments, start=1):
        sample_path = train_output_dir / f"fold_{fold_idx}" / "conditional_samples.npy"
        if not sample_path.exists():
            raise ValueError(f"Missing generated samples for fold {fold_idx}: {sample_path}")

        remaining_subjects = [subject_id for subject_id in subject_ids if subject_id not in test_subjects]
        train_subjects, _ = _split_train_val_subjects(remaining_subjects)

        train_block = _concatenate_subject_blocks([subject_blocks[s] for s in train_subjects])
        test_block = _concatenate_subject_blocks([subject_blocks[s] for s in test_subjects])
        transform = fit_cpp_label_transform(train_block.features)
        test_ready = _apply_label_transform(test_block, transform)

        generated_epochs, generated_times, inferred_samples_per_class = _prepare_generated_samples(
            sample_path=sample_path,
            reference_times=test_ready.times,
        )
        samples_per_class = (
            int(samples_per_class_from_train)
            if samples_per_class_from_train is not None
            else inferred_samples_per_class
        )
        if samples_per_class != inferred_samples_per_class:
            raise ValueError(
                f"Fold {fold_idx} sample count mismatch: train config says {samples_per_class}, "
                f"but {sample_path} implies {inferred_samples_per_class} samples per class."
            )

        generated_features = compute_cpp_features(
            epochs=generated_epochs,
            times=generated_times,
            channel_names=list(CPP_ROI_CHANNELS),
            roi_channels=CPP_ROI_CHANNELS,
        )
        generated_scores = transform_cpp_scores(generated_features, transform)

        positive = _compare_condition(
            label=1,
            real_epochs=test_ready.epochs,
            real_scores=test_ready.scores,
            real_labels=test_ready.labels,
            generated_epochs=generated_epochs,
            generated_scores=generated_scores,
            samples_per_class=samples_per_class,
            times=test_ready.times,
            generated_times=generated_times,
        )
        negative = _compare_condition(
            label=0,
            real_epochs=test_ready.epochs,
            real_scores=test_ready.scores,
            real_labels=test_ready.labels,
            generated_epochs=generated_epochs,
            generated_scores=generated_scores,
            samples_per_class=samples_per_class,
            times=test_ready.times,
            generated_times=generated_times,
        )

        fold_analysis = FoldAnalysis(
            fold=fold_idx,
            test_subject_ids=list(test_subjects),
            times=test_ready.times,
            generated_times=generated_times,
            positive=positive,
            negative=negative,
        )
        fold_analyses.append(fold_analysis)
        fold_rows.append(_fold_summary_row(fold_analysis))

    if not fold_analyses:
        raise ValueError("No fold analyses were produced.")

    y_limits = _collect_global_ylim(fold_analyses)
    for fold_analysis in fold_analyses:
        _plot_fold_overlay(
            fold_analysis=fold_analysis,
            output_path=analysis_output_dir / f"fold_{fold_analysis.fold}" / "real_vs_generated_roi_erp.png",
            y_limits=y_limits,
        )

    with open(analysis_output_dir / "fold_generation_summary.json", "w", encoding="utf-8") as f:
        json.dump(fold_rows, f, indent=2)
    pd.DataFrame(fold_rows).to_csv(analysis_output_dir / "fold_generation_summary.csv", index=False)

    aggregate_summary = _aggregate_summary(fold_analyses=fold_analyses, fold_rows=fold_rows)
    with open(analysis_output_dir / "aggregate_generation_summary.json", "w", encoding="utf-8") as f:
        json.dump(aggregate_summary, f, indent=2)
    pd.DataFrame([_aggregate_summary_row(aggregate_summary)]).to_csv(
        analysis_output_dir / "aggregate_generation_summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
