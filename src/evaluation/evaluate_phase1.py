from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, float | int | None]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    zero_division = cast(Any, 0)
    metrics: dict[str, float | int | None] = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=zero_division)),
        "recall": float(recall_score(y_true, y_pred, zero_division=zero_division)),
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics.update(
        {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        }
    )
    if y_prob is None or np.unique(y_true).size < 2:
        metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def summarize_fold_metrics(fold_metrics: list[dict[str, float | int | None]]) -> dict[str, float | int]:
    def _mean_std(values: list[float]) -> tuple[float, float]:
        arr = np.asarray(values, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=0))

    balanced = [float(item["balanced_accuracy"] or 0.0) for item in fold_metrics]
    precision = [float(item["precision"] or 0.0) for item in fold_metrics]
    recall = [float(item["recall"] or 0.0) for item in fold_metrics]
    roc_values = [float(item["roc_auc"] or 0.0) for item in fold_metrics if item["roc_auc"] is not None]

    summary: dict[str, float | int] = {}
    summary["balanced_accuracy_mean"], summary["balanced_accuracy_std"] = _mean_std(balanced)
    summary["precision_mean"], summary["precision_std"] = _mean_std(precision)
    summary["recall_mean"], summary["recall_std"] = _mean_std(recall)
    if roc_values:
        summary["roc_auc_mean"], summary["roc_auc_std"] = _mean_std(roc_values)
    else:
        summary["roc_auc_mean"] = float("nan")
        summary["roc_auc_std"] = float("nan")
    summary["roc_auc_defined_folds"] = len(roc_values)
    return summary


def plot_fold_metrics(fold_metrics: list[dict[str, float | int | None]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_ids = np.arange(1, len(fold_metrics) + 1)
    balanced = [float(item["balanced_accuracy"] or 0.0) for item in fold_metrics]
    aucs = [np.nan if item["roc_auc"] is None else float(item["roc_auc"]) for item in fold_metrics]

    plt.figure(figsize=(8, 4))
    plt.plot(fold_ids, balanced, marker="o", label="Balanced Accuracy")
    plt.plot(fold_ids, aucs, marker="s", label="ROC-AUC")
    plt.xlabel("Fold")
    plt.ylabel("Metric")
    plt.title("Phase-1 Fold Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fold_metrics.png", dpi=150)
    plt.close()


def plot_subject_balanced_accuracy(subject_metrics: dict[str, dict[str, float | int | None]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_ids = sorted(subject_metrics)
    values = [float(subject_metrics[subject_id]["balanced_accuracy"] or 0.0) for subject_id in subject_ids]

    plt.figure(figsize=(max(8, len(subject_ids) * 0.35), 4))
    plt.bar(subject_ids, values)
    plt.xticks(rotation=90)
    plt.ylabel("Balanced Accuracy")
    plt.title("Per-Subject Balanced Accuracy")
    plt.tight_layout()
    plt.savefig(output_dir / "subject_balanced_accuracy.png", dpi=150)
    plt.close()


def plot_roi_erp_comparison(
    epochs: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    channel_names: list[str],
    output_dir: Path,
    roi_channels: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    name_to_idx = {name: idx for idx, name in enumerate(channel_names)}
    roi_idx = [name_to_idx[name] for name in roi_channels if name in name_to_idx]
    if not roi_idx:
        return
    roi_signal = epochs[:, roi_idx, :].mean(axis=1)

    pos_mask = labels == 1
    neg_mask = labels == 0
    if not pos_mask.any() or not neg_mask.any():
        return

    pos_erp = roi_signal[pos_mask].mean(axis=0)
    neg_erp = roi_signal[neg_mask].mean(axis=0)

    plt.figure(figsize=(6, 4))
    plt.plot(times, pos_erp, label="Predicted positive")
    plt.plot(times, neg_erp, label="Predicted negative")
    plt.axvline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.title("ROI ERP Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roi_erp_comparison.png", dpi=150)
    plt.close()
