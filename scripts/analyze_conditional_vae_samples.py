from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.data_kosciessa import DATASET_DIR, build_subject_trial_table, filter_subject_ids_with_paths
from src.features.labels_cpp import CPP_ROI_CHANNELS, compute_cpp_features, fit_cpp_label_transform, threshold_cpp_scores, transform_cpp_scores
from src.preprocessing.epoching import extract_response_locked_epochs


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze conditional EEG VAE generated samples against real ROI ERPs.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--vae-output-dir", type=Path, required=True)
    parser.add_argument("--analysis-output-dir", type=Path, default=Path("outputs/phase1_conditional_vae_analysis"))
    parser.add_argument("--max-subjects", type=int, default=5)
    return parser.parse_args()


def _build_fold_assignment(subject_ids: list[str], n_folds: int = 5) -> list[list[str]]:
    folds = [[] for _ in range(n_folds)]
    for idx, subject_id in enumerate(sorted(subject_ids)):
        folds[idx % n_folds].append(subject_id)
    return folds


def _split_train_val_subjects(subject_ids: list[str]) -> tuple[list[str], list[str]]:
    sorted_ids = sorted(subject_ids)
    n_val = max(1, round(0.2 * len(sorted_ids)))
    return sorted_ids[n_val:], sorted_ids[:n_val]


def _load_subject_dataset(subject_id: str, dataset_dir: Path) -> SubjectDataset:
    trial_table = build_subject_trial_table(subject_id=subject_id, dataset_dir=dataset_dir)
    epoched = extract_response_locked_epochs(subject_id=subject_id, trial_table=trial_table, dataset_dir=dataset_dir)
    features = compute_cpp_features(epoched.epochs, epoched.times, epoched.channel_names)
    return SubjectDataset(
        subject_id=subject_id,
        epochs=epoched.epochs,
        times=epoched.times,
        channel_names=epoched.channel_names,
        trial_table=epoched.trial_table,
        features=features,
    )


def _concatenate_subject_blocks(blocks: list[SubjectDataset]) -> SubjectBlock:
    return SubjectBlock(
        epochs=np.concatenate([block.epochs for block in blocks], axis=0),
        trial_table=pd.concat([block.trial_table for block in blocks], ignore_index=True),
        features={key: np.concatenate([block.features[key] for block in blocks], axis=0) for key in ["ams", "pams", "slps"]},
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


def _roi_average(epochs: np.ndarray, channel_names: list[str], roi_channels: list[str]) -> np.ndarray:
    name_to_idx = {name: idx for idx, name in enumerate(channel_names)}
    roi_idx = [name_to_idx[name] for name in roi_channels]
    return epochs[:, roi_idx, :].mean(axis=1)


def _plot_real_vs_generated(
    real_times: np.ndarray,
    gen_times: np.ndarray,
    real_pos: np.ndarray,
    real_neg: np.ndarray,
    gen_pos: np.ndarray,
    gen_neg: np.ndarray,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(real_times, real_pos, label="Real high CPP-like", linewidth=2)
    plt.plot(real_times, real_neg, label="Real low CPP-like", linewidth=2)
    plt.plot(gen_times, gen_pos, label="Generated high CPP-like", linestyle="--")
    plt.plot(gen_times, gen_neg, label="Generated low CPP-like", linestyle="--")
    plt.axvline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.title("Conditional VAE: Real vs Generated ROI ERP")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def _average_generated_by_label(samples: np.ndarray, samples_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    gen_neg = samples[:samples_per_class].mean(axis=0)
    gen_pos = samples[samples_per_class : 2 * samples_per_class].mean(axis=0)
    return gen_neg, gen_pos


def main() -> None:
    args = parse_args()
    subject_ids = filter_subject_ids_with_paths(dataset_dir=args.dataset_dir)[: args.max_subjects]
    subject_blocks = {subject_id: _load_subject_dataset(subject_id, args.dataset_dir) for subject_id in subject_ids}
    fold_assignments = _build_fold_assignment(subject_ids, n_folds=5)

    fold_summaries: list[dict[str, float | int]] = []
    output_dir = args.analysis_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, test_subjects in enumerate(fold_assignments, start=1):
        remaining_subjects = [subject_id for subject_id in subject_ids if subject_id not in test_subjects]
        train_subjects, _ = _split_train_val_subjects(remaining_subjects)
        train_block = _concatenate_subject_blocks([subject_blocks[s] for s in train_subjects])
        test_block = _concatenate_subject_blocks([subject_blocks[s] for s in test_subjects])
        transform = fit_cpp_label_transform(train_block.features)
        test_ready = _apply_label_transform(test_block, transform)

        sample_path = args.vae_output_dir / f"fold_{fold_idx}" / "conditional_samples.npy"
        if not sample_path.exists():
            continue

        generated = np.load(sample_path)
        if generated.shape[0] < 4:
            continue

        real_roi = _roi_average(test_ready.epochs, test_ready.channel_names, CPP_ROI_CHANNELS)
        gen_roi = _roi_average(generated.squeeze(-1), test_ready.channel_names, CPP_ROI_CHANNELS)
        gen_times = np.linspace(float(test_ready.times[0]), float(test_ready.times[-1]), gen_roi.shape[-1], endpoint=True)

        real_pos = real_roi[test_ready.labels == 1].mean(axis=0)
        real_neg = real_roi[test_ready.labels == 0].mean(axis=0)
        samples_per_class = generated.shape[0] // 2
        gen_neg, gen_pos = _average_generated_by_label(gen_roi, samples_per_class)

        real_gap = float(np.mean(real_pos - real_neg))
        gen_gap = float(np.mean(gen_pos - gen_neg))
        gap_abs_error = abs(real_gap - gen_gap)
        fold_summaries.append(
            {
                "fold": fold_idx,
                "real_gap_mean": real_gap,
                "generated_gap_mean": gen_gap,
                "gap_abs_error": float(gap_abs_error),
            }
        )

        _plot_real_vs_generated(
            real_times=test_ready.times,
            gen_times=gen_times,
            real_pos=real_pos,
            real_neg=real_neg,
            gen_pos=gen_pos,
            gen_neg=gen_neg,
            output_path=output_dir / f"fold_{fold_idx}" / "real_vs_generated_roi_erp.png",
        )

    with open(output_dir / "fold_generation_summary.json", "w", encoding="utf-8") as f:
        json.dump(fold_summaries, f, indent=2)


if __name__ == "__main__":
    main()
