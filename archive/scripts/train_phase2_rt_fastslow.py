from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, TensorDataset

from src.data.data_kosciessa import DATASET_DIR, build_subject_trial_table, filter_subject_ids_with_paths
from src.models.EEGModels_PyTorch import EEGNet
from src.preprocessing.epoching import extract_response_locked_epochs
from src.evaluation.evaluate_phase1 import (
    compute_binary_metrics,
    plot_fold_metrics,
    plot_roi_erp_comparison,
    plot_subject_balanced_accuracy,
    summarize_fold_metrics,
)
from src.features.labels_cpp import CPP_ROI_CHANNELS


@dataclass(frozen=True)
class SubjectDataset:
    subject_id: str
    epochs: np.ndarray
    times: np.ndarray
    channel_names: list[str]
    trial_table: pd.DataFrame


@dataclass(frozen=True)
class LabeledSubjectBlock:
    epochs: np.ndarray
    labels: np.ndarray
    trial_table: pd.DataFrame
    times: np.ndarray
    channel_names: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train phase-2 EEGNet on fast-vs-slow RT labels.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase2_rt_fastslow"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


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


def _epochs_to_model_input(epochs: np.ndarray) -> np.ndarray:
    return epochs.reshape(epochs.shape[0], epochs.shape[1], epochs.shape[2], 1).astype(np.float32)


def _load_subject_dataset(subject_id: str, dataset_dir: Path) -> SubjectDataset:
    trial_table = build_subject_trial_table(subject_id=subject_id, dataset_dir=dataset_dir)
    epoched = extract_response_locked_epochs(
        subject_id=subject_id,
        trial_table=trial_table,
        dataset_dir=dataset_dir,
    )
    return SubjectDataset(
        subject_id=subject_id,
        epochs=epoched.epochs,
        times=epoched.times,
        channel_names=epoched.channel_names,
        trial_table=epoched.trial_table,
    )


def _concatenate_subject_blocks(blocks: list[SubjectDataset]) -> SubjectDataset:
    return SubjectDataset(
        subject_id="combined",
        epochs=np.concatenate([block.epochs for block in blocks], axis=0),
        times=blocks[0].times,
        channel_names=blocks[0].channel_names,
        trial_table=pd.concat([block.trial_table for block in blocks], ignore_index=True),
    )


def _fit_subject_rt_medians(trial_table: pd.DataFrame) -> dict[str, float]:
    medians: dict[str, float] = {}
    for subject_id, group in trial_table.groupby("subject_id"):
        medians[str(subject_id)] = float(group["probe_rt"].median())
    return medians


def _apply_rt_labels(block: SubjectDataset, subject_medians: dict[str, float]) -> LabeledSubjectBlock:
    labels = np.full(len(block.trial_table), fill_value=-1, dtype=np.int64)
    for idx, row in block.trial_table.iterrows():
        subject_id = str(row["subject_id"])
        rt_value = float(row["probe_rt"])
        median_rt = subject_medians[subject_id]
        labels[idx] = 0 if rt_value <= median_rt else 1

    return LabeledSubjectBlock(
        epochs=block.epochs,
        labels=labels,
        trial_table=block.trial_table.reset_index(drop=True),
        times=block.times,
        channel_names=block.channel_names,
    )


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.FloatTensor(x), torch.LongTensor(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _evaluate_loader(model: EEGNet, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    y_prob_all: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            y_true_all.append(batch_y.cpu().numpy())
            y_pred_all.append(preds.cpu().numpy())
            y_prob_all.append(probs.cpu().numpy())
    return np.concatenate(y_true_all), np.concatenate(y_pred_all), np.concatenate(y_prob_all)


def _subject_metrics(trial_table: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, dict[str, float | int | None]]:
    output: dict[str, dict[str, float | int | None]] = {}
    for subject_id, group in trial_table.groupby("subject_id"):
        idx = group.index.to_numpy()
        output[str(subject_id)] = compute_binary_metrics(y_true=y_true[idx], y_pred=y_pred[idx], y_prob=y_prob[idx])
    return output


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    subject_ids = filter_subject_ids_with_paths(dataset_dir=args.dataset_dir)
    if args.max_subjects is not None:
        subject_ids = subject_ids[: args.max_subjects]
    if len(subject_ids) < 5:
        raise ValueError("Need at least 5 subjects for grouped 5-fold CV.")

    subject_blocks: dict[str, SubjectDataset] = {
        subject_id: _load_subject_dataset(subject_id=subject_id, dataset_dir=args.dataset_dir)
        for subject_id in subject_ids
    }

    fold_assignments = _build_fold_assignment(subject_ids=subject_ids, n_folds=5)
    fold_metrics: list[dict[str, float | int | None]] = []
    combined_subject_metrics: dict[str, dict[str, float | int | None]] = {}

    for fold_idx, test_subjects in enumerate(fold_assignments, start=1):
        remaining_subjects = [subject_id for subject_id in subject_ids if subject_id not in test_subjects]
        train_subjects, val_subjects = _split_train_val_subjects(remaining_subjects)

        train_block = _concatenate_subject_blocks([subject_blocks[s] for s in train_subjects])
        val_block = _concatenate_subject_blocks([subject_blocks[s] for s in val_subjects])
        test_block = _concatenate_subject_blocks([subject_blocks[s] for s in test_subjects])

        training_subject_medians = _fit_subject_rt_medians(train_block.trial_table)
        train_ready = _apply_rt_labels(train_block, training_subject_medians)

        val_subject_medians = {subject_id: training_subject_medians.get(subject_id, float(subject_blocks[subject_id].trial_table["probe_rt"].median())) for subject_id in val_subjects}
        test_subject_medians = {subject_id: training_subject_medians.get(subject_id, float(subject_blocks[subject_id].trial_table["probe_rt"].median())) for subject_id in test_subjects}
        val_ready = _apply_rt_labels(val_block, val_subject_medians)
        test_ready = _apply_rt_labels(test_block, test_subject_medians)

        x_train = _epochs_to_model_input(train_ready.epochs)
        x_val = _epochs_to_model_input(val_ready.epochs)
        x_test = _epochs_to_model_input(test_ready.epochs)
        y_train = train_ready.labels
        y_val = val_ready.labels
        y_test = test_ready.labels

        train_loader = _make_loader(x_train, y_train, batch_size=args.batch_size, shuffle=True)
        val_loader = _make_loader(x_val, y_val, batch_size=args.batch_size, shuffle=False)
        test_loader = _make_loader(x_test, y_test, batch_size=args.batch_size, shuffle=False)

        model = EEGNet(
            nb_classes=2,
            Chans=x_train.shape[1],
            Samples=x_train.shape[2],
            dropoutRate=0.5,
            kernLength=32,
            F1=8,
            D=2,
            F2=16,
            dropoutType="Dropout",
        ).to(device)
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss()

        best_val_balanced_accuracy = -1.0
        with tempfile.NamedTemporaryFile(suffix=f"_fold{fold_idx}.pt", delete=False) as checkpoint_file:
            checkpoint_path = Path(checkpoint_file.name)

        for _ in range(args.epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                model.apply_max_norm_constraints()

            y_val_true, y_val_pred, y_val_prob = _evaluate_loader(model=model, loader=val_loader, device=device)
            val_metrics = compute_binary_metrics(y_true=y_val_true, y_pred=y_val_pred, y_prob=y_val_prob)
            val_balanced_accuracy = float(val_metrics["balanced_accuracy"] or 0.0)
            if val_balanced_accuracy > best_val_balanced_accuracy:
                best_val_balanced_accuracy = val_balanced_accuracy
                torch.save(model.state_dict(), checkpoint_path)

        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        checkpoint_path.unlink(missing_ok=True)

        y_test_true, y_test_pred, y_test_prob = _evaluate_loader(model=model, loader=test_loader, device=device)
        test_metrics = compute_binary_metrics(y_true=y_test_true, y_pred=y_test_pred, y_prob=y_test_prob)
        test_metrics["fold"] = fold_idx
        test_metrics["n_test_subjects"] = len(test_subjects)
        test_metrics["n_test_trials"] = int(len(y_test_true))
        fold_metrics.append(test_metrics)

        combined_subject_metrics.update(
            _subject_metrics(
                trial_table=test_ready.trial_table,
                y_true=y_test_true,
                y_pred=y_test_pred,
                y_prob=y_test_prob,
            )
        )

        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        plot_roi_erp_comparison(
            epochs=test_ready.epochs,
            labels=y_test_pred,
            times=test_ready.times,
            channel_names=test_ready.channel_names,
            output_dir=fold_dir,
            roi_channels=CPP_ROI_CHANNELS,
        )

    summary = summarize_fold_metrics(fold_metrics)
    plot_fold_metrics(fold_metrics=fold_metrics, output_dir=output_dir)
    plot_subject_balanced_accuracy(subject_metrics=combined_subject_metrics, output_dir=output_dir)

    with open(output_dir / "fold_metrics.json", "w", encoding="utf-8") as f:
        json.dump(fold_metrics, f, indent=2)
    with open(output_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "subject_metrics.json", "w", encoding="utf-8") as f:
        json.dump(combined_subject_metrics, f, indent=2)


if __name__ == "__main__":
    main()
