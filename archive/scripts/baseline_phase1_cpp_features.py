from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.data_kosciessa import DATASET_DIR, filter_subject_ids_with_paths
from src.evaluation.evaluate_phase1 import compute_binary_metrics, summarize_fold_metrics
from scripts.train_phase1_eegnet import (
    SubjectDataset,
    _build_fold_assignment,
    _concatenate_subject_blocks,
    _load_subject_dataset,
    _split_train_val_subjects,
)
from src.features.labels_cpp import fit_cpp_label_transform, threshold_cpp_scores, transform_cpp_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase-1 CPP-feature logistic baseline.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase1_baseline"))
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=2000)
    return parser.parse_args()


def _feature_matrix(features: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([features["ams"], features["pams"], features["slps"]]).astype(np.float32)


def _filter_block_features(block, transform) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _feature_matrix(block.features)
    scores = transform_cpp_scores(block.features, transform)
    labels, keep_mask = threshold_cpp_scores(scores, transform)
    keep_mask = keep_mask.astype(bool)
    return x[keep_mask], labels[keep_mask], keep_mask


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

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
    per_subject_metrics: dict[str, dict[str, float | int | None]] = {}

    for fold_idx, test_subjects in enumerate(fold_assignments, start=1):
        remaining_subjects = [subject_id for subject_id in subject_ids if subject_id not in test_subjects]
        train_subjects, val_subjects = _split_train_val_subjects(remaining_subjects)

        train_block = _concatenate_subject_blocks([subject_blocks[subject_id] for subject_id in train_subjects])
        val_block = _concatenate_subject_blocks([subject_blocks[subject_id] for subject_id in val_subjects])
        test_block = _concatenate_subject_blocks([subject_blocks[subject_id] for subject_id in test_subjects])

        transform = fit_cpp_label_transform(train_block.features)
        x_train, y_train, _ = _filter_block_features(train_block, transform)
        x_val, y_val, val_keep_mask = _filter_block_features(val_block, transform)
        x_test, y_test, test_keep_mask = _filter_block_features(test_block, transform)
        test_trial_table = test_block.trial_table.iloc[test_keep_mask].reset_index(drop=True)

        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=args.max_iter, solver="liblinear"))
        model.fit(x_train, y_train)

        val_prob = model.predict_proba(x_val)[:, 1]
        val_pred = (val_prob >= 0.5).astype(np.int64)
        _ = compute_binary_metrics(y_true=y_val, y_pred=val_pred, y_prob=val_prob)

        test_prob = model.predict_proba(x_test)[:, 1]
        test_pred = (test_prob >= 0.5).astype(np.int64)
        test_metrics = compute_binary_metrics(y_true=y_test, y_pred=test_pred, y_prob=test_prob)
        test_metrics["fold"] = fold_idx
        test_metrics["n_test_subjects"] = len(test_subjects)
        test_metrics["n_test_trials"] = int(len(y_test))
        fold_metrics.append(test_metrics)

        for subject_id, group in test_trial_table.groupby("subject_id"):
            idx = group.index.to_numpy()
            per_subject_metrics[str(subject_id)] = compute_binary_metrics(
                y_true=y_test[idx],
                y_pred=test_pred[idx],
                y_prob=test_prob[idx],
            )

    summary = summarize_fold_metrics(fold_metrics)

    with open(output_dir / "fold_metrics.json", "w", encoding="utf-8") as f:
        json.dump(fold_metrics, f, indent=2)

    with open(output_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "subject_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_subject_metrics, f, indent=2)


if __name__ == "__main__":
    main()
