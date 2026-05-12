from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_epoch_dataset(cpp_output_root: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    metadata = pd.read_csv(cpp_output_root / "epoch_metadata.csv")
    tensor = np.load(cpp_output_root / "epoch_tensor.npy")
    labels = np.load(cpp_output_root / "epoch_label.npy")
    if len(metadata) != len(tensor) or len(metadata) != len(labels):
        raise ValueError("Metadata, tensor, and labels are not aligned.")
    return metadata, tensor, labels


def _flatten_tensor(tensor: np.ndarray) -> np.ndarray:
    return tensor.reshape(tensor.shape[0], -1).astype(np.float64, copy=False)


def split_by_subject(
    metadata: pd.DataFrame,
    tensor: np.ndarray,
    labels: np.ndarray,
    *,
    subject_column: str = "subject_id",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
    groups = metadata[subject_column].to_numpy()
    indices = np.arange(len(metadata))
    first_split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(first_split.split(indices, labels, groups))

    rel_val_size = val_size / (1 - test_size)
    second_split = GroupShuffleSplit(n_splits=1, test_size=rel_val_size, random_state=random_state)
    train_idx_local, val_idx_local = next(
        second_split.split(train_val_idx, labels[train_val_idx], groups[train_val_idx])
    )
    train_idx = train_val_idx[train_idx_local]
    val_idx = train_val_idx[val_idx_local]

    return {
        "train": (_flatten_tensor(tensor[train_idx]), labels[train_idx], metadata.iloc[train_idx].reset_index(drop=True)),
        "val": (_flatten_tensor(tensor[val_idx]), labels[val_idx], metadata.iloc[val_idx].reset_index(drop=True)),
        "test": (_flatten_tensor(tensor[test_idx]), labels[test_idx], metadata.iloc[test_idx].reset_index(drop=True)),
    }


def train_ann(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int = 42,
) -> Pipeline:
    if np.unique(y_train).size < 2:
        raise ValueError("ANN training requires at least two classes in the training split.")

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.99, svd_solver="full")),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(32,),
                    activation="tanh",
                    solver="lbfgs",
                    alpha=1e-3,
                    batch_size="auto",
                    max_iter=300,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_ann(model: Pipeline, X: np.ndarray, y: np.ndarray, split_name: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        "split": split_name,
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba) if np.unique(y).size > 1 else np.nan,
        "n_samples": len(y),
    }
    return pd.DataFrame([metrics]), y_pred, y_proba


def run_ann_training(cpp_output_root: Path, ann_output_root: Path) -> dict[str, Path]:
    ann_output_root.mkdir(parents=True, exist_ok=True)
    metadata, tensor, labels = load_epoch_dataset(cpp_output_root)

    labeled_mask = labels >= 0
    metadata = metadata.loc[labeled_mask].reset_index(drop=True)
    tensor = tensor[labeled_mask]
    labels = labels[labeled_mask]

    splits = split_by_subject(metadata, tensor, labels)
    X_train, y_train, _ = splits["train"]
    X_val, y_val, val_metadata = splits["val"]
    X_test, y_test, test_metadata = splits["test"]

    model = train_ann(X_train, y_train)

    val_metrics, val_pred, val_proba = evaluate_ann(model, X_val, y_val, "val")
    test_metrics, test_pred, test_proba = evaluate_ann(model, X_test, y_test, "test")
    metrics = pd.concat([val_metrics, test_metrics], ignore_index=True)

    predictions = pd.concat(
        [
            val_metadata.assign(split="val", true_label=y_val, pred_label=val_pred, pred_probability=val_proba),
            test_metadata.assign(split="test", true_label=y_test, pred_label=test_pred, pred_probability=test_proba),
        ],
        ignore_index=True,
    )

    metrics_path = ann_output_root / "metrics.csv"
    predictions_path = ann_output_root / "predictions.csv"
    model_path = ann_output_root / "mlp_classifier.joblib"

    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    joblib.dump(model, model_path)

    return {
        "metrics_csv": metrics_path,
        "predictions_csv": predictions_path,
        "model_path": model_path,
    }
