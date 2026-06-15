from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
S1_MODELING_DIR = REPO_ROOT / "Scripts" / "s1_modeling"
SCRIPTS_DIR = REPO_ROOT / "Scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
sys.modules.setdefault("modeling", importlib.import_module("s1_modeling"))

from modeling.config import TrainingConfig
from modeling.dataset import load_stage2_dataset


DATASET_DIR = REPO_ROOT / "Data" / "ProcessedData"
AUDIT_ROOT = REPO_ROOT / "Results" / "validation"
OUT_DIR = AUDIT_ROOT / "hidden_cpp_audit"
FIG_DIR = AUDIT_ROOT / "figures" / "publication_style"
CHECKPOINT_PATH = REPO_ROOT / "Results" / "model_checkpoints" / "best_model.pt"
LATENTS_PATH = REPO_ROOT / "Data" / "IntermediateData" / "latents_full" / "latents_full.npz"
METHODS_TRACE_PATH = OUT_DIR / "methods_trace.md"

WINDOWS = {
    "minus600_to_minus300": (-600.0, -300.0),
    "minus300_to_minus120": (-300.0, -120.0),
    "minus120_to_minus50": (-120.0, -50.0),
    "minus600_to_minus50": (-600.0, -50.0),
}
WINDOW_ORDER = [
    "minus600_to_minus300",
    "minus300_to_minus120",
    "minus120_to_minus50",
    "minus600_to_minus50",
]
WINDOW_LABELS = {
    "minus600_to_minus300": "-600 to -300 ms",
    "minus300_to_minus120": "-300 to -120 ms",
    "minus120_to_minus50": "-120 to -50 ms",
    "minus600_to_minus50": "-600 to -50 ms",
}
CPP_FEATURE_ORDER = ["amp", "slope", "auc"]
CPP_FEATURE_LABELS = {"amp": "CPP amplitude", "slope": "CPP slope", "auc": "CPP AUC"}
TASK_TARGETS = [
    ("choice", "Choice"),
    ("condition", "Condition"),
    ("correctness", "Correctness"),
    ("difficulty", "Difficulty"),
    ("rt_bin", "RT bin"),
    ("probe_leftrightwin", "Arrangement"),
]
RT_BIN_LABELS = ["fast", "medium", "slow"]
RIDGE_ALPHAS = np.logspace(-3, 5, 25)


@dataclass
class AuditResources:
    config: TrainingConfig
    raw_eeg: np.ndarray
    metadata: pd.DataFrame
    latents: np.ndarray
    times_ms: np.ndarray
    split_artifacts: Any
    checkpoint_config: dict[str, Any]


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the strict hidden-CPP audit.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--latent-path", type=Path, default=LATENTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=AUDIT_ROOT)
    return parser


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(colors="#333333")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300)
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    plt.close(fig)


def corr_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() < 3:
        return float("nan")
    a = y_true[ok]
    b = y_pred[ok]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(pearsonr(a, b).statistic)


def load_resources() -> AuditResources:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    config = TrainingConfig(**checkpoint["config"])
    _, _, _, artifacts, _ = load_stage2_dataset(DATASET_DIR, config)
    raw_eeg = np.load(DATASET_DIR / "eeg_cpp_trials.npy").astype(np.float32)
    metadata = pd.read_csv(DATASET_DIR / "metadata.csv")
    latents_npz = np.load(LATENTS_PATH, allow_pickle=True)
    latents = latents_npz["Z"].astype(np.float32)
    times_ms = latents_npz["times_ms"].astype(np.float32)
    return AuditResources(
        config=config,
        raw_eeg=raw_eeg,
        metadata=metadata,
        latents=latents,
        times_ms=times_ms,
        split_artifacts=artifacts,
        checkpoint_config=checkpoint["config"],
    )


def row_shuffle(X: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return X[rng.permutation(X.shape[0])]


def grouped_shuffle(X: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = X.copy()
    if isinstance(groups, np.ndarray) and groups.ndim > 1:
        group_list = [tuple(row.tolist()) for row in groups]
    else:
        group_list = list(groups)
    unique_groups = pd.Series(group_list, dtype="object").drop_duplicates().tolist()
    for group in unique_groups:
        mask = np.asarray([item == group for item in group_list], dtype=bool)
        idx = np.flatnonzero(mask)
        if len(idx) > 1:
            out[idx] = out[rng.permutation(idx)]
    return out


def within_subject_folds(y: np.ndarray, subjects: np.ndarray, n_splits: int, stratified: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    fold_id = np.full(len(y), -1, dtype=int)
    rng = np.random.default_rng(101)
    for subject in np.unique(subjects):
        idx = np.flatnonzero(subjects == subject)
        if len(idx) < n_splits:
            split = np.array_split(rng.permutation(idx), min(len(idx), n_splits))
            for fold, test in enumerate(split):
                fold_id[test] = fold
            continue
        if stratified and len(np.unique(y[idx])) > 1 and min(np.unique(y[idx], return_counts=True)[1]) >= n_splits:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=101)
            for fold, (_, test_local) in enumerate(splitter.split(np.zeros(len(idx)), y[idx])):
                fold_id[idx[test_local]] = fold
        else:
            for fold, test_local in enumerate(np.array_split(rng.permutation(idx), n_splits)):
                fold_id[test_local] = fold
    return [(np.flatnonzero(fold_id != fold), np.flatnonzero(fold_id == fold)) for fold in sorted(set(fold_id.tolist())) if fold >= 0]


def make_regression_folds(y: np.ndarray, subjects: np.ndarray, split_name: str) -> list[tuple[np.ndarray, np.ndarray]]:
    if split_name == "trial_level":
        return list(KFold(n_splits=5, shuffle=True, random_state=2026).split(np.zeros(len(y))))
    if split_name == "within_subject":
        return within_subject_folds(y, subjects, 5, stratified=False)
    raise ValueError(split_name)


def make_classification_folds(y: np.ndarray, subjects: np.ndarray, split_name: str) -> list[tuple[np.ndarray, np.ndarray]]:
    if split_name == "trial_level":
        return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=2026).split(np.zeros(len(y)), y))
    if split_name == "within_subject":
        return within_subject_folds(y, subjects, 5, stratified=True)
    raise ValueError(split_name)


def make_hidden_features(latents: np.ndarray, times_ms: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for window_name, (lo, hi) in WINDOWS.items():
        mask = (times_ms >= lo) & (times_ms <= hi)
        out[window_name] = latents[:, mask, :].mean(axis=1)
    return out


def make_cpp_targets(raw_eeg: np.ndarray, times_ms: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    cpp = raw_eeg.mean(axis=2)
    trapz_fn = getattr(np, "trapezoid", np.trapz)
    rows: dict[str, np.ndarray] = {}
    by_name: dict[str, np.ndarray] = {}
    for window_name, (lo, hi) in WINDOWS.items():
        mask = (times_ms >= lo) & (times_ms <= hi)
        x = times_ms[mask].astype(float)
        y = cpp[:, mask].astype(float)
        rows[f"cpp_amp_{window_name}"] = y.mean(axis=1)
        rows[f"cpp_slope_{window_name}"] = np.asarray([np.polyfit(x, row, 1)[0] for row in y], dtype=float)
        rows[f"cpp_auc_{window_name}"] = np.asarray([trapz_fn(row, x) for row in y], dtype=float)
    for key, values in rows.items():
        by_name[key] = values
    return pd.DataFrame(rows), by_name


def build_ridge_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=RIDGE_ALPHAS, cv=5)),
        ]
    )


def run_regression_cv(
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
    subjects: np.ndarray,
    trial_ids: np.ndarray,
    subject_ids: np.ndarray,
    hidden_window: str,
    target_name: str,
    control_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    folds = make_regression_folds(y, subjects, split_name)
    fold_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    coef_rows: list[dict[str, Any]] = []
    y_pred_all = np.full(len(y), np.nan, dtype=float)
    alpha_values = []
    for fold_index, (train_idx, test_idx) in enumerate(folds, start=1):
        model = build_ridge_pipeline()
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx]).astype(float)
        y_pred_all[test_idx] = y_pred
        alpha = float(model.named_steps["ridge"].alpha_)
        alpha_values.append(alpha)
        fold_rows.append(
            {
                "split": split_name,
                "hidden_window": hidden_window,
                "target_name": target_name,
                "control": control_name,
                "fold": fold_index,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "r2": float(r2_score(y[test_idx], y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y[test_idx], y_pred))),
                "mae": float(mean_absolute_error(y[test_idx], y_pred)),
                "corr": corr_1d(y[test_idx], y_pred),
                "selected_alpha": alpha,
            }
        )
        coefs = model.named_steps["ridge"].coef_.astype(float).ravel()
        for dim_index, coefficient in enumerate(coefs):
            coef_rows.append(
                {
                    "split": split_name,
                    "hidden_window": hidden_window,
                    "target_name": target_name,
                    "control": control_name,
                    "fold": fold_index,
                    "hidden_dim_index": dim_index,
                    "coefficient": float(coefficient),
                    "selected_alpha": alpha,
                }
            )
        for local_index, trial_index in enumerate(test_idx):
            pred_rows.append(
                {
                    "split": split_name,
                    "hidden_window": hidden_window,
                    "target_name": target_name,
                    "control": control_name,
                    "fold": fold_index,
                    "trial_index": int(trial_index),
                    "trial_id": str(trial_ids[trial_index]),
                    "subject_id": str(subject_ids[trial_index]),
                    "y_true": float(y[trial_index]),
                    "y_pred": float(y_pred[local_index]),
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    performance_row = {
        "split": split_name,
        "hidden_window": hidden_window,
        "target_name": target_name,
        "control": control_name,
        "n_trials": int(len(y)),
        "hidden_dim": int(X.shape[1]),
        "mean_cv_r2": float(fold_df["r2"].mean()),
        "sd_cv_r2": float(fold_df["r2"].std(ddof=1)),
        "pooled_corr": corr_1d(y.astype(float), y_pred_all),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred_all))),
        "mae": float(mean_absolute_error(y, y_pred_all)),
        "selected_alpha_mean": float(np.mean(alpha_values)),
        "selected_alpha_median": float(np.median(alpha_values)),
        "selected_alpha_values": json.dumps(alpha_values),
        "fold_r2_values": json.dumps(fold_df["r2"].tolist()),
        "fold_rmse_values": json.dumps(fold_df["rmse"].tolist()),
    }
    return performance_row, pred_rows, coef_rows


def demean_within_subject(X: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_out = X.copy().astype(float)
    y_out = y.copy().astype(float)
    for subject in np.unique(subjects):
        idx = np.flatnonzero(subjects == subject)
        X_out[idx] -= X_out[idx].mean(axis=0, keepdims=True)
        y_out[idx] -= y_out[idx].mean()
    return X_out, y_out


def run_hidden_cpp_audit(resources: AuditResources) -> dict[str, Any]:
    metadata = resources.metadata.copy()
    latents = resources.latents
    times_ms = resources.times_ms
    subjects = metadata["subject_id"].astype(str).to_numpy()
    trial_ids = metadata["trial_id"].astype(str).to_numpy()
    conditions = metadata["condition"].astype(str).to_numpy()
    hidden_features = make_hidden_features(latents, times_ms)
    cpp_df, cpp_targets = make_cpp_targets(resources.raw_eeg, times_ms)

    observed_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []

    trial_shuffle_cache = {window: row_shuffle(X, 300 + idx) for idx, (window, X) in enumerate(hidden_features.items())}
    subject_shuffle_cache = {window: grouped_shuffle(X, subjects, 400 + idx) for idx, (window, X) in enumerate(hidden_features.items())}
    subject_condition_cache = {
        window: grouped_shuffle(X, np.asarray(list(zip(subjects, conditions)), dtype=object), 500 + idx)
        for idx, (window, X) in enumerate(hidden_features.items())
    }

    for hidden_window in WINDOW_ORDER:
        X_real = hidden_features[hidden_window]
        X_trial_shuf = trial_shuffle_cache[hidden_window]
        X_subject_shuf = subject_shuffle_cache[hidden_window]
        X_subject_condition_shuf = subject_condition_cache[hidden_window]
        for target_window in WINDOW_ORDER:
            for feature_name in CPP_FEATURE_ORDER:
                target_name = f"cpp_{feature_name}_{target_window}"
                y = cpp_targets[target_name].astype(float)
                mismatch = hidden_window != target_window
                for split_name in ["trial_level", "within_subject"]:
                    perf_real, pred_real, coef_real = run_regression_cv(
                        X_real,
                        y,
                        split_name,
                        subjects,
                        trial_ids,
                        subjects,
                        hidden_window,
                        target_name,
                        "observed_hidden",
                    )
                    perf_real["target_window"] = target_window
                    perf_real["target_feature"] = feature_name
                    perf_real["time_window_relation"] = "matched" if not mismatch else "mismatched"
                    observed_rows.append(perf_real)
                    prediction_rows.extend(
                        [
                            row
                            | {
                                "target_window": target_window,
                                "target_feature": feature_name,
                                "time_window_relation": "matched" if not mismatch else "mismatched",
                            }
                            for row in pred_real
                        ]
                    )
                    coefficient_rows.extend(
                        [
                            row
                            | {
                                "target_window": target_window,
                                "target_feature": feature_name,
                                "time_window_relation": "matched" if not mismatch else "mismatched",
                            }
                            for row in coef_real
                        ]
                    )
                    for control_name, X_control in [
                        ("trial_shuffled_hidden", X_trial_shuf),
                        ("within_subject_shuffled_hidden", X_subject_shuf),
                        ("within_subject_condition_shuffled_hidden", X_subject_condition_shuf),
                    ]:
                        perf_ctrl, _, _ = run_regression_cv(
                            X_control,
                            y,
                            split_name,
                            subjects,
                            trial_ids,
                            subjects,
                            hidden_window,
                            target_name,
                            control_name,
                        )
                        perf_ctrl["target_window"] = target_window
                        perf_ctrl["target_feature"] = feature_name
                        perf_ctrl["time_window_relation"] = "matched" if not mismatch else "mismatched"
                        control_rows.append(perf_ctrl)

    observed_df = pd.DataFrame(observed_rows)
    control_df = pd.DataFrame(control_rows)
    pred_df = pd.DataFrame(prediction_rows)
    coef_df = pd.DataFrame(coefficient_rows)

    real_only_df = observed_df[observed_df["control"] == "observed_hidden"].copy()
    real_only_df.to_csv(OUT_DIR / "hidden_to_cpp_cv_performance.csv", index=False)
    pred_df.to_csv(OUT_DIR / "hidden_to_cpp_fold_predictions.csv", index=False)
    coef_df.to_csv(OUT_DIR / "hidden_to_cpp_coefficients_by_fold.csv", index=False)
    control_df.to_csv(OUT_DIR / "hidden_to_cpp_control_performance.csv", index=False)

    delta_rows: list[dict[str, Any]] = []
    for row in real_only_df.itertuples():
        shuffle_controls = control_df[
            (control_df["split"] == row.split)
            & (control_df["hidden_window"] == row.hidden_window)
            & (control_df["target_name"] == row.target_name)
        ]
        mismatch_controls = real_only_df[
            (real_only_df["split"] == row.split)
            & (real_only_df["target_name"] == row.target_name)
            & (real_only_df["hidden_window"] != row.hidden_window)
        ]
        best_shuffle_r2 = float(shuffle_controls["mean_cv_r2"].max()) if not shuffle_controls.empty else float("nan")
        best_mismatch_r2 = float(mismatch_controls["mean_cv_r2"].max()) if not mismatch_controls.empty else float("nan")
        control_candidates = [x for x in [best_shuffle_r2, best_mismatch_r2] if np.isfinite(x)]
        best_control_r2 = max(control_candidates) if control_candidates else float("nan")
        delta_rows.append(
            {
                "split": row.split,
                "hidden_window": row.hidden_window,
                "target_window": row.target_window,
                "target_feature": row.target_feature,
                "target_name": row.target_name,
                "time_window_relation": row.time_window_relation,
                "real_mean_cv_r2": row.mean_cv_r2,
                "best_shuffle_control_r2": best_shuffle_r2,
                "best_mismatch_control_r2": best_mismatch_r2,
                "best_control_r2": best_control_r2,
                "delta_r2_vs_best_control": float(row.mean_cv_r2 - best_control_r2) if np.isfinite(best_control_r2) else float("nan"),
            }
        )
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(OUT_DIR / "hidden_to_cpp_control_deltas.csv", index=False)

    demean_rows: list[dict[str, Any]] = []
    for hidden_window in WINDOW_ORDER:
        X_real = hidden_features[hidden_window]
        for target_window in WINDOW_ORDER:
            for feature_name in CPP_FEATURE_ORDER:
                target_name = f"cpp_{feature_name}_{target_window}"
                X_dm, y_dm = demean_within_subject(X_real, cpp_targets[target_name].astype(float), subjects)
                for split_name in ["trial_level", "within_subject"]:
                    perf_dm, _, _ = run_regression_cv(
                        X_dm,
                        y_dm,
                        split_name,
                        subjects,
                        trial_ids,
                        subjects,
                        hidden_window,
                        target_name,
                        "subject_demeaned_observed_hidden",
                    )
                    perf_dm["target_window"] = target_window
                    perf_dm["target_feature"] = feature_name
                    demean_rows.append(perf_dm)
    demean_df = pd.DataFrame(demean_rows)
    demean_df.to_csv(OUT_DIR / "hidden_to_cpp_subject_demeaned_performance.csv", index=False)

    leak_summary = {}
    for split_name in ["trial_level", "within_subject"]:
        amp_key = f"cpp_amp_minus600_to_minus50"
        slope_key = f"cpp_slope_minus600_to_minus50"
        real_amp = real_only_df[
            (real_only_df["split"] == split_name)
            & (real_only_df["hidden_window"] == "minus600_to_minus50")
            & (real_only_df["target_name"] == amp_key)
        ].iloc[0]
        real_slope = real_only_df[
            (real_only_df["split"] == split_name)
            & (real_only_df["hidden_window"] == "minus600_to_minus50")
            & (real_only_df["target_name"] == slope_key)
        ].iloc[0]
        demean_amp = demean_df[
            (demean_df["split"] == split_name)
            & (demean_df["hidden_window"] == "minus600_to_minus50")
            & (demean_df["target_name"] == amp_key)
        ].iloc[0]
        early_to_late = real_only_df[
            (real_only_df["split"] == split_name)
            & (real_only_df["hidden_window"] == "minus600_to_minus300")
            & (real_only_df["target_name"] == "cpp_amp_minus120_to_minus50")
        ].iloc[0]
        leak_summary[split_name] = {
            "cpp_amp_minus600_to_minus50_r2": float(real_amp["mean_cv_r2"]),
            "cpp_slope_minus600_to_minus50_r2": float(real_slope["mean_cv_r2"]),
            "subject_demeaned_cpp_amp_minus600_to_minus50_r2": float(demean_amp["mean_cv_r2"]),
            "early_hidden_to_late_cpp_amp_r2": float(early_to_late["mean_cv_r2"]),
        }

    report = {
        "checkpoint_path": str(CHECKPOINT_PATH),
        "latent_path": str(LATENTS_PATH),
        "eeg_path": str(DATASET_DIR / "eeg_cpp_trials.npy"),
        "metadata_path": str(DATASET_DIR / "metadata.csv"),
        "n_trials": int(latents.shape[0]),
        "n_timepoints": int(latents.shape[1]),
        "n_hidden_dim": int(latents.shape[2]),
        "windows": {key: list(value) for key, value in WINDOWS.items()},
        "pipeline": {
            "hidden_features": "X_hidden = Z[:, window_mask, :].mean(axis=1)",
            "cpp_signal": "CPP = mean(CP1, CP2, CPz) from raw EEG",
            "cpp_targets": ["amplitude mean", "slope np.polyfit(time, CPP, 1)[0]", "AUC np.trapezoid(CPP, time)"],
            "regression_model": "Pipeline([('scaler', StandardScaler()), ('ridge', RidgeCV(alphas=np.logspace(-3, 5, 25), cv=5))])",
            "outer_splits": ["trial_level 5-fold KFold", "within_subject 5-fold subject-balanced folds"],
        },
        "leakage_audit": {
            "same_empirical_segment_as_target": True,
            "same_empirical_segment_explanation": "Hidden states were extracted from the same response-locked EEG trial segments that were later summarized into empirical CPP targets.",
            "original_model_training_split_reproduced": {
                "train": int(len(resources.split_artifacts.train_indices)),
                "val": int(len(resources.split_artifacts.val_indices)),
                "test": int(len(resources.split_artifacts.test_indices)),
                "split_rule": "Random trial split from load_stage2_dataset with TrainingConfig.seed=42.",
            },
            "target_is_empirical_cpp": True,
            "target_computed_from_model_output": False,
            "old_audit_had_model_predicted_target": True,
            "high_r2_checks": leak_summary,
            "conservative_interpretation": (
                "High hidden-to-CPP amplitude scores are expected to be strong because the latent states are deterministic summaries of the same EEG trials "
                "used to compute the empirical CPP targets. The stricter controls therefore matter more than the raw R2 alone. If matched-window prediction "
                "still exceeds shuffled and time-mismatch controls after within-subject CV and subject demeaning, that supports stable CPP-related information, "
                "but it still should be framed as neural validation rather than a mechanistic behavioral claim."
            ),
        },
    }
    (OUT_DIR / "hidden_to_cpp_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "cpp_df": cpp_df,
        "performance": real_only_df,
        "controls": control_df,
        "deltas": delta_df,
        "demeaned": demean_df,
    }


def run_task_decoding_audit(resources: AuditResources) -> dict[str, pd.DataFrame]:
    metadata = resources.metadata.copy()
    metadata["rt_bin"] = pd.qcut(metadata["RT_ms"], 3, labels=RT_BIN_LABELS)
    subjects = metadata["subject_id"].astype(str).to_numpy()
    hidden_features = make_hidden_features(resources.latents, resources.times_ms)
    trial_shuffle_cache = {window: row_shuffle(X, 800 + idx) for idx, (window, X) in enumerate(hidden_features.items())}
    subject_shuffle_cache = {window: grouped_shuffle(X, subjects, 900 + idx) for idx, (window, X) in enumerate(hidden_features.items())}
    subject_condition_cache = {
        window: grouped_shuffle(X, np.asarray(list(zip(subjects, metadata["condition"].astype(str).to_numpy())), dtype=object), 1000 + idx)
        for idx, (window, X) in enumerate(hidden_features.items())
    }

    class_count_rows: list[dict[str, Any]] = []
    observed_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []

    for target_key, target_label in TASK_TARGETS:
        y = metadata[target_key].astype(str).to_numpy()
        counts = pd.Series(y).value_counts(dropna=False).sort_index()
        chance_level = 1.0 / float(len(counts))
        for class_name, count in counts.items():
            class_count_rows.append(
                {
                    "target": target_key,
                    "target_label": target_label,
                    "class_label": str(class_name),
                    "n_trials": int(count),
                    "proportion": float(count / len(y)),
                    "chance_level_balanced_accuracy": chance_level,
                }
            )

        for hidden_window in WINDOW_ORDER:
            X_real = hidden_features[hidden_window]
            X_trial_shuf = trial_shuffle_cache[hidden_window]
            X_subject_shuf = subject_shuffle_cache[hidden_window]
            X_subject_condition_shuf = subject_condition_cache[hidden_window]
            for split_name in ["trial_level", "within_subject"]:
                folds = make_classification_folds(y, subjects, split_name)
                controls_for_target = [
                    ("observed_hidden", X_real, "observed_hidden"),
                    ("shuffled_label", X_real, "shuffled_label"),
                    ("trial_shuffled_hidden", X_trial_shuf, "observed_hidden"),
                    ("within_subject_shuffled_hidden", X_subject_shuf, "observed_hidden"),
                    ("within_subject_condition_shuffled_hidden", X_subject_condition_shuf, "observed_hidden"),
                    ("dummy_majority", X_real, "dummy_majority"),
                ]
                fold_results: list[dict[str, Any]] = []
                for control_name, X_control, train_mode in controls_for_target:
                    scores = []
                    rng = np.random.default_rng(2026)
                    for fold_index, (train_idx, test_idx) in enumerate(folds, start=1):
                        y_train = y[train_idx].copy()
                        if train_mode == "shuffled_label":
                            y_train = rng.permutation(y_train)
                        if train_mode == "dummy_majority":
                            model = DummyClassifier(strategy="most_frequent")
                        else:
                            model = Pipeline(
                                [
                                    ("scaler", StandardScaler()),
                                    ("clf", LogisticRegression(max_iter=300, class_weight="balanced", solver="liblinear")),
                                ]
                            )
                        if len(np.setdiff1d(np.unique(y[test_idx]), np.unique(y_train))) > 0:
                            continue
                        model.fit(X_control[train_idx], y_train)
                        pred = model.predict(X_control[test_idx])
                        scores.append(float(balanced_accuracy_score(y[test_idx], pred)))
                    fold_results.append(
                        {
                            "target": target_key,
                            "target_label": target_label,
                            "hidden_window": hidden_window,
                            "split": split_name,
                            "control": control_name,
                            "class_count": int(len(counts)),
                            "chance_level_balanced_accuracy": chance_level,
                            "observed_balanced_accuracy": float(np.mean(scores)),
                            "balanced_accuracy_sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else float("nan"),
                            "fold_balanced_accuracy_values": json.dumps(scores),
                        }
                    )

                fold_df = pd.DataFrame(fold_results)
                obs = fold_df[fold_df["control"] == "observed_hidden"].iloc[0].to_dict()
                best_control = float(fold_df[fold_df["control"] != "observed_hidden"]["observed_balanced_accuracy"].max())
                obs["margin_above_best_control"] = float(obs["observed_balanced_accuracy"] - best_control)
                obs["best_control_balanced_accuracy"] = best_control
                observed_rows.append(obs)
                control_rows.extend(fold_df[fold_df["control"] != "observed_hidden"].to_dict("records"))

    observed_df = pd.DataFrame(observed_rows)
    controls_df = pd.DataFrame(control_rows)
    counts_df = pd.DataFrame(class_count_rows)
    observed_df.to_csv(OUT_DIR / "hidden_task_decoding_performance.csv", index=False)
    controls_df.to_csv(OUT_DIR / "hidden_task_decoding_controls.csv", index=False)
    counts_df.to_csv(OUT_DIR / "hidden_task_decoding_class_counts.csv", index=False)
    return {
        "performance": observed_df,
        "controls": controls_df,
        "counts": counts_df,
    }


def write_methods_trace(resources: AuditResources) -> None:
    text = f"""# Methods Trace

This file traces the actual code and data dependencies used to generate the current publication-style figures related to hidden states, behavioral validation, and shared-scale waveform plotting.

## Figure 1

### `main_figure_2_hidden_state_relations.pdf`

1. **Output figure file**
   - `{FIG_DIR / "main_figure_2_hidden_state_relations.pdf"}`
2. **Figure drawing code**
   - Script: `{AUDIT_ROOT / "make_publication_figures.py"}`
   - Function: `make_hidden_state_figure()`
3. **Immediate input tables used by the figure**
   - `{AUDIT_ROOT / "hidden_state_neural_regression_decoding.csv"}`
   - `{AUDIT_ROOT / "hidden_state_classification_decoding.csv"}`
4. **Upstream code that generated those tables**
   - Script: `{AUDIT_ROOT / "run_neural_validation_audit.py"}`
   - Function: `hidden_state_validation(...)`
5. **Model checkpoint used upstream**
   - `{CHECKPOINT_PATH}`
6. **Latent file used upstream**
   - `{LATENTS_PATH}`
7. **EEG/CPP file used upstream**
   - `{DATASET_DIR / "eeg_cpp_trials.npy"}`
   - Important detail: the upstream hidden-state validation used `load_stage2_dataset(...)`, which returns channel-normalized EEG for the regression targets in that script.
8. **Metadata file used upstream**
   - `{DATASET_DIR / "metadata.csv"}`
9. **Train/test or CV split used upstream**
   - Latent and metadata rows covered all 7297 trials from `latents_full.npz`.
   - Hidden-to-CPP regression in the old audit used **within-subject 5-fold CV only**.
   - Task decoding in the old audit used both `trial_level` and `within_subject` 5-fold CV, but the publication figure selected the `within_subject` rows.
10. **Time windows used upstream**
   - `-600 to -300 ms`
   - `-300 to -120 ms`
   - `-120 to -50 ms`
   - `-600 to -50 ms`
11. **Model type used upstream**
   - Regression: `Pipeline(StandardScaler(), Ridge(alpha=10.0))`
   - Classification: `Pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced", solver="liblinear"))`
12. **Scoring metric used upstream**
   - Regression heatmap: mean CV `R^2`
   - Task coding heatmap: balanced accuracy above the best control
13. **Control or shuffled baselines used upstream**
   - Regression controls: `shuffled_hidden_within_subject`, `shuffled_label`
   - Classification controls: `shuffled_hidden_within_subject`, `shuffled_hidden_within_subject_condition`, `shuffled_label`, `dummy_majority`

## Figure 2

### `supplementary_figure_behavioral_external_validation.pdf`

1. **Output figure file**
   - `{FIG_DIR / "supplementary_figure_behavioral_external_validation.pdf"}`
2. **Figure drawing code**
   - Script: `{AUDIT_ROOT / "make_publication_figures.py"}`
   - Function: `make_behavior_figure()`
3. **Immediate input table used by the figure**
   - `{AUDIT_ROOT / "behavioral_external_validation_rt.csv"}`
4. **Upstream code that generated the table**
   - Script: `{AUDIT_ROOT / "run_neural_validation_audit.py"}`
   - Function: `behavioral_external_validation(...)`
5. **Model checkpoint used upstream**
   - `{CHECKPOINT_PATH}`
6. **Latent file used upstream**
   - `{LATENTS_PATH}`
7. **EEG/CPP file used upstream**
   - `{DATASET_DIR / "eeg_cpp_trials.npy"}`
   - Important detail: the CPP features in the old behavior figure were derived from the EEG array returned by `load_stage2_dataset(...)`, so they were based on channel-normalized EEG units rather than raw EEG units.
8. **Metadata file used upstream**
   - `{DATASET_DIR / "metadata.csv"}`
9. **Train/test or CV split used upstream**
   - 5-fold `within_subject` cross-validation over all 7297 trials
10. **Time windows used upstream**
   - Hidden states: `-600 to -50 ms`
   - CPP features: the same four windows used in `cpp_features(...)`
11. **Model type used upstream**
   - `Pipeline(StandardScaler(), Ridge(alpha=10.0))`
12. **Scoring metric used upstream**
   - Mean cross-validated `R^2` for `log(RT_ms)`
13. **Control or shuffled baselines used upstream**
   - `behavior_only`
   - `cpp_features_only`
   - `hidden_states_only`
   - `behavior_plus_cpp`
   - `behavior_plus_hidden`
   - `behavior_plus_cpp_plus_hidden`
   - `behavior_plus_shuffled_hidden`

## Figure 3

### `supplementary_shared_scale_waveforms_from_minus600.pdf`

1. **Output figure file**
   - `{FIG_DIR / "supplementary_shared_scale_waveforms_from_minus600.pdf"}`
2. **Figure drawing code**
   - Script: `{AUDIT_ROOT / "make_publication_figures.py"}`
   - Function: `make_shared_scale_windowed_waveform_figure()`
3. **Immediate computation used by the figure**
   - Calls `load_model_predictions()` from `{AUDIT_ROOT / "run_neural_validation_audit.py"}`
4. **Model checkpoint used**
   - `{CHECKPOINT_PATH}`
5. **Latent file used**
   - None for this figure
6. **EEG/CPP file used**
   - `{DATASET_DIR / "eeg_cpp_trials.npy"}`
   - Important detail: the plotted "real" traces come from the channel-normalized EEG returned by `load_stage2_dataset(...)`, not from raw microvolt values.
7. **Metadata file used**
   - `{DATASET_DIR / "metadata.csv"}`
8. **Train/test split used**
   - Only `artifacts.test_indices` from the random trial split reproduced by `load_stage2_dataset(...)`
   - Split sizes from the current checkpoint config: train `{len(resources.split_artifacts.train_indices)}`, val `{len(resources.split_artifacts.val_indices)}`, test `{len(resources.split_artifacts.test_indices)}`
9. **Time range shown**
   - Plot displays `-600 to 200 ms`
   - Shared y-axis scale is estimated from the `-600 to 0 ms` analysis window
10. **Model type used**
   - The trained `CPPForwardGRU` checkpoint is used to produce the reconstruction traces
11. **Scoring metric used**
   - None; this figure is descriptive rather than a scored cross-validation panel
12. **Control or shuffled baseline used**
   - None

## Split provenance

- The stage-2 dataset split is defined in `{ROOT / "modeling" / "dataset.py"}` by `_random_trial_split(...)`.
- The split is a reproducible random trial split with `TrainingConfig.seed = {resources.config.seed}`.
- This means the old publication figures mix two validation layers:
  - a model-fitting split used by the GRU checkpoint itself;
  - a later readout CV analysis over the full latent file for hidden-state regressions and decoders.
"""
    METHODS_TRACE_PATH.write_text(text, encoding="utf-8")


def make_updated_figure_2(hidden_cpp_results: dict[str, Any], decoding_results: dict[str, pd.DataFrame]) -> None:
    perf = hidden_cpp_results["performance"].copy()
    deltas = hidden_cpp_results["deltas"].copy()
    task_perf = decoding_results["performance"].copy()

    perf = perf[perf["split"] == "within_subject"].copy()
    deltas = deltas[deltas["split"] == "within_subject"].copy()
    task_perf = task_perf[task_perf["split"] == "within_subject"].copy()

    row_order = [f"cpp_{feature}_{window}" for feature in CPP_FEATURE_ORDER for window in WINDOW_ORDER]
    row_labels = [f"{CPP_FEATURE_LABELS[feature]}\n{WINDOW_LABELS[window]}" for feature in CPP_FEATURE_ORDER for window in WINDOW_ORDER]

    perf_heat = perf.pivot(index="target_name", columns="hidden_window", values="mean_cv_r2").reindex(index=row_order, columns=WINDOW_ORDER)
    delta_heat = deltas.pivot(index="target_name", columns="hidden_window", values="delta_r2_vs_best_control").reindex(index=row_order, columns=WINDOW_ORDER)
    task_order = [label for _, label in TASK_TARGETS]
    task_perf["target_label"] = task_perf["target_label"].replace({"probe_leftrightwin": "Arrangement"})
    task_heat = task_perf.pivot(index="target_label", columns="hidden_window", values="margin_above_best_control").reindex(index=task_order, columns=WINDOW_ORDER)
    task_obs_text = task_perf.pivot(index="target_label", columns="hidden_window", values="observed_balanced_accuracy").reindex(index=task_order, columns=WINDOW_ORDER)

    fig = plt.figure(figsize=(14.0, 8.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.92], wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    perf_vals = perf_heat.to_numpy(dtype=float)
    im_a = ax_a.imshow(perf_vals, aspect="auto", cmap="YlGnBu", vmin=min(0.0, np.nanmin(perf_vals)), vmax=max(0.8, np.nanmax(perf_vals)))
    for i in range(perf_vals.shape[0]):
        for j in range(perf_vals.shape[1]):
            if np.isfinite(perf_vals[i, j]):
                ax_a.text(j, i, f"{perf_vals[i, j]:.2f}", ha="center", va="center", color="#16324f", fontsize=7)
    ax_a.set_xticks(range(len(WINDOW_ORDER)), [WINDOW_LABELS[w] for w in WINDOW_ORDER], rotation=25, ha="right")
    ax_a.set_yticks(range(len(row_labels)), row_labels)
    ax_a.set_title("Hidden states predicting empirical CPP features")
    clean_axis(ax_a)
    add_panel_label(ax_a, "a")
    cbar_a = fig.colorbar(im_a, ax=ax_a, fraction=0.048, pad=0.03)
    cbar_a.set_label("Mean CV $R^2$")

    delta_vals = delta_heat.to_numpy(dtype=float)
    vmax = max(0.1, np.nanmax(np.abs(delta_vals)))
    im_b = ax_b.imshow(delta_vals, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for i in range(delta_vals.shape[0]):
        for j in range(delta_vals.shape[1]):
            if np.isfinite(delta_vals[i, j]):
                ax_b.text(j, i, f"{delta_vals[i, j]:+.2f}", ha="center", va="center", color=("white" if abs(delta_vals[i, j]) > vmax * 0.45 else "#222222"), fontsize=7)
    ax_b.set_xticks(range(len(WINDOW_ORDER)), [WINDOW_LABELS[w] for w in WINDOW_ORDER], rotation=25, ha="right")
    ax_b.set_yticks(range(len(row_labels)), row_labels)
    ax_b.set_title("Control-corrected hidden-to-CPP prediction")
    clean_axis(ax_b)
    add_panel_label(ax_b, "b")
    cbar_b = fig.colorbar(im_b, ax=ax_b, fraction=0.048, pad=0.03)
    cbar_b.set_label("$R^2$ above best control")

    task_vals = task_heat.to_numpy(dtype=float)
    vmax_task = max(0.14, np.nanmax(np.abs(task_vals)))
    im_c = ax_c.imshow(task_vals, aspect="auto", cmap="RdBu_r", vmin=-vmax_task, vmax=vmax_task)
    for i in range(task_vals.shape[0]):
        for j in range(task_vals.shape[1]):
            if np.isfinite(task_vals[i, j]):
                obs = float(task_obs_text.iloc[i, j])
                ax_c.text(
                    j,
                    i,
                    f"{task_vals[i, j]:+.02f}\n{obs:.2f}",
                    ha="center",
                    va="center",
                    color=("white" if abs(task_vals[i, j]) > vmax_task * 0.45 else "#222222"),
                    fontsize=7,
                )
    ax_c.set_xticks(range(len(WINDOW_ORDER)), [WINDOW_LABELS[w] for w in WINDOW_ORDER], rotation=25, ha="right")
    ax_c.set_yticks(range(len(task_order)), task_order)
    ax_c.set_title("Task decoding")
    clean_axis(ax_c)
    add_panel_label(ax_c, "c")
    cbar_c = fig.colorbar(im_c, ax=ax_c, fraction=0.048, pad=0.03)
    cbar_c.set_label("Balanced accuracy above best control")
    ax_c.text(
        1.02,
        -0.08,
        "Task cells show margin over control\nand observed balanced accuracy",
        transform=ax_c.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#666666",
    )

    fig.suptitle("Main Figure 2. Hidden-state relation to empirical CPP features and task variables", x=0.06, y=0.99, ha="left", fontsize=12)
    save_figure(fig, "main_figure_2_hidden_state_relations")


def make_behavior_figure() -> None:
    behavior_path = AUDIT_ROOT / "behavioral_external_validation_rt.csv"
    if not behavior_path.exists():
        return
    behavior = pd.read_csv(behavior_path)
    order = [
        "behavior_only",
        "cpp_features_only",
        "hidden_states_only",
        "behavior_plus_cpp",
        "behavior_plus_hidden",
        "behavior_plus_cpp_plus_hidden",
        "behavior_plus_shuffled_hidden",
    ]
    labels = {
        "behavior_only": "Behavior only",
        "cpp_features_only": "CPP only",
        "hidden_states_only": "Hidden only",
        "behavior_plus_cpp": "Behavior + CPP",
        "behavior_plus_hidden": "Behavior + hidden",
        "behavior_plus_cpp_plus_hidden": "Behavior + CPP + hidden",
        "behavior_plus_shuffled_hidden": "Behavior + shuffled hidden",
    }
    behavior = behavior.set_index("model").reindex(order).reset_index()
    colors = []
    for model_name in behavior["model"]:
        if model_name == "behavior_plus_cpp_plus_hidden":
            colors.append("#1f3a5f")
        elif "hidden" in model_name and "shuffled" not in model_name:
            colors.append("#54708a")
        elif "cpp" in model_name:
            colors.append("#b9824d")
        elif "shuffled" in model_name:
            colors.append("#bfbfbf")
        else:
            colors.append("#7a7a7a")
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ypos = np.arange(len(behavior))
    ax.barh(
        ypos,
        behavior["r2"],
        xerr=behavior["r2_sd"],
        color=colors,
        edgecolor="none",
        height=0.65,
        error_kw={"elinewidth": 0.9, "ecolor": "#444444", "capsize": 2},
    )
    ax.axvline(0, color="#4c4c4c", linewidth=0.9)
    ax.set_yticks(ypos, [labels[name] for name in behavior["model"]])
    ax.invert_yaxis()
    ax.set_xlabel("Cross-validated $R^2$ for log RT")
    ax.set_title("Supplementary Figure. Behavioral external validation")
    clean_axis(ax)
    add_panel_label(ax, "a")
    ax.text(
        0.01,
        -0.18,
        "Displayed as secondary evidence rather than the main validation criterion.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#666666",
    )
    save_figure(fig, "supplementary_figure_behavioral_external_validation")


def main() -> None:
    global DATASET_DIR, CHECKPOINT_PATH, LATENTS_PATH, AUDIT_ROOT, OUT_DIR, FIG_DIR, METHODS_TRACE_PATH
    args = _build_parser().parse_args()
    DATASET_DIR = args.dataset_dir.resolve()
    CHECKPOINT_PATH = args.checkpoint_path.resolve()
    LATENTS_PATH = args.latent_path.resolve()
    AUDIT_ROOT = args.output_dir.resolve()
    OUT_DIR = AUDIT_ROOT / "hidden_cpp_audit"
    FIG_DIR = AUDIT_ROOT / "figures" / "publication_style"
    METHODS_TRACE_PATH = OUT_DIR / "methods_trace.md"
    ensure_dirs()
    set_style()
    resources = load_resources()
    write_methods_trace(resources)
    hidden_cpp_results = run_hidden_cpp_audit(resources)
    decoding_results = run_task_decoding_audit(resources)
    make_updated_figure_2(hidden_cpp_results, decoding_results)
    make_behavior_figure()
    print(f"Hidden CPP audit complete: {OUT_DIR}")


if __name__ == "__main__":
    main()
