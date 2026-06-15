from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
S1_MODELING_DIR = REPO_ROOT / "Scripts" / "s1_modeling"
SCRIPTS_DIR = REPO_ROOT / "Scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
sys.modules.setdefault("modeling", importlib.import_module("s1_modeling"))

from modeling.config import TrainingConfig
from modeling.dataset import load_stage2_dataset
from modeling.model import CPPForwardGRU


DATASET_DIR = REPO_ROOT / "Data" / "ProcessedData"
CHECKPOINT_PATH = REPO_ROOT / "Results" / "model_checkpoints" / "best_model.pt"
LATENTS_PATH = REPO_ROOT / "Data" / "IntermediateData" / "latents_full" / "latents_full.npz"
LATENT_REPORT_PATH = REPO_ROOT / "Data" / "IntermediateData" / "latents_full" / "latent_extraction_report.json"
OUT_DIR = REPO_ROOT / "Results" / "validation"
FIG_DIR = OUT_DIR / "figures"

WINDOWS = {
    "minus600_to_minus300": (-600.0, -300.0),
    "minus300_to_minus120": (-300.0, -120.0),
    "minus120_to_minus50": (-120.0, -50.0),
    "minus600_to_minus50": (-600.0, -50.0),
}


def ensure_dirs() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the neural validation audit.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--latent-path", type=Path, default=LATENTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def channels() -> list[str]:
    return [x.strip() for x in (DATASET_DIR / "channel_names.txt").read_text().splitlines() if x.strip()]


def corr_flat(y: np.ndarray, yhat: np.ndarray) -> float:
    a = y.reshape(-1)
    b = yhat.reshape(-1)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(pearsonr(a[ok], b[ok]).statistic)


def regression_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    return {
        "mse": float(mean_squared_error(y.reshape(-1), yhat.reshape(-1))),
        "rmse": float(np.sqrt(mean_squared_error(y.reshape(-1), yhat.reshape(-1)))),
        "r2": float(r2_score(y.reshape(-1), yhat.reshape(-1))),
        "corr": corr_flat(y, yhat),
        "empirical_mean": float(np.mean(y)),
        "predicted_mean": float(np.mean(yhat)),
        "mean_difference": float(np.mean(y) - np.mean(yhat)),
    }


def load_model_predictions() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, Any]:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    config = TrainingConfig(**checkpoint["config"])
    eeg, _, metadata, artifacts, _ = load_stage2_dataset(DATASET_DIR, config)
    times_ms = np.load(DATASET_DIR / "times_ms.npy").astype(np.float32)
    model = CPPForwardGRU(config)
    model.set_horizon(artifacts.horizon_steps)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, eeg.shape[0], config.batch_size):
            batch = torch.as_tensor(eeg[start : start + config.batch_size], dtype=torch.float32)
            preds.append(model(batch).reconstruction.detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0).astype(np.float32)
    return eeg, pred, metadata, times_ms, artifacts


def neural_goodness_tables(
    eeg: np.ndarray,
    pred: np.ndarray,
    metadata: pd.DataFrame,
    times_ms: np.ndarray,
    test_idx: np.ndarray,
) -> pd.DataFrame:
    rows = []
    y = eeg[test_idx]
    yh = pred[test_idx]
    rows.append({"group_type": "all_test_trials", "group": "all", "window": "full", **regression_metrics(y, yh)})
    for ci, ch in enumerate(channels()):
        rows.append({"group_type": "channel", "group": ch, "window": "full", **regression_metrics(y[:, :, ci], yh[:, :, ci])})
    for name, (lo, hi) in WINDOWS.items():
        mask = (times_ms >= lo) & (times_ms <= hi)
        rows.append({"group_type": "time_window", "group": "all", "window": name, **regression_metrics(y[:, mask, :], yh[:, mask, :])})
    for col in ["condition", "difficulty", "correctness"]:
        for value in sorted(metadata.iloc[test_idx][col].dropna().unique()):
            idx = test_idx[metadata.iloc[test_idx][col].to_numpy() == value]
            if len(idx) >= 5:
                rows.append({"group_type": col, "group": str(value), "window": "full", **regression_metrics(eeg[idx], pred[idx])})
    for subject, subdf in metadata.iloc[test_idx].groupby("subject_id"):
        idx = subdf.index.to_numpy()
        if len(idx) >= 10:
            rows.append({"group_type": "subject", "group": str(subject), "window": "full", **regression_metrics(eeg[idx], pred[idx])})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "neural_goodness_of_fit.csv", index=False)

    time_rows = []
    for ti, t in enumerate(times_ms):
        time_rows.append({"time_ms": float(t), **regression_metrics(y[:, ti, :], yh[:, ti, :])})
    pd.DataFrame(time_rows).to_csv(OUT_DIR / "time_resolved_neural_fit.csv", index=False)
    return out


def cpp_features(signal: np.ndarray, times_ms: np.ndarray) -> pd.DataFrame:
    cpp = signal.mean(axis=2)
    rows: dict[str, np.ndarray] = {}
    for name, (lo, hi) in WINDOWS.items():
        mask = (times_ms >= lo) & (times_ms <= hi)
        rows[f"cpp_amp_{name}"] = cpp[:, mask].mean(axis=1)
        x = times_ms[mask].astype(float)
        x = x - x.mean()
        denom = np.sum(x**2)
        rows[f"cpp_slope_{name}"] = ((cpp[:, mask] - cpp[:, mask].mean(axis=1, keepdims=True)) @ x) / denom
    return pd.DataFrame(rows)


def cpp_signature_tables(
    eeg: np.ndarray,
    pred: np.ndarray,
    metadata: pd.DataFrame,
    times_ms: np.ndarray,
    test_idx: np.ndarray,
) -> pd.DataFrame:
    emp = cpp_features(eeg[test_idx], times_ms)
    rec = cpp_features(pred[test_idx], times_ms)
    meta = metadata.iloc[test_idx].reset_index(drop=True)
    rows = []
    for col in emp.columns:
        rows.append({"group_type": "all", "group": "all", "feature": col, **regression_metrics(emp[col].to_numpy(), rec[col].to_numpy())})
    for group_col in ["condition", "difficulty", "correctness"]:
        for value in sorted(meta[group_col].dropna().unique()):
            mask = meta[group_col].to_numpy() == value
            if mask.sum() >= 5:
                for col in emp.columns:
                    rows.append(
                        {
                            "group_type": group_col,
                            "group": str(value),
                            "feature": col,
                            **regression_metrics(emp.loc[mask, col].to_numpy(), rec.loc[mask, col].to_numpy()),
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "cpp_erp_signature_fit.csv", index=False)
    return out


def save_neural_plots(eeg: np.ndarray, pred: np.ndarray, metadata: pd.DataFrame, times_ms: np.ndarray, test_idx: np.ndarray) -> None:
    y = eeg[test_idx].mean(axis=2)
    yh = pred[test_idx].mean(axis=2)
    plt.figure(figsize=(8, 4))
    plt.plot(times_ms, y.mean(axis=0), label="empirical CPP avg", color="#4C78A8")
    plt.plot(times_ms, yh.mean(axis=0), label="model reconstruction", color="#F28E2B")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Time from response (ms)")
    plt.ylabel("Normalized CPP signal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "empirical_vs_model_cpp_trajectory_test.png", dpi=170)
    plt.close()

    for group_col in ["condition", "difficulty", "correctness"]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        meta = metadata.iloc[test_idx].reset_index(drop=True)
        for value in sorted(meta[group_col].dropna().unique()):
            mask = meta[group_col].to_numpy() == value
            if mask.sum() >= 5:
                axes[0].plot(times_ms, y[mask].mean(axis=0), label=str(value))
                axes[1].plot(times_ms, yh[mask].mean(axis=0), label=str(value))
        axes[0].set_title(f"Empirical by {group_col}")
        axes[1].set_title(f"Model by {group_col}")
        for ax in axes:
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Time from response (ms)")
        axes[0].set_ylabel("Normalized CPP signal")
        axes[1].legend(title=group_col, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"cpp_trajectory_by_{group_col}.png", dpi=170)
        plt.close(fig)

    tr = pd.read_csv(OUT_DIR / "time_resolved_neural_fit.csv")
    plt.figure(figsize=(8, 4))
    plt.plot(tr["time_ms"], tr["corr"], color="#59A14F")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Time from response (ms)")
    plt.ylabel("Empirical-predicted correlation")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "time_resolved_prediction_correlation.png", dpi=170)
    plt.close()

    rng = np.random.default_rng(7)
    flat_y = eeg[test_idx].reshape(-1)
    flat_p = pred[test_idx].reshape(-1)
    sample = rng.choice(len(flat_y), size=min(30000, len(flat_y)), replace=False)
    plt.figure(figsize=(5, 5))
    plt.scatter(flat_y[sample], flat_p[sample], s=2, alpha=0.18, color="#4C78A8")
    plt.xlabel("Empirical neural value")
    plt.ylabel("Model-predicted value")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "empirical_vs_predicted_neural_scatter.png", dpi=170)
    plt.close()


def load_latents() -> tuple[np.ndarray, np.ndarray]:
    data = np.load(LATENTS_PATH, allow_pickle=True)
    return data["Z"].astype(np.float32), data["times_ms"].astype(np.float32)


def hidden_window_features(latents: np.ndarray, times_ms: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for name, (lo, hi) in WINDOWS.items():
        mask = (times_ms >= lo) & (times_ms <= hi)
        out[name] = latents[:, mask, :].mean(axis=1)
    return out


def within_subject_folds(y: np.ndarray, subjects: np.ndarray, n_splits: int, stratified: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    fold_id = np.full(len(y), -1)
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
    return [(np.flatnonzero(fold_id != f), np.flatnonzero(fold_id == f)) for f in sorted(set(fold_id.astype(int))) if f >= 0]


def shuffled_hidden(X: np.ndarray, subjects: np.ndarray, seed: int, condition: np.ndarray | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = X.copy()
    keys = list(zip(subjects, condition)) if condition is not None else subjects
    for key in pd.Series(keys).drop_duplicates():
        mask = np.array([item == key for item in keys])
        idx = np.flatnonzero(mask)
        if len(idx) > 1:
            out[idx] = out[rng.permutation(idx)]
    return out


def classification_cv(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, split: str, control: str) -> dict[str, float]:
    if split == "trial_level":
        folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=9).split(X, y))
    elif split == "within_subject":
        folds = within_subject_folds(y, subjects, 5, stratified=True)
    elif split == "leave_subject_out":
        folds = list(GroupKFold(n_splits=5).split(X, y, groups=subjects))
    else:
        raise ValueError(split)
    rows = []
    rng = np.random.default_rng(9)
    classes = np.unique(y)
    for train, test in folds:
        if len(np.setdiff1d(np.unique(y[test]), np.unique(y[train]))) > 0:
            continue
        clf = DummyClassifier(strategy="most_frequent") if control == "dummy_majority" else make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced", solver="liblinear")
        )
        y_train = rng.permutation(y[train]) if control == "shuffled_label" else y[train]
        clf.fit(X[train], y_train)
        pred = clf.predict(X[test])
        prob = clf.predict_proba(X[test]) if hasattr(clf, "predict_proba") else None
        auc = np.nan
        if prob is not None:
            try:
                if len(classes) == 2:
                    auc = roc_auc_score(y[test] == classes[1], prob[:, list(clf.classes_).index(classes[1])])
                else:
                    auc = roc_auc_score(y[test], prob, labels=classes, multi_class="ovr", average="macro")
            except Exception:
                auc = np.nan
        rows.append(
            {
                "accuracy": accuracy_score(y[test], pred),
                "balanced_accuracy": balanced_accuracy_score(y[test], pred),
                "auc": auc,
            }
        )
    frame = pd.DataFrame(rows)
    return {
        "n_folds": float(len(frame)),
        "accuracy": float(frame["accuracy"].mean()),
        "balanced_accuracy": float(frame["balanced_accuracy"].mean()),
        "auc": float(frame["auc"].mean(skipna=True)),
        "balanced_accuracy_sd": float(frame["balanced_accuracy"].std(ddof=1)),
    }


def regression_cv(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, split: str, control: str) -> dict[str, float]:
    folds = (
        list(KFold(n_splits=5, shuffle=True, random_state=11).split(X))
        if split == "trial_level"
        else within_subject_folds(y, subjects, 5, stratified=False)
    )
    rng = np.random.default_rng(11)
    rows = []
    for train, test in folds:
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        y_train = rng.permutation(y[train]) if control == "shuffled_label" else y[train]
        model.fit(X[train], y_train)
        pred = model.predict(X[test])
        rows.append(
            {
                "r2": r2_score(y[test], pred),
                "corr": corr_flat(y[test], pred),
                "mae": mean_absolute_error(y[test], pred),
                "rmse": float(np.sqrt(mean_squared_error(y[test], pred))),
            }
        )
    frame = pd.DataFrame(rows)
    return {
        "n_folds": float(len(frame)),
        "r2": float(frame["r2"].mean()),
        "corr": float(frame["corr"].mean()),
        "mae": float(frame["mae"].mean()),
        "rmse": float(frame["rmse"].mean()),
        "r2_sd": float(frame["r2"].std(ddof=1)),
    }


def hidden_state_validation(
    latents: np.ndarray,
    times_ms: np.ndarray,
    metadata: pd.DataFrame,
    eeg: np.ndarray,
    pred: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = metadata.copy()
    meta["rt_bin"] = pd.qcut(meta["RT_ms"], 3, labels=["fast", "medium", "slow"])
    subjects = meta["subject_id"].astype(str).to_numpy()
    condition = meta["condition"].to_numpy()
    X_by_window = hidden_window_features(latents, times_ms)
    class_targets = {
        "choice": meta["choice"].to_numpy(),
        "correctness": meta["correctness"].astype(int).to_numpy(),
        "difficulty": meta["difficulty"].to_numpy(),
        "condition": meta["condition"].to_numpy(),
        "arrangement_probe_leftrightwin": meta["probe_leftrightwin"].to_numpy(),
        "rt_bin": meta["rt_bin"].astype(str).to_numpy(),
    }
    emp_cpp = cpp_features(eeg, times_ms)
    pred_cpp = cpp_features(pred, times_ms)
    reg_targets = {
        "cpp_amp_minus600_to_minus50": emp_cpp["cpp_amp_minus600_to_minus50"].to_numpy(),
        "cpp_slope_minus600_to_minus50": emp_cpp["cpp_slope_minus600_to_minus50"].to_numpy(),
        "model_predicted_cpp_amp_minus600_to_minus50": pred_cpp["cpp_amp_minus600_to_minus50"].to_numpy(),
    }
    class_rows = []
    reg_rows = []
    for window, X in X_by_window.items():
        X_shuf = shuffled_hidden(X, subjects, 5)
        X_shuf_sc = shuffled_hidden(X, subjects, 6, condition=condition)
        for target, y in class_targets.items():
            for split in ["trial_level", "within_subject"]:
                for control, X_use in [
                    ("observed_hidden", X),
                    ("shuffled_hidden_within_subject", X_shuf),
                    ("shuffled_label", X),
                    ("dummy_majority", X),
                ]:
                    class_rows.append({"target": target, "window": window, "split": split, "control": control, **classification_cv(X_use, y, subjects, split, control)})
            class_rows.append(
                {
                    "target": target,
                    "window": window,
                    "split": "within_subject",
                    "control": "shuffled_hidden_within_subject_condition",
                    **classification_cv(X_shuf_sc, y, subjects, "within_subject", "shuffled_hidden_within_subject_condition"),
                }
            )
        for target, y in reg_targets.items():
            for split in ["within_subject"]:
                for control, X_use in [
                    ("observed_hidden", X),
                    ("shuffled_hidden_within_subject", X_shuf),
                    ("shuffled_label", X),
                ]:
                    reg_rows.append({"target": target, "window": window, "split": split, "control": control, **regression_cv(X_use, y, subjects, split, control)})
    class_out = pd.DataFrame(class_rows)
    reg_out = pd.DataFrame(reg_rows)
    class_out.to_csv(OUT_DIR / "hidden_state_classification_decoding.csv", index=False)
    reg_out.to_csv(OUT_DIR / "hidden_state_neural_regression_decoding.csv", index=False)
    return class_out, reg_out


def encode_behavior_matrix(meta: pd.DataFrame) -> np.ndarray:
    cols = ["choice", "correctness", "condition", "difficulty", "probe_leftrightwin", "response_hand"]
    return OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit_transform(meta[cols].astype(str))


def behavioral_external_validation(latents: np.ndarray, times_ms: np.ndarray, metadata: pd.DataFrame, eeg: np.ndarray) -> pd.DataFrame:
    meta = metadata.copy()
    y = np.log(meta["RT_ms"].to_numpy())
    subjects = meta["subject_id"].astype(str).to_numpy()
    beh = encode_behavior_matrix(meta)
    cpp = cpp_features(eeg, times_ms).to_numpy()
    Xh = hidden_window_features(latents, times_ms)["minus600_to_minus50"]
    Xhs = shuffled_hidden(Xh, subjects, 21)
    features = {
        "behavior_only": beh,
        "cpp_features_only": cpp,
        "hidden_states_only": Xh,
        "behavior_plus_cpp": np.column_stack([beh, cpp]),
        "behavior_plus_hidden": np.column_stack([beh, Xh]),
        "behavior_plus_cpp_plus_hidden": np.column_stack([beh, cpp, Xh]),
        "behavior_plus_shuffled_hidden": np.column_stack([beh, Xhs]),
    }
    folds = within_subject_folds(y, subjects, 5, stratified=False)
    rows = []
    predictions_by_model: dict[str, list[float]] = {}
    for model_name, X in features.items():
        fold_rows = []
        all_pred = np.zeros_like(y)
        for fold, (train, test) in enumerate(folds):
            model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            model.fit(X[train], y[train])
            p = model.predict(X[test])
            all_pred[test] = p
            fold_rows.append(
                {
                    "fold": fold,
                    "r2": r2_score(y[test], p),
                    "corr": corr_flat(y[test], p),
                    "mae": mean_absolute_error(y[test], p),
                    "rmse": float(np.sqrt(mean_squared_error(y[test], p))),
                }
            )
        predictions_by_model[model_name] = all_pred.tolist()
        frame = pd.DataFrame(fold_rows)
        rows.append(
            {
                "model": model_name,
                "target": "log_RT_ms",
                "split": "within_subject",
                "r2": frame["r2"].mean(),
                "r2_sd": frame["r2"].std(ddof=1),
                "corr": frame["corr"].mean(),
                "mae": frame["mae"].mean(),
                "rmse": frame["rmse"].mean(),
            }
        )
    out = pd.DataFrame(rows)
    base = float(out.loc[out["model"] == "behavior_only", "r2"].iloc[0])
    beh_cpp = float(out.loc[out["model"] == "behavior_plus_cpp", "r2"].iloc[0])
    for baseline_name, baseline_value in [("behavior_only", base), ("behavior_plus_cpp", beh_cpp)]:
        out[f"delta_r2_vs_{baseline_name}"] = out["r2"] - baseline_value
    out.to_csv(OUT_DIR / "behavioral_external_validation_rt.csv", index=False)
    save_json(OUT_DIR / "behavioral_external_validation_predictions.json", predictions_by_model)
    return out


def save_hidden_plots(class_results: pd.DataFrame, reg_results: pd.DataFrame, behavior_results: pd.DataFrame) -> None:
    obs = class_results[(class_results["split"] == "within_subject") & (class_results["control"] == "observed_hidden")]
    pivot = obs.pivot_table(index="target", columns="window", values="balanced_accuracy", aggfunc="mean")
    plt.figure(figsize=(9, 5))
    plt.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    plt.colorbar(label="Balanced accuracy")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "hidden_state_task_decoding_balanced_accuracy.png", dpi=170)
    plt.close()

    obsr = reg_results[(reg_results["split"] == "within_subject") & (reg_results["control"] == "observed_hidden")]
    pivot = obsr.pivot_table(index="target", columns="window", values="r2", aggfunc="mean")
    plt.figure(figsize=(9, 4))
    plt.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-0.1, vmax=max(0.1, np.nanmax(pivot.to_numpy())))
    plt.colorbar(label="R2")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "hidden_state_neural_target_regression_r2.png", dpi=170)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(behavior_results["model"], behavior_results["r2"], color="#4C78A8")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("CV R2 for log RT")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "behavioral_external_validation_log_rt_r2.png", dpi=170)
    plt.close()


def write_summary(
    neural: pd.DataFrame,
    cpp: pd.DataFrame,
    class_results: pd.DataFrame,
    reg_results: pd.DataFrame,
    behavior_results: pd.DataFrame,
    metadata: pd.DataFrame,
    latents: np.ndarray,
    times_ms: np.ndarray,
    artifacts: Any,
) -> None:
    all_fit = neural[(neural["group_type"] == "all_test_trials") & (neural["group"] == "all")].iloc[0]
    cpp_all = cpp[(cpp["group_type"] == "all") & (cpp["feature"] == "cpp_amp_minus600_to_minus50")].iloc[0]
    obs = class_results[(class_results["split"] == "within_subject") & (class_results["control"] == "observed_hidden")]
    ctrl = (
        class_results[class_results["control"].str.contains("shuffled|dummy", regex=True)]
        .groupby(["target", "window", "split"], as_index=False)["balanced_accuracy"]
        .max()
        .rename(columns={"balanced_accuracy": "control_max_balanced_accuracy"})
    )
    obs2 = obs.merge(ctrl, on=["target", "window", "split"], how="left")
    obs2["margin"] = obs2["balanced_accuracy"] - obs2["control_max_balanced_accuracy"]
    top_class = obs2.sort_values("margin", ascending=False).head(8)
    top_lines = "\n".join(
        f"- {r.target} / {r.window}: balanced accuracy {r.balanced_accuracy:.3f}, control max {r.control_max_balanced_accuracy:.3f}, margin {r.margin:.3f}"
        for r in top_class.itertuples()
    )
    beh_hidden = behavior_results.loc[behavior_results["model"] == "behavior_plus_hidden"].iloc[0]
    beh_base = behavior_results.loc[behavior_results["model"] == "behavior_only"].iloc[0]
    beh_cpp = behavior_results.loc[behavior_results["model"] == "behavior_plus_cpp"].iloc[0]

    checkpoint_used = str(CHECKPOINT_PATH)
    if LATENT_REPORT_PATH.exists():
        checkpoint_used = json.loads(LATENT_REPORT_PATH.read_text()).get("checkpoint_path", checkpoint_used)

    decision = "partial pass"
    if all_fit["corr"] > 0.65 and all_fit["r2"] > 0.25 and obs2["margin"].max() > 0.05:
        decision = "pass, with cautions about behavioral interpretation"
    if all_fit["corr"] < 0.3 or all_fit["r2"] <= 0:
        decision = "fail"

    text = f"""# Neural Model Validation Audit

## Inputs

- Model checkpoint evaluated: `{checkpoint_used}`
- Neural data: `{DATASET_DIR / "eeg_cpp_trials.npy"}`
- Alignment: response-locked only in the current dataset (`alignment` column has {metadata['alignment'].nunique()} value)
- Input features: CP1, CP2, CPz EEG channels
- Model output target: reconstructed current EEG signal, plus future-prediction target during training
- Behavioral metadata available: {", ".join(metadata.columns)}
- Hidden-state tensor shape: {list(latents.shape)}
- Test split size: {len(artifacts.test_indices)} trials
- Time axis: {float(times_ms[0]):.1f} to {float(times_ms[-1]):.1f} ms

## Neural Validation

Held-out test neural reconstruction:

- RMSE: {all_fit['rmse']:.4f}
- R2: {all_fit['r2']:.4f}
- empirical-predicted correlation: {all_fit['corr']:.4f}

CPP amplitude fit across -600 to -50 ms:

- RMSE: {cpp_all['rmse']:.4f}
- R2: {cpp_all['r2']:.4f}
- correlation: {cpp_all['corr']:.4f}

The figures folder contains average trajectory, condition/difficulty/correct-error trajectory, time-resolved fit, residual/scatter, hidden-state decoding, and RT external-validation plots.

## Hidden-State Validation

Strongest within-subject task decoding margins over controls:

{top_lines}

Neural continuous targets are reported in `hidden_state_neural_regression_decoding.csv`.

## Behavioral External Validation

For log RT, behavior-only R2 was {beh_base['r2']:.4f}; behavior + hidden R2 was {beh_hidden['r2']:.4f}; behavior + CPP R2 was {beh_cpp['r2']:.4f}. The incremental comparisons are in `behavioral_external_validation_rt.csv`.

## Recommendation

Conservative decision: **{decision}**.

Use the model for downstream hidden-state analysis only to the extent supported by the neural reconstruction and shuffled-control decoding results. Behavioral claims should be framed as external validation, not as proof that the model itself generates RT or choices.
"""
    (OUT_DIR / "validation_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    global DATASET_DIR, CHECKPOINT_PATH, LATENTS_PATH, LATENT_REPORT_PATH, OUT_DIR, FIG_DIR
    args = _build_parser().parse_args()
    DATASET_DIR = args.dataset_dir.resolve()
    CHECKPOINT_PATH = args.checkpoint_path.resolve()
    LATENTS_PATH = args.latent_path.resolve()
    LATENT_REPORT_PATH = LATENTS_PATH.with_name("latent_extraction_report.json")
    OUT_DIR = args.output_dir.resolve()
    FIG_DIR = OUT_DIR / "figures"
    ensure_dirs()
    eeg, pred, metadata, times_ms, artifacts = load_model_predictions()
    latents, latent_times = load_latents()
    if not np.allclose(times_ms, latent_times):
        raise RuntimeError("Latent and dataset time axes do not match.")
    if latents.shape[0] != eeg.shape[0]:
        raise RuntimeError("Latent trial count does not match EEG trial count.")

    save_json(
        OUT_DIR / "input_manifest.json",
        {
            "checkpoint_path": str(CHECKPOINT_PATH),
            "neural_data_path": str(DATASET_DIR / "eeg_cpp_trials.npy"),
            "metadata_path": str(DATASET_DIR / "metadata.csv"),
            "latent_path": str(LATENTS_PATH),
            "alignment_values": metadata["alignment"].dropna().unique().tolist(),
            "input_channels": channels(),
            "model_output_target": "current EEG reconstruction; future EEG prediction used during training",
            "eeg_shape": list(eeg.shape),
            "hidden_state_shape": list(latents.shape),
            "train_val_test_sizes": {
                "train": int(len(artifacts.train_indices)),
                "val": int(len(artifacts.val_indices)),
                "test": int(len(artifacts.test_indices)),
            },
            "metadata_columns": metadata.columns.tolist(),
        },
    )
    neural = neural_goodness_tables(eeg, pred, metadata, times_ms, artifacts.test_indices)
    cpp = cpp_signature_tables(eeg, pred, metadata, times_ms, artifacts.test_indices)
    save_neural_plots(eeg, pred, metadata, times_ms, artifacts.test_indices)
    class_results, reg_results = hidden_state_validation(latents, times_ms, metadata, eeg, pred)
    behavior_results = behavioral_external_validation(latents, times_ms, metadata, eeg)
    save_hidden_plots(class_results, reg_results, behavior_results)
    write_summary(neural, cpp, class_results, reg_results, behavior_results, metadata, latents, times_ms, artifacts)
    print(f"Neural validation audit complete: {OUT_DIR}")


if __name__ == "__main__":
    main()
