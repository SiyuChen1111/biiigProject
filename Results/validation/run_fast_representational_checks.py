from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
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
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "Data" / "ProcessedData"
OUT_DIR = REPO_ROOT / "Results" / "validation"
FIG_DIR = OUT_DIR / "figures"
LATENTS_PATH = REPO_ROOT / "Data" / "IntermediateData" / "latents_full" / "latents_full.npz"

WINDOWS = {
    "minus600_to_minus300": (-600.0, -300.0),
    "minus300_to_minus120": (-300.0, -120.0),
    "minus120_to_minus50": (-120.0, -50.0),
    "minus600_to_minus50": (-600.0, -50.0),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fast representational checks.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--latent-path", type=Path, default=LATENTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(pearsonr(a, b).statistic)


def cpp_features(signal: np.ndarray, times_ms: np.ndarray) -> pd.DataFrame:
    cpp = signal.mean(axis=2)
    rows = {}
    for name, (lo, hi) in WINDOWS.items():
        mask = (times_ms >= lo) & (times_ms <= hi)
        rows[f"cpp_amp_{name}"] = cpp[:, mask].mean(axis=1)
        x = times_ms[mask].astype(float)
        x = x - x.mean()
        rows[f"cpp_slope_{name}"] = ((cpp[:, mask] - cpp[:, mask].mean(axis=1, keepdims=True)) @ x) / np.sum(x**2)
    return pd.DataFrame(rows)


def hidden_features(latents: np.ndarray, times_ms: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for name, (lo, hi) in WINDOWS.items():
        mask = (times_ms >= lo) & (times_ms <= hi)
        out[name] = latents[:, mask, :].mean(axis=1)
    return out


def shuffle_within_subject(X: np.ndarray, subjects: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = X.copy()
    for s in np.unique(subjects):
        idx = np.flatnonzero(subjects == s)
        out[idx] = out[rng.permutation(idx)]
    return out


def balanced_sample(meta: pd.DataFrame, max_n: int = 2400) -> np.ndarray:
    rng = np.random.default_rng(2026)
    parts = []
    per_subject = max(20, max_n // meta["subject_id"].nunique())
    for _, group in meta.groupby("subject_id"):
        idx = group.index.to_numpy()
        take = min(len(idx), per_subject)
        parts.append(rng.choice(idx, size=take, replace=False))
    idx = np.concatenate(parts)
    if len(idx) > max_n:
        idx = rng.choice(idx, size=max_n, replace=False)
    return np.sort(idx)


def class_cv(X: np.ndarray, y: np.ndarray, control: str) -> dict[str, float]:
    rows = []
    rng = np.random.default_rng(33)
    classes = np.unique(y)
    folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=33)
    for train, test in folds.split(X, y):
        clf = DummyClassifier(strategy="most_frequent") if control == "dummy_majority" else make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=250, class_weight="balanced", solver="liblinear")
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
                pass
        rows.append(
            {
                "accuracy": accuracy_score(y[test], pred),
                "balanced_accuracy": balanced_accuracy_score(y[test], pred),
                "auc": auc,
            }
        )
    frame = pd.DataFrame(rows)
    return {
        "accuracy": frame["accuracy"].mean(),
        "balanced_accuracy": frame["balanced_accuracy"].mean(),
        "auc": frame["auc"].mean(skipna=True),
        "balanced_accuracy_sd": frame["balanced_accuracy"].std(ddof=1),
    }


def reg_cv(X: np.ndarray, y: np.ndarray, control: str) -> dict[str, float]:
    rows = []
    rng = np.random.default_rng(44)
    folds = KFold(n_splits=4, shuffle=True, random_state=44)
    for train, test in folds.split(X):
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        y_train = rng.permutation(y[train]) if control == "shuffled_label" else y[train]
        model.fit(X[train], y_train)
        pred = model.predict(X[test])
        rows.append(
            {
                "r2": r2_score(y[test], pred),
                "corr": corr(y[test], pred),
                "mae": mean_absolute_error(y[test], pred),
                "rmse": float(np.sqrt(mean_squared_error(y[test], pred))),
            }
        )
    frame = pd.DataFrame(rows)
    return {
        "r2": frame["r2"].mean(),
        "corr": frame["corr"].mean(),
        "mae": frame["mae"].mean(),
        "rmse": frame["rmse"].mean(),
        "r2_sd": frame["r2"].std(ddof=1),
    }


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    meta = pd.read_csv(DATASET_DIR / "metadata.csv")
    eeg = np.load(DATASET_DIR / "eeg_cpp_trials.npy")
    lat = np.load(LATENTS_PATH, allow_pickle=True)
    Z = lat["Z"]
    times_ms = lat["times_ms"]
    sample = balanced_sample(meta)
    meta_s = meta.iloc[sample].reset_index(drop=True)
    subjects = meta_s["subject_id"].astype(str).to_numpy()
    Zs = Z[sample]
    eeg_s = eeg[sample]
    X_by_window = hidden_features(Zs, times_ms)
    cpp = cpp_features(eeg_s, times_ms)
    meta_s["rt_bin"] = pd.qcut(meta_s["RT_ms"], 3, labels=["fast", "medium", "slow"])

    class_targets = {
        "choice": meta_s["choice"].to_numpy(),
        "condition": meta_s["condition"].to_numpy(),
        "rt_bin": meta_s["rt_bin"].astype(str).to_numpy(),
        "correctness": meta_s["correctness"].astype(int).to_numpy(),
    }
    class_rows = []
    for window, X in X_by_window.items():
        Xsh = shuffle_within_subject(X, subjects, 12)
        for target, y in class_targets.items():
            for control, Xuse in [
                ("observed_hidden", X),
                ("shuffled_hidden_within_subject", Xsh),
                ("shuffled_label", X),
                ("dummy_majority", X),
            ]:
                class_rows.append(
                    {
                        "target": target,
                        "window": window,
                        "split": "sampled_trial_level_subject_balanced",
                        "control": control,
                        "n_trials": len(sample),
                        **class_cv(Xuse, y, control),
                    }
                )
    class_out = pd.DataFrame(class_rows)
    class_out.to_csv(OUT_DIR / "hidden_state_classification_decoding_fast.csv", index=False)

    reg_targets = {
        "cpp_amp_minus600_to_minus50": cpp["cpp_amp_minus600_to_minus50"].to_numpy(),
        "cpp_slope_minus600_to_minus50": cpp["cpp_slope_minus600_to_minus50"].to_numpy(),
        "log_RT_ms": np.log(meta_s["RT_ms"].to_numpy()),
    }
    reg_rows = []
    for window, X in X_by_window.items():
        Xsh = shuffle_within_subject(X, subjects, 13)
        for target, y in reg_targets.items():
            for control, Xuse in [("observed_hidden", X), ("shuffled_hidden_within_subject", Xsh), ("shuffled_label", X)]:
                reg_rows.append(
                    {
                        "target": target,
                        "window": window,
                        "split": "sampled_trial_level_subject_balanced",
                        "control": control,
                        "n_trials": len(sample),
                        **reg_cv(Xuse, y, control),
                    }
                )
    reg_out = pd.DataFrame(reg_rows)
    reg_out.to_csv(OUT_DIR / "hidden_state_regression_decoding_fast.csv", index=False)

    beh = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit_transform(
        meta_s[["choice", "correctness", "condition", "difficulty", "probe_leftrightwin", "response_hand"]].astype(str)
    )
    Xh = X_by_window["minus600_to_minus50"]
    Xhs = shuffle_within_subject(Xh, subjects, 14)
    yrt = np.log(meta_s["RT_ms"].to_numpy())
    feature_sets = {
        "behavior_only": beh,
        "cpp_features_only": cpp.to_numpy(),
        "hidden_states_only": Xh,
        "behavior_plus_cpp": np.column_stack([beh, cpp.to_numpy()]),
        "behavior_plus_hidden": np.column_stack([beh, Xh]),
        "behavior_plus_cpp_plus_hidden": np.column_stack([beh, cpp.to_numpy(), Xh]),
        "behavior_plus_shuffled_hidden": np.column_stack([beh, Xhs]),
    }
    beh_rows = []
    for name, X in feature_sets.items():
        beh_rows.append({"model": name, "target": "log_RT_ms", **reg_cv(X, yrt, "observed")})
    beh_out = pd.DataFrame(beh_rows)
    base = beh_out.loc[beh_out["model"] == "behavior_only", "r2"].iloc[0]
    beh_cpp = beh_out.loc[beh_out["model"] == "behavior_plus_cpp", "r2"].iloc[0]
    beh_out["delta_r2_vs_behavior_only"] = beh_out["r2"] - base
    beh_out["delta_r2_vs_behavior_plus_cpp"] = beh_out["r2"] - beh_cpp
    beh_out.to_csv(OUT_DIR / "behavioral_external_validation_rt_fast.csv", index=False)

    obs = class_out[(class_out["control"] == "observed_hidden")]
    pivot = obs.pivot_table(index="target", columns="window", values="balanced_accuracy")
    plt.figure(figsize=(8, 4.5))
    plt.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    plt.colorbar(label="Balanced accuracy")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "hidden_state_task_decoding_fast.png", dpi=170)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(beh_out["model"], beh_out["r2"], color="#4C78A8")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("CV R2 for log RT")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "behavioral_external_validation_log_rt_fast.png", dpi=170)
    plt.close()
    print("Fast representational checks complete.")


if __name__ == "__main__":
    args = _build_parser().parse_args()
    DATASET_DIR = args.dataset_dir.resolve()
    LATENTS_PATH = args.latent_path.resolve()
    OUT_DIR = args.output_dir.resolve()
    FIG_DIR = OUT_DIR / "figures"
    main()
