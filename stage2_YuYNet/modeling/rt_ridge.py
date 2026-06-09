from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .utils import ensure_dir, write_json


DEFAULT_WINDOWS: Dict[str, tuple[float, float]] = {
    "full_pre_response": (-600.0, -50.0),
    "early_pre_response": (-600.0, -300.0),
    "mid_pre_response": (-300.0, -120.0),
    "late_pre_response": (-120.0, -50.0),
}

DEFAULT_ALPHAS = np.logspace(-3, 5, 25)


@dataclass(frozen=True)
class RidgeRTConfig:
    windows: Dict[str, tuple[float, float]] | None = None
    n_splits: int = 5
    seed: int = 42
    alphas: Sequence[float] = tuple(float(value) for value in DEFAULT_ALPHAS)
    baseline_columns: tuple[str, ...] = ("subject_id", "difficulty", "correctness")
    targets: tuple[str, ...] = ("log_RT_ms", "RT_ms")


def _load_latents(latent_npz: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    loaded = np.load(latent_npz, allow_pickle=True)
    metadata = pd.DataFrame(loaded["metadata"].item())
    return loaded["Z"], loaded["times_ms"].astype(float), metadata


def _validate_inputs(
    latents: np.ndarray,
    times_ms: np.ndarray,
    metadata: pd.DataFrame,
    dataset_dir: Path | None,
) -> Dict[str, object]:
    if latents.ndim != 3:
        raise ValueError(f"Expected latent array with shape trial x time x dimension, got {latents.shape}.")
    if latents.shape[0] != len(metadata):
        raise ValueError("Latent trial count does not match metadata row count.")
    if latents.shape[1] != len(times_ms):
        raise ValueError("Latent time dimension does not match times_ms length.")
    if "RT_ms" not in metadata.columns:
        raise ValueError("Latent metadata must contain RT_ms for RT regression.")
    if not np.isfinite(metadata["RT_ms"].to_numpy(dtype=float)).all():
        raise ValueError("RT_ms contains missing or non-finite values.")
    if (metadata["RT_ms"].to_numpy(dtype=float) <= 0).any():
        raise ValueError("RT_ms must be positive to compute log_RT_ms.")

    dataset_alignment_checked = False
    if dataset_dir is not None:
        metadata_path = dataset_dir / "metadata.csv"
        times_path = dataset_dir / "times_ms.npy"
        if metadata_path.exists() and times_path.exists():
            source_metadata = pd.read_csv(metadata_path)
            source_times = np.load(times_path)
            if "trial_id" in metadata.columns and "trial_id" in source_metadata.columns:
                if metadata["trial_id"].tolist() != source_metadata["trial_id"].tolist():
                    raise ValueError("Latent metadata trial_id order does not match dataset metadata.")
                dataset_alignment_checked = True
            if not np.array_equal(times_ms.astype(source_times.dtype, copy=False), source_times):
                raise ValueError("Latent times_ms does not match dataset times_ms.")

    finite_latents = np.isfinite(latents)
    report = {
        "n_trials": int(latents.shape[0]),
        "n_timepoints": int(latents.shape[1]),
        "hidden_dim": int(latents.shape[2]),
        "rt_missing_count": int(metadata["RT_ms"].isna().sum()),
        "latent_nonfinite_count": int((~finite_latents).sum()),
        "dataset_alignment_checked": dataset_alignment_checked,
    }
    if report["latent_nonfinite_count"]:
        raise ValueError("Latents contain missing or non-finite values.")
    return report


def _window_features(latents: np.ndarray, times_ms: np.ndarray, start_ms: float, end_ms: float) -> tuple[np.ndarray, int]:
    mask = (times_ms >= start_ms) & (times_ms <= end_ms)
    if int(mask.sum()) < 2:
        raise ValueError(f"Window {start_ms} to {end_ms} ms contains fewer than 2 time points.")
    return latents[:, mask, :].mean(axis=1), int(mask.sum())


def _baseline_design(metadata: pd.DataFrame, baseline_columns: Iterable[str]) -> pd.DataFrame:
    available = [column for column in baseline_columns if column in metadata.columns]
    if not available:
        raise ValueError("None of the requested baseline columns are available in metadata.")
    baseline = metadata.loc[:, available].copy()
    categorical = [column for column in baseline.columns if baseline[column].dtype == object or column == "subject_id"]
    numeric = [column for column in baseline.columns if column not in categorical]
    parts = []
    if numeric:
        parts.append(baseline[numeric].apply(pd.to_numeric, errors="coerce"))
    if categorical:
        parts.append(pd.get_dummies(baseline[categorical].astype(str), drop_first=False, dtype=float))
    design = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=metadata.index)
    if design.isna().any().any():
        design = design.fillna(design.mean(numeric_only=True)).fillna(0.0)
    return design.astype(float)


def _hidden_design(features: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(features, columns=[f"hidden_{idx:02d}" for idx in range(features.shape[1])])


def _fit_predict_outer_cv(
    x: pd.DataFrame,
    y: np.ndarray,
    model_name: str,
    target_name: str,
    window_name: str,
    n_splits: int,
    seed: int,
    alphas: Sequence[float],
    hidden_columns: Sequence[str],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    predictions = []
    beta_rows = []
    alpha_grid = np.asarray(alphas, dtype=float)

    for fold_idx, (train_idx, test_idx) in enumerate(folds.split(x), start=1):
        rng = np.random.default_rng(seed + fold_idx)
        shuffled_train = rng.permutation(train_idx)
        n_alpha_val = max(1, int(round(len(shuffled_train) * 0.2)))
        alpha_val_idx = shuffled_train[:n_alpha_val]
        alpha_train_idx = shuffled_train[n_alpha_val:]
        if len(alpha_train_idx) == 0:
            alpha_train_idx = train_idx
            alpha_val_idx = train_idx

        alpha_scores = []
        for alpha in alpha_grid:
            candidate = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=float(alpha))),
                ]
            )
            candidate.fit(x.iloc[alpha_train_idx], y[alpha_train_idx])
            alpha_pred = candidate.predict(x.iloc[alpha_val_idx])
            alpha_scores.append(mean_squared_error(y[alpha_val_idx], alpha_pred))
        alpha = float(alpha_grid[int(np.argmin(alpha_scores))])

        model = Pipeline(steps=[("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
        model.fit(x.iloc[train_idx], y[train_idx])
        pred = model.predict(x.iloc[test_idx])

        predictions.extend(
            {
                "window": window_name,
                "target": target_name,
                "model": model_name,
                "fold": fold_idx,
                "row_index": int(idx),
                "y_true": float(y[idx]),
                "y_pred": float(value),
                "alpha": alpha,
            }
            for idx, value in zip(test_idx, pred)
        )

        coefficients = model.named_steps["ridge"].coef_
        feature_names = list(x.columns)
        hidden_set = set(hidden_columns)
        for feature_name, coefficient in zip(feature_names, coefficients):
            if feature_name in hidden_set:
                beta_rows.append(
                    {
                        "window": window_name,
                        "target": target_name,
                        "model": model_name,
                        "fold": fold_idx,
                        "feature": feature_name,
                        "coefficient": float(coefficient),
                        "alpha": alpha,
                    }
                )

    predictions_df = pd.DataFrame(predictions)
    beta_df = pd.DataFrame(beta_rows)
    y_true = predictions_df["y_true"].to_numpy(dtype=float)
    y_pred = predictions_df["y_pred"].to_numpy(dtype=float)
    fold_metrics = []
    for fold_idx, fold_df in predictions_df.groupby("fold"):
        fold_y = fold_df["y_true"].to_numpy(dtype=float)
        fold_pred = fold_df["y_pred"].to_numpy(dtype=float)
        fold_metrics.append(
            {
                "fold": int(fold_idx),
                "r2": float(r2_score(fold_y, fold_pred)),
                "rmse": float(np.sqrt(mean_squared_error(fold_y, fold_pred))),
                "mae": float(mean_absolute_error(fold_y, fold_pred)),
            }
        )
    fold_metrics_df = pd.DataFrame(fold_metrics)
    summary = {
        "window": window_name,
        "target": target_name,
        "model": model_name,
        "n_trials": int(len(y)),
        "n_features": int(x.shape[1]),
        "mean_cv_r2": float(fold_metrics_df["r2"].mean()),
        "std_cv_r2": float(fold_metrics_df["r2"].std(ddof=0)),
        "pooled_cv_r2": float(r2_score(y_true, y_pred)),
        "mean_cv_rmse": float(fold_metrics_df["rmse"].mean()),
        "mean_cv_mae": float(fold_metrics_df["mae"].mean()),
        "mean_alpha": float(predictions_df.groupby("fold")["alpha"].first().mean()),
    }
    return summary, predictions_df, beta_df


def _summarize_beta_stability(beta: pd.DataFrame) -> pd.DataFrame:
    if beta.empty:
        return pd.DataFrame(
            columns=[
                "window",
                "target",
                "model",
                "feature",
                "mean_coefficient",
                "std_coefficient",
                "abs_mean_coefficient",
                "sign_consistency",
            ]
        )
    grouped = beta.groupby(["window", "target", "model", "feature"], as_index=False)
    rows = []
    for keys, values in grouped:
        coeff = values["coefficient"].to_numpy(dtype=float)
        signs = np.sign(coeff)
        positive = float(np.mean(signs > 0))
        negative = float(np.mean(signs < 0))
        rows.append(
            {
                "window": keys[0],
                "target": keys[1],
                "model": keys[2],
                "feature": keys[3],
                "mean_coefficient": float(np.mean(coeff)),
                "std_coefficient": float(np.std(coeff)),
                "abs_mean_coefficient": float(abs(np.mean(coeff))),
                "sign_consistency": max(positive, negative),
            }
        )
    return pd.DataFrame(rows)


def _summarize_model_deltas(performance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (window, target), group in performance.groupby(["window", "target"]):
        lookup = group.set_index("model")
        baseline = float(lookup.loc["baseline", "mean_cv_r2"]) if "baseline" in lookup.index else float("nan")
        hidden = float(lookup.loc["hidden", "mean_cv_r2"]) if "hidden" in lookup.index else float("nan")
        incremental = float(lookup.loc["baseline_plus_hidden", "mean_cv_r2"]) if "baseline_plus_hidden" in lookup.index else float("nan")
        shuffled = (
            float(lookup.loc["baseline_plus_shuffled_hidden", "mean_cv_r2"])
            if "baseline_plus_shuffled_hidden" in lookup.index
            else float("nan")
        )
        rows.append(
            {
                "window": window,
                "target": target,
                "baseline_r2": baseline,
                "hidden_r2": hidden,
                "baseline_plus_hidden_r2": incremental,
                "baseline_plus_shuffled_hidden_r2": shuffled,
                "hidden_minus_baseline_r2": hidden - baseline,
                "incremental_minus_baseline_r2": incremental - baseline,
                "incremental_minus_shuffled_r2": incremental - shuffled,
            }
        )
    return pd.DataFrame(rows)


def _save_performance_figure(output_dir: Path, deltas: pd.DataFrame) -> None:
    plot_data = deltas[deltas["target"] == "log_RT_ms"].copy()
    if plot_data.empty:
        return
    x = np.arange(len(plot_data))
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.4)
    axis.bar(x - 0.18, plot_data["incremental_minus_baseline_r2"], width=0.36, label="hidden added to baseline")
    axis.bar(x + 0.18, plot_data["incremental_minus_shuffled_r2"], width=0.36, label="real hidden vs shuffled")
    axis.set_xticks(x)
    axis.set_xticklabels(plot_data["window"], rotation=25, ha="right")
    axis.set_ylabel("mean CV R2 difference")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "ridge_rt_window_deltas.png", dpi=150)
    plt.close(fig)


def run_ridge_rt_analysis(
    latent_npz: Path,
    output_dir: Path,
    dataset_dir: Path | None = None,
    config: RidgeRTConfig | None = None,
) -> Dict[str, object]:
    """Evaluate response-time prediction from time-averaged hidden states with Ridge regression."""
    config = config or RidgeRTConfig()
    output_dir = ensure_dir(output_dir)
    latents, times_ms, metadata = _load_latents(latent_npz)
    quality = _validate_inputs(latents, times_ms, metadata, dataset_dir)

    metadata = metadata.copy()
    metadata["RT_ms"] = pd.to_numeric(metadata["RT_ms"], errors="raise")
    metadata["log_RT_ms"] = np.log(metadata["RT_ms"].to_numpy(dtype=float))
    baseline = _baseline_design(metadata, config.baseline_columns)

    windows = config.windows or DEFAULT_WINDOWS
    rng = np.random.default_rng(config.seed)
    performance_rows = []
    prediction_tables = []
    beta_tables = []
    feature_quality_rows = []

    for window_name, (start_ms, end_ms) in windows.items():
        features, n_timepoints = _window_features(latents, times_ms, start_ms, end_ms)
        hidden = _hidden_design(features)
        hidden_columns = list(hidden.columns)
        shuffled_indices = rng.permutation(features.shape[0])
        shuffled_hidden = _hidden_design(features[shuffled_indices]).add_prefix("shuffled_")

        variances = hidden.var(axis=0).to_numpy(dtype=float)
        feature_quality_rows.append(
            {
                "window": window_name,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "n_timepoints": n_timepoints,
                "hidden_dim": int(features.shape[1]),
                "near_zero_variance_dimensions": int(np.sum(variances < 1e-10)),
                "min_feature_variance": float(np.min(variances)),
                "median_feature_variance": float(np.median(variances)),
                "max_feature_variance": float(np.max(variances)),
            }
        )

        designs = {
            "baseline": baseline,
            "hidden": hidden,
            "baseline_plus_hidden": pd.concat([baseline.reset_index(drop=True), hidden], axis=1),
            "baseline_plus_shuffled_hidden": pd.concat([baseline.reset_index(drop=True), shuffled_hidden], axis=1),
        }
        model_hidden_columns = {
            "baseline": [],
            "hidden": hidden_columns,
            "baseline_plus_hidden": hidden_columns,
            "baseline_plus_shuffled_hidden": list(shuffled_hidden.columns),
        }

        for target_name in config.targets:
            y = metadata[target_name].to_numpy(dtype=float)
            for model_name, design in designs.items():
                summary, predictions, beta = _fit_predict_outer_cv(
                    design,
                    y,
                    model_name=model_name,
                    target_name=target_name,
                    window_name=window_name,
                    n_splits=config.n_splits,
                    seed=config.seed,
                    alphas=config.alphas,
                    hidden_columns=model_hidden_columns[model_name],
                )
                performance_rows.append(summary)
                prediction_tables.append(predictions)
                beta_tables.append(beta)

    performance = pd.DataFrame(performance_rows)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    beta = pd.concat(beta_tables, ignore_index=True) if beta_tables else pd.DataFrame()
    beta_stability = _summarize_beta_stability(beta)
    deltas = _summarize_model_deltas(performance)
    feature_quality = pd.DataFrame(feature_quality_rows)

    performance.to_csv(output_dir / "ridge_rt_model_performance.csv", index=False)
    predictions.to_csv(output_dir / "ridge_rt_predictions.csv", index=False)
    beta.to_csv(output_dir / "ridge_rt_hidden_coefficients_by_fold.csv", index=False)
    beta_stability.to_csv(output_dir / "ridge_rt_beta_stability.csv", index=False)
    deltas.to_csv(output_dir / "ridge_rt_model_deltas.csv", index=False)
    feature_quality.to_csv(output_dir / "ridge_rt_feature_quality.csv", index=False)
    _save_performance_figure(output_dir, deltas)

    log_deltas = deltas[deltas["target"] == "log_RT_ms"].copy()
    best_window = None
    if not log_deltas.empty:
        best_row = log_deltas.sort_values("incremental_minus_baseline_r2", ascending=False).iloc[0]
        best_window = {
            "window": str(best_row["window"]),
            "incremental_minus_baseline_r2": float(best_row["incremental_minus_baseline_r2"]),
            "incremental_minus_shuffled_r2": float(best_row["incremental_minus_shuffled_r2"]),
        }

    report = {
        "passed": True,
        "latent_path": str(latent_npz),
        "dataset_dir": str(dataset_dir) if dataset_dir is not None else None,
        "quality": quality,
        "windows": {name: [float(bounds[0]), float(bounds[1])] for name, bounds in windows.items()},
        "targets": list(config.targets),
        "baseline_columns": [column for column in config.baseline_columns if column in metadata.columns],
        "n_splits": int(config.n_splits),
        "alpha_grid": [float(alpha) for alpha in config.alphas],
        "best_log_rt_window": best_window,
        "generated_files": [
            "ridge_rt_model_performance.csv",
            "ridge_rt_predictions.csv",
            "ridge_rt_hidden_coefficients_by_fold.csv",
            "ridge_rt_beta_stability.csv",
            "ridge_rt_model_deltas.csv",
            "ridge_rt_feature_quality.csv",
            "ridge_rt_window_deltas.png",
        ],
    }
    write_json(output_dir / "ridge_rt_analysis_report.json", report)
    return report
