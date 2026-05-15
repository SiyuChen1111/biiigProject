from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from .config import AnalysisConfig
from .utils import ensure_dir, write_json


WINDOWS: Dict[str, Tuple[float, float]] = {
    "early_pre_response": (-600.0, -300.0),
    "mid_pre_response": (-300.0, -120.0),
    "late_pre_response": (-120.0, -50.0),
    "peri_response_contaminated": (-50.0, 100.0),
}
PRE_RESPONSE_WINDOWS = ("early_pre_response", "mid_pre_response", "late_pre_response")
SIGNIFICANCE_COMPARISONS = (
    ("late_pre_response", "early_pre_response"),
    ("late_pre_response", "mid_pre_response"),
    ("mid_pre_response", "early_pre_response"),
)
SIGNIFICANCE_METRICS = ("pc1_explained", "participation_ratio")
CONTAMINATION_NOTE = (
    "The -50 to 100 ms window is response/motor contaminated and is not interpreted "
    "as pure evidence accumulation."
)


def _participation_ratio(eigenvalues: np.ndarray) -> float:
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    numerator = float(np.sum(eigenvalues) ** 2)
    denominator = float(np.sum(eigenvalues ** 2))
    return numerator / denominator if denominator else 0.0


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def _safe_r2(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return float("nan")
    model = LinearRegression()
    model.fit(x[valid, None], y[valid])
    return float(r2_score(y[valid], model.predict(x[valid, None])))


def _fit_global_pca(latents: np.ndarray, n_components: int) -> Tuple[PCA, np.ndarray]:
    flattened = latents.reshape(-1, latents.shape[-1]).astype(np.float64)
    pca = PCA(n_components=min(n_components, flattened.shape[0], flattened.shape[1]))
    transformed = pca.fit_transform(flattened).reshape(latents.shape[0], latents.shape[1], pca.n_components_)
    if not np.isfinite(transformed).all():
        transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
    return pca, transformed


def _pca_explained(snapshot: np.ndarray, n_components: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    snapshot = np.asarray(snapshot, dtype=np.float64)
    max_components = min(snapshot.shape[0], snapshot.shape[1])
    if n_components is not None:
        max_components = min(max_components, n_components)
    pca = PCA(n_components=max_components)
    pca.fit(snapshot)
    return pca.explained_variance_ratio_, pca.explained_variance_


def _pca_metrics_from_snapshot(snapshot: np.ndarray) -> Dict[str, float]:
    snapshot = np.asarray(snapshot, dtype=np.float64)
    if snapshot.shape[0] < 2 or snapshot.shape[1] < 1:
        return {"pc1_explained": float("nan"), "participation_ratio": float("nan")}
    centered = snapshot - snapshot.mean(axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) == 0:
        return {"pc1_explained": float("nan"), "participation_ratio": float("nan")}
    explained = eigenvalues / eigenvalues.sum()
    return {
        "pc1_explained": float(explained[0]),
        "participation_ratio": _participation_ratio(eigenvalues),
    }


def _compute_time_resolved_pca(latents: np.ndarray, times_ms: np.ndarray) -> pd.DataFrame:
    rows = []
    for time_idx, time_ms in enumerate(times_ms):
        snapshot = latents[:, time_idx, :]
        explained_ratio, eigenvalues = _pca_explained(snapshot)
        cumulative = np.cumsum(explained_ratio)
        pc1 = float(explained_ratio[0]) if len(explained_ratio) else float("nan")
        pc2 = float(explained_ratio[1]) if len(explained_ratio) > 1 else 0.0
        rows.append(
            {
                "time_index": int(time_idx),
                "time_ms": float(time_ms),
                "pc1_explained": pc1,
                "pc2_explained": pc2,
                "pc1_pc2_cumulative": float(cumulative[min(1, len(cumulative) - 1)]) if len(cumulative) else float("nan"),
                "n_pc_80": int(np.searchsorted(cumulative, 0.80) + 1) if len(cumulative) else 0,
                "n_pc_90": int(np.searchsorted(cumulative, 0.90) + 1) if len(cumulative) else 0,
                "participation_ratio": _participation_ratio(eigenvalues),
            }
        )
    return pd.DataFrame(rows)


def _window_mask(times_ms: np.ndarray, start_ms: float, end_ms: float) -> np.ndarray:
    return (times_ms >= start_ms) & (times_ms <= end_ms)


def _windowed_pca_summary(latents: np.ndarray, times_ms: np.ndarray, n_components: int) -> pd.DataFrame:
    rows = []
    summary_components = max(n_components, 4)
    for label, (start_ms, end_ms) in WINDOWS.items():
        mask = _window_mask(times_ms, start_ms, end_ms)
        if mask.sum() < 2:
            continue
        snapshot = latents[:, mask, :].reshape(-1, latents.shape[-1])
        explained_ratio, _ = _pca_explained(snapshot, summary_components)
        _, full_eigenvalues = _pca_explained(snapshot)
        cumulative = np.cumsum(explained_ratio)
        rows.append(
            {
                "window": label,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "n_timepoints": int(mask.sum()),
                "n_observations": int(snapshot.shape[0]),
                "pc1_explained": float(explained_ratio[0]) if len(explained_ratio) else float("nan"),
                "pc2_explained": float(explained_ratio[1]) if len(explained_ratio) > 1 else float("nan"),
                "pc3_explained": float(explained_ratio[2]) if len(explained_ratio) > 2 else float("nan"),
                "pc4_explained": float(explained_ratio[3]) if len(explained_ratio) > 3 else float("nan"),
                "pc1_pc2_cumulative": float(cumulative[min(1, len(cumulative) - 1)]) if len(cumulative) else float("nan"),
                "participation_ratio": _participation_ratio(full_eigenvalues),
                "interpretation_guardrail": CONTAMINATION_NOTE if label == "peri_response_contaminated" else "",
            }
        )
    return pd.DataFrame(rows)


def _metadata_from_npz(loaded: np.lib.npyio.NpzFile) -> pd.DataFrame:
    metadata = loaded["metadata"].item()
    return pd.DataFrame(metadata)


def _load_latent_export(latent_npz: Path) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    loaded = np.load(latent_npz, allow_pickle=True)
    return loaded["Z"], loaded["times_ms"], _metadata_from_npz(loaded)


def _load_pooled_latents(latent_paths: Sequence[Path]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    if not latent_paths:
        raise ValueError("At least one latent path is required.")

    latent_blocks = []
    metadata_blocks = []
    reference_times = None
    included_splits = []
    for latent_path in latent_paths:
        latents, times_ms, metadata = _load_latent_export(latent_path)
        if reference_times is None:
            reference_times = times_ms
        elif not np.allclose(reference_times, times_ms):
            raise ValueError(f"Latent time axis does not match reference: {latent_path}")
        metadata = metadata.copy()
        split_name = latent_path.stem.replace("latents_", "")
        metadata["latent_split"] = split_name
        latent_blocks.append(latents)
        metadata_blocks.append(metadata)
        included_splits.append(split_name)

    combined_metadata = pd.concat(metadata_blocks, ignore_index=True)
    if "trial_id" not in combined_metadata.columns:
        raise ValueError("Pooled latent metadata must contain trial_id.")
    duplicated = combined_metadata.loc[combined_metadata["trial_id"].duplicated(), "trial_id"].tolist()
    if duplicated:
        raise ValueError(f"Pooled latent exports contain duplicate trial_id values: {duplicated[:10]}")

    return np.concatenate(latent_blocks, axis=0), reference_times, combined_metadata, included_splits


def _load_best_model_summary(output_dir: Path) -> Dict[str, object]:
    summary_path = output_dir.parent / "best_run_summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    keys = [
        "run_id",
        "source_short_run",
        "checkpoint_path",
        "corr",
        "slope_corr",
        "amp_ratio",
        "mse",
        "late_err",
        "param_hidden_dim",
        "param_projection_dim",
        "param_lambda_cpp_prior",
        "param_lambda_late_amplitude",
        "param_lambda_cpp_mean_alignment",
        "param_lambda_slope_floor",
        "param_slope_floor_ratio",
    ]
    return {key: summary[key] for key in keys if key in summary}


def _load_aligned_cpp(
    latent_metadata: pd.DataFrame, dataset_dir: Path | None
) -> Tuple[np.ndarray | None, np.ndarray | None, str | None, Dict[str, int]]:
    if dataset_dir is None:
        return None, None, "dataset_dir was not provided; CPP proxy mapping was skipped.", {}

    eeg_path = dataset_dir / "eeg_cpp_trials.npy"
    metadata_path = dataset_dir / "metadata.csv"
    if not eeg_path.exists() or not metadata_path.exists():
        return None, None, f"Missing dataset files under {dataset_dir}; CPP proxy mapping was skipped.", {}

    if "trial_id" not in latent_metadata.columns:
        raise ValueError("Latent metadata must contain trial_id to align test trials with raw EEG.")

    eeg = np.load(eeg_path)
    dataset_metadata = pd.read_csv(metadata_path)
    if "trial_id" not in dataset_metadata.columns:
        raise ValueError("Dataset metadata must contain trial_id to align test trials with raw EEG.")

    trial_to_index = {}
    for idx, trial_id in enumerate(dataset_metadata["trial_id"].tolist()):
        if trial_id in trial_to_index:
            raise ValueError(f"Duplicate trial_id in dataset metadata: {trial_id}")
        trial_to_index[trial_id] = idx

    indices = []
    missing = []
    for trial_id in latent_metadata["trial_id"].tolist():
        if trial_id in trial_to_index:
            indices.append(trial_to_index[trial_id])
        else:
            missing.append(trial_id)
    if missing:
        raise ValueError(f"Could not align latent trial_id values with raw EEG metadata: {missing[:10]}")

    aligned = eeg[np.asarray(indices, dtype=int)]
    if aligned.shape[0] != len(latent_metadata):
        raise ValueError("Aligned EEG trial count does not match latent trial count.")
    eeg_nonfinite_count = int((~np.isfinite(aligned)).sum())
    aligned_for_control = _impute_nonfinite_by_time_channel_mean(aligned) if eeg_nonfinite_count else aligned.astype(np.float32)
    finite = np.isfinite(aligned)
    channel_counts = finite.sum(axis=2)
    channel_sums = np.where(finite, aligned, 0.0).sum(axis=2)
    cpp_trials = np.divide(
        channel_sums,
        channel_counts,
        out=np.full(channel_sums.shape, np.nan, dtype=np.float64),
        where=channel_counts > 0,
    )
    nonfinite_count = int((~np.isfinite(cpp_trials)).sum())
    if nonfinite_count:
        cpp_trials = _impute_nonfinite_by_time_mean(cpp_trials)
    quality = {
        "aligned_trials": int(cpp_trials.shape[0]),
        "raw_eeg_nonfinite_values_imputed": eeg_nonfinite_count,
        "cpp_nonfinite_values_imputed": nonfinite_count,
    }
    return cpp_trials, aligned_for_control, None, quality


def _impute_nonfinite_by_time_mean(values: np.ndarray) -> np.ndarray:
    imputed = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(imputed)
    column_means = np.nanmean(np.where(finite, imputed, np.nan), axis=0)
    global_mean = float(np.nanmean(imputed)) if np.isfinite(np.nanmean(imputed)) else 0.0
    column_means = np.where(np.isfinite(column_means), column_means, global_mean)
    bad_rows, bad_cols = np.where(~finite)
    imputed[bad_rows, bad_cols] = column_means[bad_cols]
    return imputed.astype(np.float32)


def _impute_nonfinite_by_time_channel_mean(values: np.ndarray) -> np.ndarray:
    imputed = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(imputed)
    means = np.nanmean(np.where(finite, imputed, np.nan), axis=0)
    global_mean = float(np.nanmean(imputed)) if np.isfinite(np.nanmean(imputed)) else 0.0
    means = np.where(np.isfinite(means), means, global_mean)
    bad_trials, bad_times, bad_channels = np.where(~finite)
    imputed[bad_trials, bad_times, bad_channels] = means[bad_times, bad_channels]
    return imputed.astype(np.float32)


def _compute_cpp_proxies(cpp_trials: np.ndarray, times_ms: np.ndarray) -> pd.DataFrame:
    pre_mask = _window_mask(times_ms, -600.0, -50.0)
    late_mask = _window_mask(times_ms, -120.0, -50.0)
    if pre_mask.sum() < 2 or late_mask.sum() < 2:
        raise ValueError("CPP proxy windows do not contain enough time points.")

    rows = []
    pre_times = times_ms[pre_mask].astype(float)
    for trial_idx, curve in enumerate(cpp_trials):
        pre_curve = curve[pre_mask].astype(float)
        slope = np.polyfit(pre_times, pre_curve, 1)[0]
        rows.append(
            {
                "latent_trial_index": int(trial_idx),
                "cpp_late_amplitude": float(np.mean(curve[late_mask])),
                "cpp_pre_response_slope": float(slope),
                "cpp_pre_response_auc": float(np.trapezoid(pre_curve, pre_times)),
            }
        )
    return pd.DataFrame(rows)


def _summarize_pc_cpp_mapping(
    scores: np.ndarray,
    cpp_proxies: pd.DataFrame,
    times_ms: np.ndarray,
    n_components: int,
) -> pd.DataFrame:
    summary_rows = []
    score_windows = {
        "pc_late_score": _window_mask(times_ms, -120.0, -50.0),
        "pc_pre_response_mean_score": _window_mask(times_ms, -600.0, -50.0),
    }
    targets = ("cpp_late_amplitude", "cpp_pre_response_slope", "cpp_pre_response_auc")
    for pc_idx in range(min(n_components, scores.shape[-1])):
        for score_name, mask in score_windows.items():
            if mask.sum() < 2:
                continue
            pc_score = scores[:, mask, pc_idx].mean(axis=1)
            for target in targets:
                target_values = cpp_proxies[target].to_numpy(dtype=float)
                summary_rows.append(
                    {
                        "pc": f"PC{pc_idx + 1}",
                        "score_window": score_name,
                        "target": target,
                        "pearson_r": _safe_corr(pc_score, target_values),
                        "linear_r2": _safe_r2(pc_score, target_values),
                    }
                )
    return pd.DataFrame(summary_rows)


def _shuffle_time(latents: np.ndarray, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled = latents.copy()
    for trial_idx in range(shuffled.shape[0]):
        shuffled[trial_idx] = shuffled[trial_idx, rng.permutation(shuffled.shape[1]), :]
    return shuffled


def _shuffle_trials_within_time(latents: np.ndarray, seed: int = 43) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled = latents.copy()
    for time_idx in range(shuffled.shape[1]):
        for dim_idx in range(shuffled.shape[2]):
            shuffled[:, time_idx, dim_idx] = shuffled[rng.permutation(shuffled.shape[0]), time_idx, dim_idx]
    return shuffled


def _mean_metric_by_windows(time_pca: pd.DataFrame, metric: str, windows: Iterable[str]) -> Dict[str, float]:
    values = {}
    for window in windows:
        start_ms, end_ms = WINDOWS[window]
        mask = (time_pca["time_ms"] >= start_ms) & (time_pca["time_ms"] <= end_ms)
        values[window] = float(time_pca.loc[mask, metric].mean())
    return values


def _control_summary(latents: np.ndarray, times_ms: np.ndarray, raw_eeg: np.ndarray | None) -> Tuple[pd.DataFrame, pd.DataFrame | None]:
    rows = []
    control_inputs = {
        "latent_observed": latents,
        "latent_time_shuffle": _shuffle_time(latents),
        "latent_trial_shuffle": _shuffle_trials_within_time(latents),
    }
    for label, data in control_inputs.items():
        time_pca = _compute_time_resolved_pca(data, times_ms)
        pc1_values = _mean_metric_by_windows(time_pca, "pc1_explained", WINDOWS.keys())
        pr_values = _mean_metric_by_windows(time_pca, "participation_ratio", WINDOWS.keys())
        for window in WINDOWS:
            rows.append(
                {
                    "analysis": label,
                    "window": window,
                    "mean_pc1_explained": pc1_values[window],
                    "mean_participation_ratio": pr_values[window],
                }
            )

    raw_time_pca = None
    if raw_eeg is not None:
        raw_time_pca = _compute_time_resolved_pca(raw_eeg, times_ms)
        pc1_values = _mean_metric_by_windows(raw_time_pca, "pc1_explained", WINDOWS.keys())
        pr_values = _mean_metric_by_windows(raw_time_pca, "participation_ratio", WINDOWS.keys())
        for window in WINDOWS:
            rows.append(
                {
                    "analysis": "raw_cpp_observed",
                    "window": window,
                    "mean_pc1_explained": pc1_values[window],
                    "mean_participation_ratio": pr_values[window],
                }
            )
    return pd.DataFrame(rows), raw_time_pca


def _window_metric_table_for_indices(latents: np.ndarray, times_ms: np.ndarray, trial_indices: np.ndarray) -> Dict[str, Dict[str, float]]:
    table = {}
    for window, (start_ms, end_ms) in WINDOWS.items():
        if window == "peri_response_contaminated":
            continue
        mask = _window_mask(times_ms, start_ms, end_ms)
        if mask.sum() < 2:
            table[window] = {"pc1_explained": float("nan"), "participation_ratio": float("nan")}
            continue
        snapshot = latents[trial_indices][:, mask, :].reshape(-1, latents.shape[-1])
        table[window] = _pca_metrics_from_snapshot(snapshot)
    return table


def _difference_rows(metric_table: Dict[str, Dict[str, float]]) -> list[Dict[str, object]]:
    rows = []
    for minuend, subtrahend in SIGNIFICANCE_COMPARISONS:
        for metric in SIGNIFICANCE_METRICS:
            rows.append(
                {
                    "comparison": f"{minuend}_minus_{subtrahend}",
                    "metric": metric,
                    "difference": float(metric_table[minuend][metric] - metric_table[subtrahend][metric]),
                }
            )
    return rows


def _bootstrap_effects(
    latents: np.ndarray,
    times_ms: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 123,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    n_trials = latents.shape[0]
    for sample_idx in range(n_bootstrap):
        trial_indices = rng.integers(0, n_trials, size=n_trials)
        metric_table = _window_metric_table_for_indices(latents, times_ms, trial_indices)
        for row in _difference_rows(metric_table):
            row["sample_index"] = sample_idx
            rows.append(row)
    return pd.DataFrame(rows)


def _permuted_window_metric_table(latents: np.ndarray, times_ms: np.ndarray, rng: np.random.Generator) -> Dict[str, Dict[str, float]]:
    source_masks = {window: _window_mask(times_ms, *WINDOWS[window]) for window in PRE_RESPONSE_WINDOWS}
    if any(mask.sum() < 2 for mask in source_masks.values()):
        return {window: {"pc1_explained": float("nan"), "participation_ratio": float("nan")} for window in PRE_RESPONSE_WINDOWS}
    assigned_segments = {window: [] for window in PRE_RESPONSE_WINDOWS}
    for trial_idx in range(latents.shape[0]):
        shuffled_sources = rng.permutation(PRE_RESPONSE_WINDOWS)
        for target_window, source_window in zip(PRE_RESPONSE_WINDOWS, shuffled_sources):
            assigned_segments[target_window].append(latents[trial_idx, source_masks[source_window], :])

    table = {}
    for window, segments in assigned_segments.items():
        snapshot = np.concatenate(segments, axis=0)
        table[window] = _pca_metrics_from_snapshot(snapshot)
    return table


def _permutation_effects(
    latents: np.ndarray,
    times_ms: np.ndarray,
    n_permutations: int = 2000,
    seed: int = 456,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sample_idx in range(n_permutations):
        metric_table = _permuted_window_metric_table(latents, times_ms, rng)
        for row in _difference_rows(metric_table):
            row["sample_index"] = sample_idx
            rows.append(row)
    return pd.DataFrame(rows)


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    result = np.full_like(p_values, np.nan)
    finite_mask = np.isfinite(p_values)
    finite_values = p_values[finite_mask]
    if len(finite_values) == 0:
        return result
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    adjusted = np.empty_like(ranked)
    n_tests = len(ranked)
    running = 1.0
    for idx in range(n_tests - 1, -1, -1):
        running = min(running, ranked[idx] * n_tests / (idx + 1))
        adjusted[idx] = running
    finite_result = np.empty_like(adjusted)
    finite_result[order] = np.clip(adjusted, 0.0, 1.0)
    result[finite_mask] = finite_result
    return result


def _summarize_significance(
    observed_effects: pd.DataFrame,
    bootstrap_effects: pd.DataFrame,
    permutation_effects: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, observed in observed_effects.iterrows():
        comparison = observed["comparison"]
        metric = observed["metric"]
        observed_difference = float(observed["difference"])
        boot = bootstrap_effects[
            (bootstrap_effects["comparison"] == comparison) & (bootstrap_effects["metric"] == metric)
        ]["difference"].to_numpy(dtype=float)
        perm = permutation_effects[
            (permutation_effects["comparison"] == comparison) & (permutation_effects["metric"] == metric)
        ]["difference"].to_numpy(dtype=float)
        boot = boot[np.isfinite(boot)]
        perm = perm[np.isfinite(perm)]
        if not np.isfinite(observed_difference) or len(boot) == 0 or len(perm) == 0:
            p_value = float("nan")
            ci_low, ci_high = float("nan"), float("nan")
            direction_probability = float("nan")
        else:
            p_value = float((np.sum(np.abs(perm) >= abs(observed_difference)) + 1) / (len(perm) + 1))
            ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
            if observed_difference >= 0:
                direction_probability = float(np.mean(boot > 0))
            else:
                direction_probability = float(np.mean(boot < 0))
        rows.append(
            {
                "comparison": comparison,
                "metric": metric,
                "observed_difference": observed_difference,
                "bootstrap_mean_difference": float(np.mean(boot)) if len(boot) else float("nan"),
                "bootstrap_ci_low": float(ci_low),
                "bootstrap_ci_high": float(ci_high),
                "direction_probability": direction_probability,
                "permutation_p_value": p_value,
            }
        )

    summary = pd.DataFrame(rows)
    summary["fdr_corrected_p_value"] = _bh_fdr(summary["permutation_p_value"].to_numpy(dtype=float))
    ci_supported = (summary["bootstrap_ci_low"] > 0) | (summary["bootstrap_ci_high"] < 0)
    p_supported = summary["fdr_corrected_p_value"] < 0.05
    summary["statistical_label"] = np.where(
        ci_supported & p_supported,
        "statistically_supported",
        np.where(ci_supported | p_supported, "partial_support", "trend_only"),
    )
    return summary


def _run_window_significance(
    latents: np.ndarray,
    times_ms: np.ndarray,
    n_bootstrap: int = 2000,
    n_permutations: int = 2000,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observed_table = _window_metric_table_for_indices(latents, times_ms, np.arange(latents.shape[0]))
    observed_effects = pd.DataFrame(_difference_rows(observed_table))
    bootstrap = _bootstrap_effects(latents, times_ms, n_bootstrap=n_bootstrap)
    permutation = _permutation_effects(latents, times_ms, n_permutations=n_permutations)
    summary = _summarize_significance(observed_effects, bootstrap, permutation)
    return summary, bootstrap, permutation


def _trend_summary(time_pca: pd.DataFrame) -> Dict[str, float | bool]:
    early = _mean_metric_by_windows(time_pca, "pc1_explained", ("early_pre_response",))["early_pre_response"]
    late = _mean_metric_by_windows(time_pca, "pc1_explained", ("late_pre_response",))["late_pre_response"]
    early_pr = _mean_metric_by_windows(time_pca, "participation_ratio", ("early_pre_response",))["early_pre_response"]
    late_pr = _mean_metric_by_windows(time_pca, "participation_ratio", ("late_pre_response",))["late_pre_response"]
    return {
        "early_pc1_explained": early,
        "late_pc1_explained": late,
        "late_minus_early_pc1_explained": late - early,
        "early_participation_ratio": early_pr,
        "late_participation_ratio": late_pr,
        "late_minus_early_participation_ratio": late_pr - early_pr,
        "pc1_explained_increases_pre_response": bool(late > early),
        "participation_ratio_decreases_pre_response": bool(late_pr < early_pr),
    }


def _interpretation_sentence(trend: Dict[str, float | bool], mapping: pd.DataFrame | None, analysis_scope: str = "single_split") -> str:
    strongest_mapping = float("nan")
    if mapping is not None and not mapping.empty:
        strongest_mapping = float(mapping["pearson_r"].abs().max())

    low_dimensional = bool(trend["pc1_explained_increases_pre_response"]) and bool(
        trend["participation_ratio_decreases_pre_response"]
    )
    aligned = np.isfinite(strongest_mapping) and strongest_mapping >= 0.30
    prefix = "The pooled latent dynamics" if analysis_scope == "pooled_train_val_test" else "The test-split latent dynamics"
    if low_dimensional and aligned:
        return (
            f"{prefix} show response-preceding low-dimensional concentration "
            "and PC/CPP proxy alignment, supporting a CPP-like response-proximal accumulation axis."
        )
    if low_dimensional:
        return (
            f"{prefix} show response-preceding low-dimensional concentration, "
            "but PC/CPP proxy alignment is weak or unavailable."
        )
    return (
        f"{prefix} do not show the full expected pattern of stronger PC1 "
        "concentration plus lower participation ratio before response."
    )


def _save_figures(
    output_dir: Path,
    times_ms: np.ndarray,
    time_pca: pd.DataFrame,
    windowed_pca: pd.DataFrame,
    scores: np.ndarray,
    mapping: pd.DataFrame | None,
    cpp_proxies: pd.DataFrame | None,
    control_summary: pd.DataFrame,
) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(time_pca["time_ms"], time_pca["pc1_explained"], label="PC1")
    plt.plot(time_pca["time_ms"], time_pca["pc1_pc2_cumulative"], label="PC1+PC2")
    plt.axvspan(-50, 100, color="tab:red", alpha=0.12, label="response/motor contaminated")
    plt.xlabel("time (ms)")
    plt.ylabel("explained variance ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "stage3_pc_explained_over_time.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(time_pca["time_ms"], time_pca["n_pc_90"], label="n_PC_90")
    plt.plot(time_pca["time_ms"], time_pca["participation_ratio"], label="participation ratio")
    plt.axvspan(-50, 100, color="tab:red", alpha=0.12, label="response/motor contaminated")
    plt.xlabel("time (ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "stage3_effective_dimensionality_over_time.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(windowed_pca["window"], windowed_pca["pc1_explained"], label="PC1")
    plt.plot(windowed_pca["window"], windowed_pca["pc1_pc2_cumulative"], color="black", marker="o", label="PC1+PC2")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("explained variance ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "stage3_windowed_pca_bar.png", dpi=150)
    plt.close()

    fig, axis = plt.subplots(figsize=(5, 4))
    pc_indices = np.arange(1, 5)
    window_labels = {
        "early_pre_response": "early pre",
        "mid_pre_response": "mid pre",
        "late_pre_response": "late pre",
        "peri_response_contaminated": "peri response",
    }
    colors = {
        "early_pre_response": "black",
        "mid_pre_response": "tab:blue",
        "late_pre_response": "tab:green",
        "peri_response_contaminated": "tab:red",
    }
    for _, row in windowed_pca.iterrows():
        values = [row[f"pc{pc_idx}_explained"] for pc_idx in pc_indices]
        axis.plot(
            pc_indices,
            values,
            marker="o",
            linewidth=1.4,
            markersize=4,
            color=colors.get(row["window"], "gray"),
            label=window_labels.get(row["window"], row["window"]),
        )
    axis.set_xticks(pc_indices)
    axis.set_xlabel("principal component")
    axis.set_ylabel("fraction variance explained")
    axis.set_ylim(bottom=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "stage3_window_pc_spectrum.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 4))
    ordered_windows = [window for window in WINDOWS if window in set(windowed_pca["window"])]
    ordered = windowed_pca.set_index("window").loc[ordered_windows].reset_index()
    x = np.arange(len(ordered))
    axis.plot(x, ordered["participation_ratio"], marker="o", color="black", linewidth=1.5)
    axis.set_xticks(x)
    axis.set_xticklabels([window_labels.get(window, window) for window in ordered["window"]], rotation=20, ha="right")
    axis.set_ylabel("participation ratio")
    axis.set_xlabel("response-locked window")
    axis.set_title("Lower participation ratio = more low-dimensional concentration", fontsize=9)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "stage3_window_participation_ratio.png", dpi=150)
    plt.close(fig)

    plt.figure(figsize=(8, 4))
    plt.plot(times_ms, scores.mean(axis=0)[:, 0], label="PC1")
    if scores.shape[-1] > 1:
        plt.plot(times_ms, scores.mean(axis=0)[:, 1], label="PC2")
    plt.axvspan(-50, 100, color="tab:red", alpha=0.12, label="response/motor contaminated")
    plt.xlabel("time (ms)")
    plt.ylabel("mean global PC score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "stage3_pc1_pc2_trajectory.png", dpi=150)
    plt.close()

    if mapping is not None and cpp_proxies is not None and not mapping.empty:
        late_mask = _window_mask(times_ms, -120.0, -50.0)
        pc1_late = scores[:, late_mask, 0].mean(axis=1)
        targets = [
            ("cpp_late_amplitude", "CPP late amplitude"),
            ("cpp_pre_response_slope", "CPP pre-response slope"),
            ("cpp_pre_response_auc", "CPP pre-response AUC"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for axis, (column, title) in zip(axes, targets):
            axis.scatter(pc1_late, cpp_proxies[column], s=18, alpha=0.75)
            axis.set_xlabel("PC1 late score")
            axis.set_title(title)
        axes[0].set_ylabel("CPP proxy")
        fig.tight_layout()
        fig.savefig(output_dir / "stage3_pc1_cpp_proxy_scatter.png", dpi=150)
        plt.close(fig)

    plt.figure(figsize=(9, 4))
    pivot = control_summary.pivot(index="window", columns="analysis", values="mean_pc1_explained")
    pivot.loc[list(WINDOWS.keys())].plot(kind="bar", ax=plt.gca())
    plt.ylabel("mean PC1 explained")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "stage3_control_comparison.png", dpi=150)
    plt.close()


def _save_effect_size_ci_figure(output_dir: Path, significance_summary: pd.DataFrame) -> None:
    plot_data = significance_summary.copy()
    plot_data["label"] = plot_data["comparison"].str.replace("_pre_response", "", regex=False).str.replace("_minus_", " - ", regex=False)
    plot_data["label"] = plot_data["label"] + "\n" + plot_data["metric"].str.replace("_", " ")
    y = np.arange(len(plot_data))
    lower = np.maximum(plot_data["observed_difference"] - plot_data["bootstrap_ci_low"], 0.0)
    upper = np.maximum(plot_data["bootstrap_ci_high"] - plot_data["observed_difference"], 0.0)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.axvline(0.0, color="black", linewidth=1.0, alpha=0.5)
    axis.errorbar(
        plot_data["observed_difference"],
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color="black",
        ecolor="gray",
        elinewidth=1.4,
        capsize=3,
    )
    axis.set_yticks(y)
    axis.set_yticklabels(plot_data["label"])
    axis.set_xlabel("window difference with bootstrap 95% CI")
    axis.set_title("Pooled trial window effects", fontsize=10)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "stage3_effect_size_ci.png", dpi=150)
    plt.close(fig)


def _run_latent_analysis_arrays(
    latents: np.ndarray,
    times_ms: np.ndarray,
    latent_metadata: pd.DataFrame,
    output_dir: Path,
    dataset_dir: Path | None = None,
    config: AnalysisConfig | None = None,
    analysis_split: str = "latents_test.npz",
    analysis_scope: str = "single_split",
    included_splits: Sequence[str] | None = None,
    run_significance: bool = False,
    n_bootstrap: int = 2000,
    n_permutations: int = 2000,
) -> Dict[str, object]:
    config = config or AnalysisConfig()
    output_dir = ensure_dir(output_dir)

    pca, transformed = _fit_global_pca(latents, config.pca_components)
    time_pca = _compute_time_resolved_pca(latents, times_ms)
    windowed_pca = _windowed_pca_summary(latents, times_ms, config.pca_components)
    trend = _trend_summary(time_pca)

    cpp_trials, raw_eeg, cpp_mapping_warning, cpp_quality = _load_aligned_cpp(latent_metadata, dataset_dir)
    cpp_proxies = None
    pc_cpp_mapping = None
    if cpp_trials is not None:
        cpp_proxies = _compute_cpp_proxies(cpp_trials, times_ms)
        pc_cpp_mapping = _summarize_pc_cpp_mapping(transformed, cpp_proxies, times_ms, pca.n_components_)

    controls, raw_time_pca = _control_summary(latents, times_ms, raw_eeg)

    np.savez_compressed(
        output_dir / "stage3_global_pca_scores.npz",
        scores=transformed,
        explained_variance_ratio=pca.explained_variance_ratio_,
        times_ms=times_ms,
    )
    time_pca.to_csv(output_dir / "stage3_time_resolved_pca.csv", index=False)
    time_pca.to_csv(output_dir / "stage3_time_varying_dimensionality.csv", index=False)
    windowed_pca.to_csv(output_dir / "stage3_window_pca_summary.csv", index=False)
    windowed_pca.to_csv(output_dir / "stage3_windowed_pca_summary.csv", index=False)
    controls.to_csv(output_dir / "stage3_control_summary.csv", index=False)
    if raw_time_pca is not None:
        raw_time_pca.to_csv(output_dir / "stage3_raw_cpp_time_resolved_pca.csv", index=False)
    if cpp_proxies is not None:
        cpp_proxies.to_csv(output_dir / "stage3_cpp_trial_proxies.csv", index=False)
    if pc_cpp_mapping is not None:
        pc_cpp_mapping.to_csv(output_dir / "stage3_pc_cpp_mapping.csv", index=False)

    _save_figures(output_dir, times_ms, time_pca, windowed_pca, transformed, pc_cpp_mapping, cpp_proxies, controls)

    significance_summary = None
    if run_significance:
        significance_summary, bootstrap_effects, permutation_effects = _run_window_significance(
            latents,
            times_ms,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
        )
        significance_summary.to_csv(output_dir / "stage3_window_significance_summary.csv", index=False)
        bootstrap_effects.to_csv(output_dir / "stage3_bootstrap_effects.csv", index=False)
        permutation_effects.to_csv(output_dir / "stage3_permutation_effects.csv", index=False)
        _save_effect_size_ci_figure(output_dir, significance_summary)

    report = {
        "passed": True,
        "latent_shape": list(latents.shape),
        "n_trials": int(latents.shape[0]),
        "analysis_split": analysis_split,
        "analysis_scope": analysis_scope,
        "included_splits": list(included_splits or []),
        "best_model_summary": _load_best_model_summary(output_dir),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "time_resolved_pca_rows": int(len(time_pca)),
        "window_pca_rows": int(len(windowed_pca)),
        "windows_ms": {name: list(bounds) for name, bounds in WINDOWS.items()},
        "contamination_warning": CONTAMINATION_NOTE,
        "trend_summary": trend,
        "cpp_mapping_available": pc_cpp_mapping is not None,
        "cpp_mapping_warning": cpp_mapping_warning,
        "cpp_data_quality": cpp_quality,
        "strongest_abs_pc_cpp_correlation": (
            float(pc_cpp_mapping["pearson_r"].abs().max()) if pc_cpp_mapping is not None and not pc_cpp_mapping.empty else float("nan")
        ),
        "interpretation": _interpretation_sentence(trend, pc_cpp_mapping, analysis_scope),
        "generated_files": [
            "stage3_time_resolved_pca.csv",
            "stage3_window_pca_summary.csv",
            "stage3_global_pca_scores.npz",
            "stage3_pc_explained_over_time.png",
            "stage3_effective_dimensionality_over_time.png",
            "stage3_windowed_pca_bar.png",
            "stage3_window_pc_spectrum.png",
            "stage3_window_participation_ratio.png",
            "stage3_pc1_pc2_trajectory.png",
            "stage3_control_summary.csv",
            "stage3_control_comparison.png",
        ],
    }
    if analysis_scope == "pooled_train_val_test":
        report["pooled_inference_warning"] = (
            "This pooled analysis includes train split latents seen during model fitting; use it to assess "
            "sample-size stability, not as independent test-set inference."
        )
    if significance_summary is not None:
        report["significance_tests"] = {
            "n_bootstrap": n_bootstrap,
            "n_permutations": n_permutations,
            "n_primary_tests": int(len(significance_summary)),
            "statistically_supported_tests": int((significance_summary["statistical_label"] == "statistically_supported").sum()),
        }
        report["generated_files"].extend(
            [
                "stage3_window_significance_summary.csv",
                "stage3_bootstrap_effects.csv",
                "stage3_permutation_effects.csv",
                "stage3_effect_size_ci.png",
            ]
        )
    if pc_cpp_mapping is not None:
        report["generated_files"].extend(["stage3_cpp_trial_proxies.csv", "stage3_pc_cpp_mapping.csv", "stage3_pc1_cpp_proxy_scatter.png"])
    write_json(output_dir / "stage3_analysis_report.json", report)
    return report


def run_core_latent_analysis(
    latent_npz: Path,
    output_dir: Path,
    config: AnalysisConfig | None = None,
    dataset_dir: Path | None = None,
) -> Dict[str, object]:
    """Run response-locked PCA/CPP mapping for one exported latent split."""
    latents, times_ms, latent_metadata = _load_latent_export(latent_npz)
    return _run_latent_analysis_arrays(
        latents,
        times_ms,
        latent_metadata,
        output_dir,
        dataset_dir=dataset_dir,
        config=config,
        analysis_split=Path(latent_npz).name,
        analysis_scope="single_split",
    )


def run_pooled_latent_analysis(
    latent_paths: Sequence[Path],
    output_dir: Path,
    config: AnalysisConfig | None = None,
    dataset_dir: Path | None = None,
    analysis_label: str = "pooled_train_val_test",
    n_bootstrap: int = 2000,
    n_permutations: int = 2000,
) -> Dict[str, object]:
    """Run pooled train/val/test latent analysis with window-level significance tests."""
    latents, times_ms, latent_metadata, included_splits = _load_pooled_latents(latent_paths)
    return _run_latent_analysis_arrays(
        latents,
        times_ms,
        latent_metadata,
        output_dir,
        dataset_dir=dataset_dir,
        config=config,
        analysis_split=",".join(Path(path).name for path in latent_paths),
        analysis_scope=analysis_label,
        included_splits=included_splits,
        run_significance=True,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
    )
