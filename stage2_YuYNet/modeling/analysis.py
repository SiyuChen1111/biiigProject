from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from .config import AnalysisConfig
from .utils import ensure_dir, write_json


def _participation_ratio(eigenvalues: np.ndarray) -> float:
    numerator = float(np.sum(eigenvalues) ** 2)
    denominator = float(np.sum(eigenvalues ** 2))
    return numerator / denominator if denominator else 0.0


def _fit_global_pca(latents: np.ndarray, n_components: int) -> Tuple[PCA, np.ndarray]:
    flattened = latents.reshape(-1, latents.shape[-1])
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(flattened).reshape(latents.shape[0], latents.shape[1], n_components)
    return pca, transformed


def _compute_time_varying_dimensionality(latents: np.ndarray) -> pd.DataFrame:
    """Track how many PCs are needed to explain each time slice."""
    rows = []
    for time_idx in range(latents.shape[1]):
        snapshot = latents[:, time_idx, :]
        pca = PCA(n_components=min(snapshot.shape[0], snapshot.shape[1]))
        pca.fit(snapshot)
        explained = np.cumsum(pca.explained_variance_ratio_)
        rows.append(
            {
                "time_index": time_idx,
                "n_pc_80": int(np.searchsorted(explained, 0.80) + 1),
                "n_pc_90": int(np.searchsorted(explained, 0.90) + 1),
                "participation_ratio": _participation_ratio(pca.explained_variance_),
            }
        )
    return pd.DataFrame(rows)


def _windowed_pca_summary(latents: np.ndarray, times_ms: np.ndarray, config: AnalysisConfig) -> pd.DataFrame:
    rows = []
    window_mask = (times_ms >= config.response_locked_window_ms[0]) & (times_ms <= config.response_locked_window_ms[1])
    contaminated_mask = (times_ms >= config.contaminated_window_ms[0]) & (times_ms <= config.contaminated_window_ms[1])
    for label, mask in {
        "response_window": window_mask,
        "contaminated_window": contaminated_mask,
    }.items():
        if mask.sum() < 2:
            continue
        snapshot = latents[:, mask, :].reshape(-1, latents.shape[-1])
        pca = PCA(n_components=min(config.pca_components, snapshot.shape[1], snapshot.shape[0]))
        pca.fit(snapshot)
        explained = pca.explained_variance_ratio_
        rows.append(
            {
                "window": label,
                "n_points": int(mask.sum()),
                "pc1": float(explained[0]) if len(explained) > 0 else float("nan"),
                "pc2": float(explained[1]) if len(explained) > 1 else float("nan"),
                "pc3": float(explained[2]) if len(explained) > 2 else float("nan"),
                "participation_ratio": _participation_ratio(pca.explained_variance_),
            }
        )
    return pd.DataFrame(rows)


def _latent_cpp_proxy_summary(latents: np.ndarray, times_ms: np.ndarray, config: AnalysisConfig) -> pd.DataFrame:
    analysis_mask = (times_ms >= config.response_locked_window_ms[0]) & (times_ms <= config.response_locked_window_ms[1])
    late_mask = (times_ms >= config.contaminated_window_ms[0]) & (times_ms <= config.contaminated_window_ms[1])
    X = latents.mean(axis=1)
    time_curve = np.broadcast_to(times_ms[None, :], (latents.shape[0], len(times_ms)))
    slope_target = np.diff(time_curve[:, analysis_mask], axis=1).mean(axis=1)
    late_target = time_curve[:, late_mask].mean(axis=1)
    cumulative_target = time_curve[:, analysis_mask].sum(axis=1)

    rows = []
    for name, target in {
        "cpp_slope_proxy": slope_target,
        "cpp_late_amplitude_proxy": late_target,
        "cpp_cumulative_proxy": cumulative_target,
    }.items():
        reg = LinearRegression()
        reg.fit(X, target)
        pred = reg.predict(X)
        rows.append({"target": name, "r2": float(r2_score(target, pred))})
    return pd.DataFrame(rows)


def run_core_latent_analysis(latent_npz: Path, output_dir: Path, config: AnalysisConfig | None = None) -> Dict[str, object]:
    """Run the main latent-space analysis suite."""
    config = config or AnalysisConfig()
    output_dir = ensure_dir(output_dir)
    loaded = np.load(latent_npz, allow_pickle=True)
    latents = loaded["Z"]
    times_ms = loaded["times_ms"]

    pca, transformed = _fit_global_pca(latents, config.pca_components)
    dimensionality = _compute_time_varying_dimensionality(latents)
    windowed_pca = _windowed_pca_summary(latents, times_ms, config)
    latent_cpp_summary = _latent_cpp_proxy_summary(latents, times_ms, config)

    np.savez_compressed(output_dir / "stage3_global_pca_scores.npz", scores=transformed, explained_variance_ratio=pca.explained_variance_ratio_)
    dimensionality.to_csv(output_dir / "stage3_time_varying_dimensionality.csv", index=False)
    windowed_pca.to_csv(output_dir / "stage3_windowed_pca_summary.csv", index=False)
    latent_cpp_summary.to_csv(output_dir / "stage3_latent_cpp_proxy_summary.csv", index=False)

    plt.figure(figsize=(8, 4))
    plt.plot(times_ms, transformed.mean(axis=0)[:, 0], label="PC1")
    plt.plot(times_ms, transformed.mean(axis=0)[:, 1], label="PC2")
    plt.xlabel("time (ms)")
    plt.ylabel("mean PC score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "stage3_global_pca_trajectory.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(dimensionality["time_index"], dimensionality["n_pc_90"], label="n_PC_90")
    plt.plot(dimensionality["time_index"], dimensionality["participation_ratio"], label="participation_ratio")
    plt.xlabel("time index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "stage3_dimensionality_metrics.png", dpi=150)
    plt.close()

    report = {
        "passed": True,
        "latent_shape": list(latents.shape),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "windowed_pca_rows": int(len(windowed_pca)),
        "latent_cpp_proxy_rows": int(len(latent_cpp_summary)),
        "contaminated_window_ms": list(config.contaminated_window_ms),
    }
    write_json(output_dir / "stage3_analysis_report.json", report)
    return report
