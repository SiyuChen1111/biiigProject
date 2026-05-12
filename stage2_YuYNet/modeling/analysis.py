from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

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


def _response_locked_distance(latents: np.ndarray, times_ms: np.ndarray, metadata: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """Measure convergence toward a response-aligned latent state."""
    rt_ms = metadata["RT_ms"].to_numpy(dtype=float)
    response_state_mask = (times_ms >= config.response_locked_window_ms[0]) & (times_ms <= config.response_locked_window_ms[1])
    rows = []
    for trial_idx in range(latents.shape[0]):
        relative_time = times_ms - rt_ms[trial_idx]
        protected = (relative_time >= config.response_locked_window_ms[0]) & (relative_time <= config.response_locked_window_ms[1])
        if protected.sum() == 0:
            continue
        response_state = latents[trial_idx, protected, :].mean(axis=0)
        distance = np.linalg.norm(latents[trial_idx] - response_state[None, :], axis=1)
        for time_idx, value in enumerate(distance):
            rows.append(
                {
                    "trial_index": trial_idx,
                    "time_ms": float(times_ms[time_idx]),
                    "relative_to_rt_ms": float(relative_time[time_idx]),
                    "distance_to_response_state": float(value),
                }
            )
    return pd.DataFrame(rows)


def run_core_latent_analysis(latent_npz: Path, output_dir: Path, config: AnalysisConfig | None = None) -> Dict[str, object]:
    """Run the main latent-space analysis suite."""
    config = config or AnalysisConfig()
    output_dir = ensure_dir(output_dir)
    loaded = np.load(latent_npz, allow_pickle=True)
    latents = loaded["Z"]
    times_ms = loaded["times_ms"]
    metadata = pd.DataFrame(loaded["metadata"].item())

    pca, transformed = _fit_global_pca(latents, config.pca_components)
    dimensionality = _compute_time_varying_dimensionality(latents)
    response_distance = _response_locked_distance(latents, times_ms, metadata, config)

    np.savez_compressed(output_dir / "stage3_global_pca_scores.npz", scores=transformed, explained_variance_ratio=pca.explained_variance_ratio_)
    dimensionality.to_csv(output_dir / "stage3_time_varying_dimensionality.csv", index=False)
    response_distance.to_csv(output_dir / "stage3_response_locked_convergence.csv", index=False)

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
        "response_locked_rows": int(len(response_distance)),
        "contaminated_window_ms": list(config.contaminated_window_ms),
    }
    write_json(output_dir / "stage3_analysis_report.json", report)
    return report
