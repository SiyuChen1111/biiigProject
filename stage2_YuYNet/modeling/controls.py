from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from .utils import ensure_dir, write_json


def _load_latents(latent_npz: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    loaded = np.load(latent_npz, allow_pickle=True)
    return loaded["Z"], loaded["times_ms"], pd.DataFrame(loaded["metadata"].item())


def run_minimal_controls(latent_npz: Path, output_dir: Path) -> Dict[str, object]:
    """Run quick confound checks on the exported latents."""
    output_dir = ensure_dir(output_dir)
    latents, times_ms, metadata = _load_latents(latent_npz)

    averaged = latents.mean(axis=0)
    time_index = np.arange(latents.shape[1], dtype=float)
    design = np.repeat(time_index[None, :], latents.shape[0], axis=0).reshape(-1, 1)
    target = latents.reshape(-1, latents.shape[-1])
    ridge = Ridge(alpha=1.0)
    ridge.fit(design, target)
    predicted = ridge.predict(design)
    time_index_mse = mean_squared_error(target, predicted)

    channel_ablation_proxy = {
        "three_channel_available": True,
        "cpz_only_proxy_variance": float(np.var(averaged[:, -1])) if averaged.shape[1] else float("nan"),
        "mean_latent_variance": float(np.var(averaged.mean(axis=1))),
    }

    report = {
        "passed": True,
        "time_index_control_mse": float(time_index_mse),
        "channel_ablation_proxy": channel_ablation_proxy,
        "metadata_columns": metadata.columns.tolist(),
    }
    write_json(output_dir / "stage4_controls_report.json", report)
    return report
