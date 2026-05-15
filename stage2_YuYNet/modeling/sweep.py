from __future__ import annotations

from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import TrainingConfig, default_evidence_dir
from .train import train_stage2_pipeline
from .utils import ensure_dir, write_json


def _waveform_metrics(run_dir: Path) -> Dict[str, float]:
    data = np.load(run_dir / "stage2_cpp_average_sanity.npz")
    real = data["real_avg"].mean(axis=0)
    recon = data["recon_avg"].mean(axis=0)
    real_delta = np.diff(real)
    recon_delta = np.diff(recon)
    late_slice = slice(-40, -13)

    corr = float(np.corrcoef(real, recon)[0, 1]) if np.std(real) and np.std(recon) else 0.0
    slope_corr = float(np.corrcoef(real_delta, recon_delta)[0, 1]) if np.std(real_delta) and np.std(recon_delta) else 0.0
    amp_ratio = float(np.std(recon) / np.std(real)) if np.std(real) else 0.0
    mse = float(np.mean((real - recon) ** 2))
    late_err = float(abs(recon[late_slice].mean() - real[late_slice].mean()))
    score = (
        0.40 * corr
        + 0.25 * slope_corr
        + 0.20 * max(0.0, 1.0 - abs(1.0 - amp_ratio))
        - 0.10 * late_err
        - 0.05 * mse
    )
    return {
        "corr": corr,
        "slope_corr": slope_corr,
        "amp_ratio": amp_ratio,
        "mse": mse,
        "late_err": late_err,
        "score": float(score),
    }


def _plot_best_overlays(run_dir: Path, output_dir: Path) -> Dict[str, str]:
    data = np.load(run_dir / "stage2_cpp_average_sanity.npz")
    real = data["real_avg"].mean(axis=0)
    recon = data["recon_avg"].mean(axis=0)
    real_delta = np.diff(real)
    recon_delta = np.diff(recon)

    cpp_path = output_dir / "best_cpp_overlay.png"
    plt.figure(figsize=(10, 4))
    plt.plot(real, label="real CPP")
    plt.plot(recon, label="recon CPP")
    plt.xlabel("time index")
    plt.ylabel("mean(CP1, CP2, CPz)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(cpp_path, dpi=150)
    plt.close()

    slope_path = output_dir / "best_cpp_slope_overlay.png"
    plt.figure(figsize=(10, 4))
    plt.plot(real_delta, label="real slope")
    plt.plot(recon_delta, label="recon slope")
    plt.xlabel("time index")
    plt.ylabel("mean slope")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(slope_path, dpi=150)
    plt.close()

    loss_src = run_dir / "stage2_validation_loss_history.png"
    loss_dst = output_dir / "best_training_loss.png"
    if loss_src.exists():
        loss_dst.write_bytes(loss_src.read_bytes())

    return {
        "best_cpp_overlay": str(cpp_path),
        "best_cpp_slope_overlay": str(slope_path),
        "best_training_loss": str(loss_dst),
    }


def _reduced_grid() -> List[Dict[str, float]]:
    keys = [
        "lambda_cpp_prior",
        "lambda_late_amplitude",
        "lambda_cpp_mean_alignment",
        "lambda_slope_floor",
        "slope_floor_ratio",
    ]
    values = [
        [0.05, 0.1],
        [1.0, 2.0, 4.0],
        [0.01, 0.05],
        [0.5],
        [0.4, 0.5],
    ]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def _small_grid() -> List[Dict[str, float]]:
    return [
        {
            "lambda_cpp_prior": 0.05,
            "lambda_late_amplitude": 1.0,
            "lambda_cpp_mean_alignment": 0.01,
            "lambda_slope_floor": 0.5,
            "slope_floor_ratio": 0.4,
        },
        {
            "lambda_cpp_prior": 0.1,
            "lambda_late_amplitude": 2.0,
            "lambda_cpp_mean_alignment": 0.05,
            "lambda_slope_floor": 0.5,
            "slope_floor_ratio": 0.5,
        },
    ]


_SWEEP_PARAM_KEYS = {
    "lambda_cpp_prior",
    "lambda_late_amplitude",
    "lambda_cpp_mean_alignment",
    "lambda_slope_floor",
    "slope_floor_ratio",
}


def run_cpp_prior_sweep(
    dataset_dir: Path,
    output_dir: Path | None = None,
    base_config: TrainingConfig | None = None,
    grid: Iterable[Dict[str, float]] | None = None,
    short_epochs: int = 8,
    long_epochs: int = 50,
    top_k: int = 3,
) -> Dict[str, object]:
    config = base_config or TrainingConfig()
    root = ensure_dir(output_dir or default_evidence_dir(dataset_dir.parent) / "sweep_cpp_prior")
    grid_list = list(grid or _reduced_grid())
    rows: List[Dict[str, object]] = []

    baseline_config = replace(config, enable_cpp_shape_prior=False, max_epochs=short_epochs)
    baseline_dir = root / "baseline"
    baseline_report = train_stage2_pipeline(dataset_dir, baseline_dir, baseline_config)
    baseline_metrics = _waveform_metrics(baseline_dir)
    rows.append(
        {
            "run_id": "baseline",
            "stage": "baseline",
            "run_dir": str(baseline_dir),
            "max_epochs": short_epochs,
            **baseline_metrics,
            **{f"param_{k}": v for k, v in asdict(baseline_config).items()},
            "test_total_loss": baseline_report["test_metrics"]["total_loss"],
        }
    )

    for run_idx, params in enumerate(grid_list):
        run_id = f"run_{run_idx:03d}"
        run_dir = root / run_id
        run_config = replace(config, enable_cpp_shape_prior=True, max_epochs=short_epochs, **params)
        report = train_stage2_pipeline(dataset_dir, run_dir, run_config)
        metrics = _waveform_metrics(run_dir)
        rows.append(
            {
                "run_id": run_id,
                "stage": "short",
                "run_dir": str(run_dir),
                "max_epochs": short_epochs,
                **metrics,
                **{f"param_{k}": v for k, v in asdict(run_config).items()},
                "test_total_loss": report["test_metrics"]["total_loss"],
            }
        )

    short_rows = [row for row in rows if row["stage"] == "short"]
    top_rows = sorted(short_rows, key=lambda item: float(item["score"]), reverse=True)[:top_k]
    for rank, row in enumerate(top_rows):
        params = {
            key.replace("param_", ""): row[key]
            for key in row
            if key.startswith("param_") and key.replace("param_", "") in _SWEEP_PARAM_KEYS
        }
        run_id = f"long_{rank:03d}"
        run_dir = root / run_id
        long_config = replace(config, enable_cpp_shape_prior=True, max_epochs=long_epochs, **params)
        report = train_stage2_pipeline(dataset_dir, run_dir, long_config)
        metrics = _waveform_metrics(run_dir)
        rows.append(
            {
                "run_id": run_id,
                "stage": "long",
                "source_short_run": row["run_id"],
                "run_dir": str(run_dir),
                "max_epochs": long_epochs,
                **metrics,
                **{f"param_{k}": v for k, v in asdict(long_config).items()},
                "test_total_loss": report["test_metrics"]["total_loss"],
            }
        )

    results = pd.DataFrame(rows)
    results = results.sort_values("score", ascending=False).reset_index(drop=True)
    results.to_csv(root / "sweep_results.csv", index=False)
    write_json(root / "sweep_results.json", {"runs": results.to_dict(orient="records")})

    best = results.iloc[0].to_dict()
    best_dir = Path(str(best["run_dir"]))
    plot_paths = _plot_best_overlays(best_dir, root)
    best_summary = {
        **best,
        "checkpoint_path": str(best_dir / "best_model.pt"),
        **plot_paths,
    }
    write_json(root / "best_run_summary.json", best_summary)
    return best_summary


def run_small_cpp_prior_sweep(dataset_dir: Path, output_dir: Path, config: TrainingConfig | None = None) -> Dict[str, object]:
    return run_cpp_prior_sweep(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        base_config=config or TrainingConfig(max_epochs=2, early_stopping_patience=1),
        grid=_small_grid(),
        short_epochs=2,
        long_epochs=2,
        top_k=1,
    )
