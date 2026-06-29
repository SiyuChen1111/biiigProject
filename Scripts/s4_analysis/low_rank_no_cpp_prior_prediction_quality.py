from __future__ import annotations

import argparse
import json
import os
import runpy
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


WINDOWS: dict[str, tuple[float, float]] = {
    "far_baseline": (-1000.0, -600.0),
    "early": (-600.0, -300.0),
    "mid": (-300.0, -120.0),
    "late": (-120.0, -50.0),
    "analysis_full": (-600.0, -50.0),
    "post_response_check": (-50.0, 100.0),
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_dataframe(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def save_figure(fig: plt.Figure, path_stem: Path) -> None:
    ensure_dir(path_stem.parent)
    fig.savefig(path_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".svg"), bbox_inches="tight")


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def notebook_namespace(notebook_path: Path) -> dict[str, Any]:
    import nbformat

    nb = nbformat.read(notebook_path, as_version=4)
    chunks: list[str] = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        source = str(cell.source)
        if any(
            marker in source
            for marker in [
                "_train_result = train_low_rank_r5_model",
                "test_metrics = test_low_rank_r5_model",
                "_latent_path = export_low_rank_r5_latents_from_checkpoint",
                "diagnostics_summary = run_low_rank_r5_diagnostics",
                "ridge_results = run_low_rank_r5_ridge_rt_analysis",
                "comparison_summary = run_original_vs_no_prior_comparison",
                "This cleanup cell can delete",
            ]
        ):
            continue
        chunks.append(source)
    code = "\n\n".join(chunks)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
        handle.write(code)
        temp_path = Path(handle.name)
    try:
        return runpy.run_path(str(temp_path))
    finally:
        temp_path.unlink(missing_ok=True)


def extract_test_predictions(ns: dict[str, Any], checkpoint_path: Path, dataset_dir: Path, batch_size: int) -> dict[str, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    globals().update(
        {
            "DataContractConfig": ns["DataContractConfig"],
            "LowRankRNNConfig": ns["LowRankRNNConfig"],
            "LossWeights": ns["LossWeights"],
            "TrainingConfig": ns["TrainingConfig"],
            "AnalysisConfig": ns["AnalysisConfig"],
        }
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ns["_coerce_training_config"](ckpt["config"])
    eeg, targets, mask, times_ms, metadata = ns["load_stage2_dataset"](dataset_dir, config)
    _, _, test_loader, split_indices = ns["make_dataloaders"](eeg, targets, mask, times_ms, config)
    model = ns["CPPLowRankRNN"](n_channels=eeg.shape[-1], config=config.model).to(device)
    ns["_load_checkpoint_weights"](model, ckpt)
    model.eval()

    rows: dict[str, list[np.ndarray]] = {
        "eeg": [],
        "future": [],
        "mask": [],
        "reconstructed": [],
        "predicted": [],
    }
    with torch.no_grad():
        for batch in test_loader:
            x = batch["eeg"].to(device)
            out = model(x)
            rows["eeg"].append(batch["eeg"].cpu().numpy())
            rows["future"].append(batch["target_future"].cpu().numpy())
            rows["mask"].append(batch["mask"].cpu().numpy())
            rows["reconstructed"].append(out.reconstructed.cpu().numpy())
            rows["predicted"].append(out.predicted.cpu().numpy())
    if hasattr(split_indices, "test_indices"):
        test_indices = split_indices.test_indices
    elif isinstance(split_indices, dict):
        test_indices = split_indices.get("test_indices", split_indices.get("test", np.array([], dtype=int)))
    else:
        test_indices = np.array([], dtype=int)
    return {
        "eeg": np.concatenate(rows["eeg"], axis=0).astype(np.float32),
        "future": np.concatenate(rows["future"], axis=0).astype(np.float32),
        "mask": np.concatenate(rows["mask"], axis=0).astype(bool),
        "reconstructed": np.concatenate(rows["reconstructed"], axis=0).astype(np.float32),
        "predicted": np.concatenate(rows["predicted"], axis=0).astype(np.float32),
        "times_ms": times_ms.astype(np.float32),
        "test_indices": np.asarray(test_indices),
    }


def masked_time_mse(error: np.ndarray, valid_time: np.ndarray) -> np.ndarray:
    valid = valid_time[None, :, None]
    numerator = np.where(valid, error, np.nan)
    return np.nanmean(numerator, axis=(0, 2))


def masked_cpp_time_mse(error: np.ndarray, valid_time: np.ndarray) -> np.ndarray:
    cpp_error = error.mean(axis=2)
    return np.nanmean(np.where(valid_time[None, :], cpp_error, np.nan), axis=0)


def build_quality_tables(preds: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    eeg = preds["eeg"]
    future = preds["future"]
    reconstructed = preds["reconstructed"]
    predicted = preds["predicted"]
    times = preds["times_ms"]
    training_mask = preds["mask"]
    finite_time = (
        np.isfinite(eeg).all(axis=(0, 2))
        & np.isfinite(future).all(axis=(0, 2))
        & np.isfinite(reconstructed).all(axis=(0, 2))
        & np.isfinite(predicted).all(axis=(0, 2))
    )

    recon_sq = (reconstructed - eeg) ** 2
    future_sq = (predicted - future) ** 2
    naive_future_sq = (eeg - future) ** 2
    cpp_recon_sq = (reconstructed.mean(axis=2) - eeg.mean(axis=2)) ** 2
    cpp_future_sq = (predicted.mean(axis=2) - future.mean(axis=2)) ** 2
    cpp_naive_future_sq = (eeg.mean(axis=2) - future.mean(axis=2)) ** 2

    time_df = pd.DataFrame(
        {
            "time_ms": times,
            "training_loss_mask_fraction": training_mask.mean(axis=0),
            "quality_eval_valid": finite_time.astype(float),
            "reconstruction_mse": masked_time_mse(recon_sq, finite_time),
            "future_prediction_mse": masked_time_mse(future_sq, finite_time),
            "naive_future_mse": masked_time_mse(naive_future_sq, finite_time),
            "cpp_reconstruction_mse": masked_cpp_time_mse(cpp_recon_sq[:, :, None], finite_time),
            "cpp_future_prediction_mse": masked_cpp_time_mse(cpp_future_sq[:, :, None], finite_time),
            "cpp_naive_future_mse": masked_cpp_time_mse(cpp_naive_future_sq[:, :, None], finite_time),
        }
    )
    time_df["future_minus_naive_mse"] = time_df["future_prediction_mse"] - time_df["naive_future_mse"]
    time_df["future_improvement_ratio"] = 1.0 - (time_df["future_prediction_mse"] / time_df["naive_future_mse"])
    time_df["cpp_future_minus_naive_mse"] = time_df["cpp_future_prediction_mse"] - time_df["cpp_naive_future_mse"]
    time_df["cpp_future_improvement_ratio"] = 1.0 - (time_df["cpp_future_prediction_mse"] / time_df["cpp_naive_future_mse"])

    window_rows: list[dict[str, float | str | int]] = []
    for label, (lo, hi) in WINDOWS.items():
        in_window = (times >= lo) & (times <= hi)
        row: dict[str, float | str | int] = {"window": label, "start_ms": lo, "end_ms": hi, "n_timepoints": int(in_window.sum())}
        for col in time_df.columns:
            if col == "time_ms":
                continue
            row[col] = float(time_df.loc[in_window, col].mean(skipna=True)) if in_window.any() else float("nan")
        window_rows.append(row)
    return time_df, pd.DataFrame(window_rows)


def plot_quality(time_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(time_df["time_ms"], time_df["reconstruction_mse"], label="EEG reconstruction", linewidth=1.3)
    ax.axvspan(-600, -50, color="0.9", alpha=0.6, label="analysis window")
    ax.axvline(0, color="0.2", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Time from response (ms)")
    ax.set_ylabel("MSE")
    ax.set_title("Reconstruction error by time")
    clean_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_dir / "reconstruction_error_by_time")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(time_df["time_ms"], time_df["future_prediction_mse"], label="model future prediction", linewidth=1.3)
    ax.plot(time_df["time_ms"], time_df["naive_future_mse"], label="naive current-as-future", linewidth=1.1, linestyle="--")
    ax.axvspan(-600, -50, color="0.9", alpha=0.6, label="analysis window")
    ax.axvline(0, color="0.2", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Time from response (ms)")
    ax.set_ylabel("MSE")
    ax.set_title("Future prediction error by time")
    clean_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_dir / "future_prediction_error_by_time")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(time_df["time_ms"], time_df["cpp_reconstruction_mse"], label="CPP reconstruction", linewidth=1.3)
    ax.plot(time_df["time_ms"], time_df["cpp_future_prediction_mse"], label="CPP future prediction", linewidth=1.3)
    ax.axvspan(-600, -50, color="0.9", alpha=0.6, label="analysis window")
    ax.axvline(0, color="0.2", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Time from response (ms)")
    ax.set_ylabel("CPP proxy MSE")
    ax.set_title("CPP reconstruction and future error by time")
    clean_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_dir / "cpp_reconstruction_future_error_by_time")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(time_df["time_ms"], time_df["future_improvement_ratio"], label="EEG improvement over naive", linewidth=1.3)
    ax.plot(time_df["time_ms"], time_df["cpp_future_improvement_ratio"], label="CPP improvement over naive", linewidth=1.1)
    ax.axhline(0, color="0.2", linewidth=0.8, alpha=0.7)
    ax.axvspan(-600, -50, color="0.9", alpha=0.6, label="analysis window")
    ax.axvline(0, color="0.2", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Time from response (ms)")
    ax.set_ylabel("1 - model MSE / naive MSE")
    ax.set_title("Future prediction vs naive baseline")
    clean_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_dir / "future_prediction_vs_naive_baseline")
    plt.close(fig)


def interpretation(window_df: pd.DataFrame) -> str:
    row = {r["window"]: r for _, r in window_df.iterrows()}
    far = row["far_baseline"]
    analysis = row["analysis_full"]
    recon_ratio = float(analysis["reconstruction_mse"] / far["reconstruction_mse"])
    future_improvement = float(analysis["future_improvement_ratio"])
    cpp_future_improvement = float(analysis["cpp_future_improvement_ratio"])
    analysis_better_recon = recon_ratio < 0.90
    future_beats_naive = future_improvement > 0.02
    cpp_future_beats_naive = cpp_future_improvement > 0.02
    if analysis_better_recon:
        recon_sentence = "analysis window reconstruction is better than the far-baseline period"
    else:
        recon_sentence = "analysis window reconstruction is not clearly better than the far-baseline period"
    if future_beats_naive or cpp_future_beats_naive:
        future_sentence = "future prediction beats the naive current-as-future baseline in at least one analysis-window signal summary"
    else:
        future_sentence = "future prediction does not clearly beat the naive current-as-future baseline in the analysis window"
    return "\n".join(
        [
            "### Prediction quality interpretation",
            "",
            f"- Analysis-window reconstruction MSE / far-baseline reconstruction MSE = `{recon_ratio:.3f}`; {recon_sentence}.",
            f"- Analysis-window EEG future improvement over naive = `{future_improvement:.3f}`.",
            f"- Analysis-window CPP future improvement over naive = `{cpp_future_improvement:.3f}`.",
            f"- Summary: {future_sentence}.",
            "",
            "Cautious interpretation: use the no-prior z variables primarily as learned representations for RT/CPP analyses. Avoid describing this model as a strong generative account of CPP evolution unless the future-prediction checks are clearly positive.",
        ]
    )


def run_quality_check(notebook_path: Path, checkpoint_path: Path, dataset_dir: Path, output_dir: Path, batch_size: int) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    ns = notebook_namespace(notebook_path)
    preds = extract_test_predictions(ns, checkpoint_path, dataset_dir, batch_size=batch_size)
    time_df, window_df = build_quality_tables(preds)
    save_dataframe(output_dir / "time_resolved_prediction_quality.csv", time_df)
    save_dataframe(output_dir / "windowed_prediction_quality.csv", window_df)
    plot_quality(time_df, output_dir)
    interpretation_md = interpretation(window_df)
    (output_dir / "prediction_quality_interpretation.md").write_text(interpretation_md, encoding="utf-8")
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "n_test_trials": int(preds["eeg"].shape[0]),
        "n_timepoints": int(preds["eeg"].shape[1]),
        "n_channels": int(preds["eeg"].shape[2]),
        "windows": {key: list(value) for key, value in WINDOWS.items()},
        "analysis_full": window_df[window_df["window"] == "analysis_full"].iloc[0].to_dict(),
        "far_baseline": window_df[window_df["window"] == "far_baseline"].iloc[0].to_dict(),
        "outputs": sorted(path.name for path in output_dir.glob("*")),
    }
    write_json(output_dir / "prediction_quality_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Rank-5 no-CPP-prior prediction quality without retraining.")
    parser.add_argument("--notebook", type=Path, default=Path("Scripts/low_rank_rnn_rank5_no_cpp_prior_ablation.ipynb"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("Data/ProcessedData"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_quality_check(
        notebook_path=args.notebook,
        checkpoint_path=args.checkpoint,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
