from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from s1_modeling.config import LossWeights, TrainingConfig
from s1_modeling.dataset import load_stage2_dataset, make_dataloaders
from s1_modeling.low_rank_model import CPPLowRankRNN, LowRankRNNConfig, low_rank_self_supervised_loss
from s1_modeling.utils import set_global_seed


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

RANKS = (2, 3, 5)
WINDOWS = {
    "minus600_to_minus50": (-600.0, -50.0),
    "minus300_to_minus120": (-300.0, -120.0),
    "minus120_to_minus50": (-120.0, -50.0),
}


def _corr_1d(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(pearsonr(a[ok], b[ok]).statistic)


def _cpp_slope(cpp: np.ndarray, times_ms: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (times_ms >= lo) & (times_ms <= hi)
    x = times_ms[mask].astype(float)
    x = x - x.mean()
    denom = np.sum(x ** 2)
    y = cpp[:, mask].astype(float)
    return ((y - y.mean(axis=1, keepdims=True)) @ x) / denom


def _sample_subject_balanced(metadata: pd.DataFrame, max_trials: int, seed: int) -> np.ndarray:
    if max_trials >= len(metadata):
        return np.arange(len(metadata))
    rng = np.random.default_rng(seed)
    samples = []
    subjects = metadata["subject_id"].astype(str).to_numpy()
    unique_subjects = np.unique(subjects)
    per_subject = max(1, max_trials // len(unique_subjects))
    for subject in unique_subjects:
        idx = np.flatnonzero(subjects == subject)
        take = min(per_subject, len(idx))
        samples.extend(rng.choice(idx, size=take, replace=False).tolist())
    if len(samples) < max_trials:
        remaining = np.setdiff1d(np.arange(len(metadata)), np.asarray(samples), assume_unique=False)
        extra = rng.choice(remaining, size=min(max_trials - len(samples), len(remaining)), replace=False)
        samples.extend(extra.tolist())
    samples = np.asarray(samples[:max_trials], dtype=int)
    rng.shuffle(samples)
    return samples


def _subset_dataset(dataset_dir: Path, output_dir: Path, max_trials: int, seed: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    eeg = np.load(dataset_dir / "eeg_cpp_trials.npy").astype(np.float32)
    times_ms = np.load(dataset_dir / "times_ms.npy").astype(np.float32)
    metadata = pd.read_csv(dataset_dir / "metadata.csv")
    sample_idx = _sample_subject_balanced(metadata, max_trials=max_trials, seed=seed)

    np.save(output_dir / "eeg_cpp_trials.npy", eeg[sample_idx])
    np.save(output_dir / "times_ms.npy", times_ms)
    metadata.iloc[sample_idx].reset_index(drop=True).to_csv(output_dir / "metadata.csv", index=False)
    for fname in ("channel_names.txt", "preprocessing_notes.md"):
        (output_dir / fname).write_text((dataset_dir / fname).read_text(encoding="utf-8"), encoding="utf-8")
    np.save(output_dir / "sample_indices.npy", sample_idx)
    return output_dir


def _run_epoch(
    model: CPPLowRankRNN,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    config: TrainingConfig,
    device: torch.device,
    train: bool,
) -> dict[str, float]:
    model.train(train)
    totals: dict[str, float] = {}
    n_batches = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            x = batch["eeg"].to(device)
            target_future = batch["target_future"].to(device)
            mask = batch["mask"].to(device)
            times_ms = batch["times_ms"][0].to(device)
            out = model(x)
            loss, metrics = low_rank_self_supervised_loss(
                out, x, target_future, mask, times_ms, config.loss
            )
            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            n_batches += 1
    return {key: value / max(n_batches, 1) for key, value in totals.items()}


def _predict(
    model: CPPLowRankRNN,
    eeg: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    recon, latents = [], []
    with torch.no_grad():
        for start in range(0, len(eeg), batch_size):
            x = torch.as_tensor(eeg[start : start + batch_size], dtype=torch.float32, device=device)
            out = model(x)
            recon.append(out.reconstructed.cpu().numpy())
            latents.append(out.latents.cpu().numpy())
    return np.concatenate(recon, axis=0), np.concatenate(latents, axis=0)


def _evaluate_rank(
    rank: int,
    dataset_dir: Path,
    rank_dir: Path,
    config: TrainingConfig,
    population_dim: int,
    device: torch.device,
) -> dict[str, Any]:
    rank_dir.mkdir(parents=True, exist_ok=True)
    eeg, targets, mask, times_ms, metadata = load_stage2_dataset(dataset_dir, config)
    train_loader, val_loader, test_loader, splits = make_dataloaders(eeg, targets, mask, times_ms, config)

    model = CPPLowRankRNN(
        n_channels=eeg.shape[-1],
        config=LowRankRNNConfig(rank=rank, population_dim=population_dim),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    history = []
    for epoch in range(config.max_epochs):
        train_metrics = _run_epoch(model, train_loader, optimizer, config, device, train=True)
        val_metrics = _run_epoch(model, val_loader, None, config, device, train=False)
        row = {
            "rank": rank,
            "epoch": epoch + 1,
            "train_total_loss": train_metrics["total_loss"],
            "val_total_loss": val_metrics["total_loss"],
        }
        history.append(row)
    pd.DataFrame(history).to_csv(rank_dir / "training_history.csv", index=False)

    recon, latents = _predict(model, eeg, config.batch_size, device)
    test_idx = splits["test"]
    target_cpp = eeg[test_idx].mean(axis=2)
    recon_cpp = recon[test_idx].mean(axis=2)
    target_flat = eeg[test_idx].reshape(-1)
    recon_flat = recon[test_idx].reshape(-1)

    rows = []
    rows.append(
        {
            "rank": rank,
            "metric_group": "all_channels_test",
            "feature": "full_signal",
            "r2": float(r2_score(target_flat, recon_flat)),
            "rmse": float(np.sqrt(mean_squared_error(target_flat, recon_flat))),
            "corr": _corr_1d(target_flat, recon_flat),
        }
    )
    rows.append(
        {
            "rank": rank,
            "metric_group": "cpp_test",
            "feature": "average_waveform",
            "r2": float(r2_score(target_cpp.mean(axis=0), recon_cpp.mean(axis=0))),
            "rmse": float(np.sqrt(mean_squared_error(target_cpp.mean(axis=0), recon_cpp.mean(axis=0)))),
            "corr": _corr_1d(target_cpp.mean(axis=0), recon_cpp.mean(axis=0)),
        }
    )
    for name, (lo, hi) in WINDOWS.items():
        win_mask = (times_ms >= lo) & (times_ms <= hi)
        target_amp = target_cpp[:, win_mask].mean(axis=1)
        recon_amp = recon_cpp[:, win_mask].mean(axis=1)
        rows.append(
            {
                "rank": rank,
                "metric_group": "cpp_feature_test",
                "feature": f"amp_{name}",
                "r2": float(r2_score(target_amp, recon_amp)),
                "rmse": float(np.sqrt(mean_squared_error(target_amp, recon_amp))),
                "corr": _corr_1d(target_amp, recon_amp),
            }
        )
        if win_mask.sum() >= 2:
            target_slope = _cpp_slope(target_cpp, times_ms, lo, hi)
            recon_slope = _cpp_slope(recon_cpp, times_ms, lo, hi)
            rows.append(
                {
                    "rank": rank,
                    "metric_group": "cpp_feature_test",
                    "feature": f"slope_{name}",
                    "r2": float(r2_score(target_slope, recon_slope)),
                    "rmse": float(np.sqrt(mean_squared_error(target_slope, recon_slope))),
                    "corr": _corr_1d(target_slope, recon_slope),
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(rank_dir / "metrics.csv", index=False)
    np.savez(rank_dir / "low_rank_latents_test.npz", latents=latents[test_idx], times_ms=times_ms, trial_indices=test_idx)
    torch.save({"model_state_dict": model.state_dict(), "rank": rank, "config": asdict(config)}, rank_dir / "model.pt")

    _save_figures(rank_dir, rank, eeg[test_idx], recon[test_idx], latents[test_idx], metadata.iloc[test_idx], times_ms)
    return {"rank": rank, "metrics": rows, "final_val_loss": history[-1]["val_total_loss"]}


def _save_figures(
    rank_dir: Path,
    rank: int,
    eeg_test: np.ndarray,
    recon_test: np.ndarray,
    latents_test: np.ndarray,
    metadata_test: pd.DataFrame,
    times_ms: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    target_cpp = eeg_test.mean(axis=2)
    recon_cpp = recon_test.mean(axis=2)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times_ms, target_cpp.mean(axis=0), label="empirical CPP", lw=1.6)
    ax.plot(times_ms, recon_cpp.mean(axis=0), label=f"low-rank RNN rank {rank}", lw=1.6, ls="--")
    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Time from response (ms)")
    ax.set_ylabel("Normalized CPP proxy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(rank_dir / "cpp_average_reconstruction.png", dpi=170)
    plt.close(fig)

    z = latents_test
    if z.shape[2] == 1:
        z_plot = np.concatenate([z[:, :, :1], np.zeros_like(z[:, :, :1])], axis=2)
    elif z.shape[2] == 2:
        z_plot = z
    else:
        flat = z.reshape(-1, z.shape[2])
        pca = PCA(n_components=2, random_state=42)
        z_plot = pca.fit_transform(flat).reshape(z.shape[0], z.shape[1], 2)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    group_specs = [
        ("correctness", "Correctness"),
        ("difficulty", "Difficulty"),
        ("rt_bin", "RT tertile"),
    ]
    meta = metadata_test.reset_index(drop=True).copy()
    try:
        meta["rt_bin"] = pd.qcut(meta["RT_ms"], q=3, labels=["fast", "medium", "slow"], duplicates="drop")
    except ValueError:
        meta["rt_bin"] = "all"
    for ax, (column, title) in zip(axes, group_specs):
        if column not in meta.columns:
            ax.set_title(f"{title} unavailable")
            ax.set_xlabel("latent axis 1")
            continue
        for value in sorted(meta[column].dropna().unique(), key=lambda x: str(x)):
            idx = meta.index[meta[column].astype(str) == str(value)].to_numpy()
            if len(idx) < 2:
                continue
            traj = z_plot[idx].mean(axis=0)
            ax.plot(traj[:, 0], traj[:, 1], label=str(value), lw=1.4)
            ax.scatter(traj[-1, 0], traj[-1, 1], s=20)
        ax.set_title(title)
        ax.set_xlabel("latent axis 1")
    axes[0].set_ylabel("latent axis 2")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, fontsize=7)
    fig.tight_layout()
    fig.savefig(rank_dir / "latent_trajectories_by_group.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for dim in range(min(rank, 5)):
        ax.plot(times_ms, latents_test[:, :, dim].mean(axis=0), label=f"z{dim + 1}")
    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Time from response (ms)")
    ax.set_ylabel("Mean latent state")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(rank_dir / "mean_latent_timecourses.png", dpi=170)
    plt.close(fig)


def run_low_rank_smoke(
    dataset_dir: Path,
    output_dir: Path,
    ranks: tuple[int, ...] = RANKS,
    max_trials: int = 1200,
    max_epochs: int = 12,
    batch_size: int = 64,
    population_dim: int = 64,
    seed: int = 42,
) -> dict[str, Any]:
    set_global_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_dir = output_dir / "sampled_dataset"
    sampled_dataset = _subset_dataset(dataset_dir, subset_dir, max_trials=max_trials, seed=seed)

    loss = LossWeights(lambda_cpp_prior=0.05, lambda_smooth=0.01)
    config = TrainingConfig(
        seed=seed,
        batch_size=batch_size,
        max_epochs=max_epochs,
        early_stopping_patience=max_epochs,
        learning_rate=1e-3,
        weight_decay=1e-4,
        loss=loss,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rank_reports = []
    all_metrics = []
    for rank in ranks:
        rank_report = _evaluate_rank(
            rank=rank,
            dataset_dir=sampled_dataset,
            rank_dir=output_dir / f"rank_{rank}",
            config=config,
            population_dim=population_dim,
            device=device,
        )
        rank_reports.append(rank_report)
        all_metrics.extend(rank_report["metrics"])

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(output_dir / "low_rank_smoke_metrics.csv", index=False)
    summary = {
        "purpose": "Exploratory smoke test for low-rank RNN CPP modeling.",
        "dataset_dir": str(dataset_dir),
        "sampled_dataset_dir": str(sampled_dataset),
        "output_dir": str(output_dir),
        "ranks": list(ranks),
        "max_trials": max_trials,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "population_dim": population_dim,
        "seed": seed,
        "device": str(device),
        "rank_reports": [{"rank": r["rank"], "final_val_loss": r["final_val_loss"]} for r in rank_reports],
    }
    (output_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_smoke_readout(output_dir, metrics_df, summary)
    return summary


def _write_smoke_readout(output_dir: Path, metrics: pd.DataFrame, summary: dict[str, Any]) -> None:
    lines = [
        "# Low-Rank RNN Smoke Readout",
        "",
        "This is an exploratory smoke test, not a final model comparison.",
        "",
        "## Run Settings",
        "",
        f"- Dataset: `{summary['dataset_dir']}`",
        f"- Sampled trials: `{summary['max_trials']}`",
        f"- Ranks: `{summary['ranks']}`",
        f"- Epochs per rank: `{summary['max_epochs']}`",
        f"- Device: `{summary['device']}`",
        "",
        "## Main Metrics",
        "",
        "| Rank | Full signal R2 | CPP average corr | CPP average R2 | Late CPP amp corr | Full-window slope corr |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for rank in summary["ranks"]:
        rank_df = metrics[metrics["rank"] == rank]
        full = rank_df[(rank_df["metric_group"] == "all_channels_test") & (rank_df["feature"] == "full_signal")]
        cpp = rank_df[(rank_df["metric_group"] == "cpp_test") & (rank_df["feature"] == "average_waveform")]
        late_amp = rank_df[rank_df["feature"] == "amp_minus120_to_minus50"]
        slope = rank_df[rank_df["feature"] == "slope_minus600_to_minus50"]
        def val(df: pd.DataFrame, column: str) -> float:
            return float(df.iloc[0][column]) if len(df) else float("nan")
        lines.append(
            "| "
            f"{rank} | "
            f"{val(full, 'r2'):.3f} | "
            f"{val(cpp, 'corr'):.3f} | "
            f"{val(cpp, 'r2'):.3f} | "
            f"{val(late_amp, 'corr'):.3f} | "
            f"{val(slope, 'corr'):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- Treat a high CPP average correlation as evidence that the model preserved the broad response-locked CPP shape.",
            "- Treat high CPP slope or amplitude correlations as evidence that the low-rank state preserved trial-level CPP features.",
            "- Inspect the trajectory figures before making any claim about mechanism; numerical reconstruction alone is not enough.",
            "",
            "## Generated Figures",
            "",
            "- `rank_*/cpp_average_reconstruction.png`",
            "- `rank_*/latent_trajectories_by_group.png`",
            "- `rank_*/mean_latent_timecourses.png`",
        ]
    )
    (output_dir / "smoke_readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_ranks(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exploratory low-rank RNN smoke test.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("Data/ProcessedData"))
    parser.add_argument("--output-dir", type=Path, default=Path("Results/low_rank_rnn_smoke"))
    parser.add_argument("--ranks", type=_parse_ranks, default=RANKS)
    parser.add_argument("--max-trials", type=int, default=1200)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--population-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_low_rank_smoke(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        ranks=args.ranks,
        max_trials=args.max_trials,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        population_dim=args.population_dim,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
