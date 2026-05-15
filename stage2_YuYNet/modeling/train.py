from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.adamw import AdamW
from dataclasses import replace

from .config import TrainingConfig, default_evidence_dir
from .dataset import Stage2SplitArtifacts, make_dataloaders
from .model import CPPForwardGRU, masked_self_supervised_loss
from .utils import ensure_dir, set_global_seed, write_json


def _run_epoch(
    model: CPPForwardGRU,
    loader,
    optimizer: Optional[Any],
    config: TrainingConfig,
    times_ms: np.ndarray,
    train: bool,
) -> Dict[str, float]:
    """Run one epoch and aggregate scalar metrics."""
    model.train(mode=train)
    metrics: List[Dict[str, float]] = []
    for batch in loader:
        eeg = batch["eeg"]
        future_targets = batch["future_targets"]
        mask = batch["mask"]
        if train and optimizer is not None:
            optimizer.zero_grad()
        outputs = model(eeg)
        loss, batch_metrics = masked_self_supervised_loss(
            outputs=outputs,
            target_current=eeg,
            target_future=future_targets,
            mask=mask,
            future_weight_scale=config.future_weight_scale,
            lambda_future=config.lambda_future,
            lambda_recon=config.lambda_recon,
            lambda_derivative=config.lambda_derivative,
            lambda_variance=config.lambda_variance,
            lambda_cpp_mean=config.lambda_cpp_mean,
            lambda_cpp_prior=config.lambda_cpp_prior,
            lambda_monotonic=config.lambda_monotonic,
            lambda_slope_floor=config.lambda_slope_floor,
            lambda_late_amplitude=config.lambda_late_amplitude,
            lambda_cpp_mean_alignment=config.lambda_cpp_mean_alignment,
            enable_cpp_shape_prior=config.enable_cpp_shape_prior,
            analysis_window=config.analysis_window_ms,
            late_window=config.late_window_ms,
            slope_floor_ratio=config.slope_floor_ratio,
            times_ms=torch.as_tensor(times_ms, dtype=torch.float32),
            lambda_smooth=config.lambda_smooth,
        )
        if train and optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
        metrics.append(batch_metrics)
    return {
        metric_name: float(np.mean([item[metric_name] for item in metrics]))
        for metric_name in metrics[0].keys()
    }


def _save_loss_plot(history: Dict[str, List[float]], output_dir: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(history["train_total_loss"], label="train")
    plt.plot(history["val_total_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("self-supervised loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "stage2_validation_loss_history.png", dpi=150)
    plt.close()


def _save_reconstruction_examples(model: CPPForwardGRU, loader, output_dir: Path) -> None:
    batch = next(iter(loader))
    eeg = batch["eeg"]
    outputs = model(eeg)
    sample_index = 0
    plt.figure(figsize=(10, 6))
    for channel_idx, channel_name in enumerate(("CP1", "CP2", "CPz")):
        ax = plt.subplot(3, 1, channel_idx + 1)
        ax.plot(eeg[sample_index, :, channel_idx].numpy(), label=f"real {channel_name}")
        ax.plot(outputs.reconstruction[sample_index, :, channel_idx].detach().numpy(), label=f"recon {channel_name}")
        ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "stage2_reconstruction_sanity.png", dpi=150)
    plt.close()


def _save_cpp_average_examples(model: CPPForwardGRU, loader, output_dir: Path) -> None:
    batch = next(iter(loader))
    eeg = batch["eeg"]
    outputs = model(eeg)
    real_avg = eeg.mean(dim=-1).detach().cpu().numpy()
    recon_avg = outputs.reconstruction.mean(dim=-1).detach().cpu().numpy()
    future_avg = outputs.future_prediction.mean(dim=-1).detach().cpu().numpy()
    sample_index = 0
    horizon_index = 0

    plt.figure(figsize=(10, 4))
    plt.plot(real_avg[sample_index], label="real CPP avg")
    plt.plot(recon_avg[sample_index], label="recon CPP avg")
    plt.plot(future_avg[sample_index, :, horizon_index], label="future CPP avg")
    plt.xlabel("time index")
    plt.ylabel("mean(CP1, CP2, CPz)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "stage2_cpp_average_sanity.png", dpi=150)
    plt.close()

    np.savez_compressed(
        output_dir / "stage2_cpp_average_sanity.npz",
        real_avg=real_avg,
        recon_avg=recon_avg,
        future_avg=future_avg,
    )


def _save_average_comparison_plot(output_dir: Path) -> None:
    data = np.load(output_dir / "stage2_cpp_average_sanity.npz")
    real_avg = data["real_avg"]
    recon_avg = data["recon_avg"]
    future_avg = data["future_avg"]

    plt.figure(figsize=(10, 4))
    plt.plot(real_avg.mean(axis=-1), label="real avg")
    plt.plot(recon_avg.mean(axis=-1), label="recon avg")
    plt.plot(future_avg.mean(axis=-1)[:, 0], label="future avg")
    plt.xlabel("time index")
    plt.ylabel("mean(CP1, CP2, CPz)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "stage2_average_waveform_comparison.png", dpi=150)
    plt.close()


def _save_derivative_comparison_plot(output_dir: Path) -> None:
    data = np.load(output_dir / "stage2_cpp_average_sanity.npz")
    real_waveform = data["real_avg"].mean(axis=0)
    recon_waveform = data["recon_avg"].mean(axis=0)
    real_delta = np.diff(real_waveform)
    recon_delta = np.diff(recon_waveform)

    plt.figure(figsize=(10, 4))
    plt.plot(real_delta, label="real slope")
    plt.plot(recon_delta, label="recon slope")
    plt.xlabel("time index")
    plt.ylabel("mean slope")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "stage2_derivative_comparison.png", dpi=150)
    plt.close()


def export_latents(
    model: CPPForwardGRU,
    loader,
    metadata,
    times_ms: np.ndarray,
    output_dir: Path,
    split_name: str,
) -> Path:
    """Export trial-wise latents for downstream PCA and control analyses."""
    latents = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["eeg"])
            latents.append(outputs.latents.detach().cpu().numpy())
    latent_array = np.concatenate(latents, axis=0)
    latent_path = output_dir / f"latents_{split_name}.npz"
    np.savez_compressed(
        latent_path,
        Z=latent_array,
        times_ms=times_ms,
        metadata=metadata.to_dict(orient="list"),
    )
    return latent_path


def train_stage2_pipeline(
    dataset_dir: Path,
    output_dir: Path | None = None,
    config: TrainingConfig | None = None,
) -> Dict[str, object]:
    """Train the baseline and write all stage outputs to disk."""
    config = config or TrainingConfig()
    set_global_seed(config.seed)
    output_dir = ensure_dir(output_dir or default_evidence_dir(dataset_dir.parent) / "stage2_training")

    loaders, times_ms, metadata, artifacts = make_dataloaders(dataset_dir, config)
    model = CPPForwardGRU(config)
    model.set_horizon(artifacts.horizon_steps)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    history: Dict[str, List[float]] = {
        "train_total_loss": [],
        "val_total_loss": [],
    }
    best_val = float("inf")
    best_epoch = -1
    patience = 0
    checkpoint_path = output_dir / "best_model.pt"

    for epoch in range(config.max_epochs):
        train_metrics = _run_epoch(model, loaders["train"], optimizer, config, times_ms, train=True)
        with torch.no_grad():
            val_metrics = _run_epoch(model, loaders["val"], optimizer, config, times_ms, train=False)

        history["train_total_loss"].append(train_metrics["total_loss"])
        history["val_total_loss"].append(val_metrics["total_loss"])
        if val_metrics["total_loss"] < best_val:
            best_val = val_metrics["total_loss"]
            best_epoch = epoch
            patience = 0
            torch.save({"model_state": model.state_dict(), "config": config.__dict__}, checkpoint_path)
        else:
            patience += 1
        if patience >= config.early_stopping_patience:
            break

    if not checkpoint_path.exists():
        torch.save({"model_state": model.state_dict(), "config": config.__dict__}, checkpoint_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model_state"])
    with torch.no_grad():
        test_metrics = _run_epoch(model, loaders["test"], None, config, times_ms, train=False)
    _save_loss_plot(history, output_dir)
    _save_reconstruction_examples(model, loaders["val"], output_dir)
    _save_cpp_average_examples(model, loaders["val"], output_dir)
    _save_average_comparison_plot(output_dir)
    _save_derivative_comparison_plot(output_dir)
    latent_paths = {
        split: str(export_latents(model, loader, metadata.iloc[getattr(artifacts, f"{split}_indices")].reset_index(drop=True), times_ms, output_dir, split))
        for split, loader in loaders.items()
    }

    report = {
        "passed": True,
        "best_epoch": best_epoch,
        "best_val_total_loss": best_val,
        "test_metrics": test_metrics,
        "checkpoint_path": str(checkpoint_path),
        "cpp_average_sanity_path": str(output_dir / "stage2_cpp_average_sanity.npz"),
        "derivative_comparison_path": str(output_dir / "stage2_derivative_comparison.png"),
        "latent_exports": latent_paths,
        "horizon_steps": artifacts.horizon_steps,
        "history": history,
    }
    write_json(output_dir / "stage2_training_report.json", report)
    return report


def train_baseline_and_soft_prior(
    dataset_dir: Path,
    output_dir: Path | None = None,
    config: TrainingConfig | None = None,
) -> Dict[str, object]:
    config = config or TrainingConfig()
    root = ensure_dir(output_dir or default_evidence_dir(dataset_dir.parent) / "stage2_comparison")
    baseline_config = replace(config, enable_cpp_shape_prior=False)
    soft_prior_config = replace(config, enable_cpp_shape_prior=True)

    baseline_report = train_stage2_pipeline(dataset_dir, root / "baseline", baseline_config)
    soft_prior_report = train_stage2_pipeline(dataset_dir, root / "soft_prior", soft_prior_config)

    summary = {
        "baseline_report": str(root / "baseline" / "stage2_training_report.json"),
        "soft_prior_report": str(root / "soft_prior" / "stage2_training_report.json"),
        "baseline_test_metrics": baseline_report["test_metrics"],
        "soft_prior_test_metrics": soft_prior_report["test_metrics"],
    }
    write_json(root / "stage2_baseline_vs_soft_prior_summary.json", summary)
    return summary
