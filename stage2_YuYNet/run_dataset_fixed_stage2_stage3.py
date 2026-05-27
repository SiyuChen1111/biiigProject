from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from modeling.config import AnalysisConfig, TrainingConfig
from modeling.dataset import make_dataloaders
from modeling.model import CPPForwardGRU, masked_self_supervised_loss
from modeling.train import export_latents


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset_fixed"
OUTPUT_ROOT = ROOT / "evidence" / "dataset_fixed_forward_gru"
STAGE2_DIR = OUTPUT_ROOT / "stage2"
STAGE3_DIR = OUTPUT_ROOT / "stage3_test"
CHECKPOINT = STAGE2_DIR / "best_model.pt"
CHANNELS = ["CP1", "CP2", "CPz"]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _window_mask(times_ms: np.ndarray, start: float, end: float) -> np.ndarray:
    return (times_ms >= start) & (times_ms <= end)


def _safe_corr_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan"), float("nan")
    r = float(pearsonr(x[ok], y[ok]).statistic)
    model = LinearRegression().fit(x[ok, None], y[ok])
    r2 = float(r2_score(y[ok], model.predict(x[ok, None])))
    return r, r2


def _load_model() -> tuple[CPPForwardGRU, dict[str, Any], dict[str, Any], np.ndarray, pd.DataFrame]:
    config = TrainingConfig()
    loaders, times_ms, metadata, artifacts = make_dataloaders(DATASET_DIR, config)
    model = CPPForwardGRU(config)
    model.set_horizon(artifacts.horizon_steps)
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model_state"])
    model.eval()
    return model, loaders, {"config": config, "artifacts": artifacts}, times_ms, metadata


def _evaluate_loader(model: CPPForwardGRU, loader, config: TrainingConfig, times_ms: np.ndarray) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["eeg"])
            _, metrics = masked_self_supervised_loss(
                outputs=outputs,
                target_current=batch["eeg"],
                target_future=batch["future_targets"],
                mask=batch["mask"],
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
            rows.append(metrics)
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _collect_recon(model: CPPForwardGRU, loader) -> tuple[np.ndarray, np.ndarray]:
    real, recon = [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["eeg"])
            real.append(batch["eeg"].cpu().numpy())
            recon.append(outputs.reconstruction.cpu().numpy())
    return np.concatenate(real, axis=0), np.concatenate(recon, axis=0)


def _save_stage2_figures(real: np.ndarray, recon: np.ndarray, times_ms: np.ndarray) -> dict[str, str]:
    STAGE2_DIR.mkdir(parents=True, exist_ok=True)
    real_cpp = real.mean(axis=2)
    recon_cpp = recon.mean(axis=2)

    plt.figure(figsize=(8, 4))
    plt.plot(times_ms, real_cpp.mean(axis=0), label="real CPP")
    plt.plot(times_ms, recon_cpp.mean(axis=0), label="recon CPP")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("time from response (ms)")
    plt.ylabel("mean normalized amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(STAGE2_DIR / "real_vs_recon_cpp_waveform.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(times_ms[1:], np.diff(real_cpp.mean(axis=0)), label="real CPP slope")
    plt.plot(times_ms[1:], np.diff(recon_cpp.mean(axis=0)), label="recon CPP slope")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("time from response (ms)")
    plt.ylabel("mean first difference")
    plt.legend()
    plt.tight_layout()
    plt.savefig(STAGE2_DIR / "real_vs_recon_cpp_slope.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for idx, channel in enumerate(CHANNELS):
        axes[idx].plot(times_ms, real[:, :, idx].mean(axis=0), label=f"real {channel}")
        axes[idx].plot(times_ms, recon[:, :, idx].mean(axis=0), label=f"recon {channel}")
        axes[idx].axvline(0, color="black", linestyle="--", linewidth=1)
        axes[idx].legend(loc="upper right")
        axes[idx].set_ylabel(channel)
    axes[-1].set_xlabel("time from response (ms)")
    fig.tight_layout()
    fig.savefig(STAGE2_DIR / "channel_wise_reconstruction.png", dpi=160)
    plt.close(fig)

    return {
        "real_vs_recon_cpp_waveform": str(STAGE2_DIR / "real_vs_recon_cpp_waveform.png"),
        "real_vs_recon_cpp_slope": str(STAGE2_DIR / "real_vs_recon_cpp_slope.png"),
        "channel_wise_reconstruction": str(STAGE2_DIR / "channel_wise_reconstruction.png"),
    }


def finish_stage2() -> dict[str, Any]:
    model, loaders, info, times_ms, metadata = _load_model()
    config: TrainingConfig = info["config"]
    artifacts = info["artifacts"]
    split_metrics = {
        split: _evaluate_loader(model, loader, config, times_ms)
        for split, loader in loaders.items()
    }
    real_test, recon_test = _collect_recon(model, loaders["test"])
    figure_paths = _save_stage2_figures(real_test, recon_test, times_ms)
    latent_paths = {
        split: str(
            export_latents(
                model,
                loader,
                metadata.iloc[getattr(artifacts, f"{split}_indices")].reset_index(drop=True),
                times_ms,
                STAGE2_DIR,
                split,
            )
        )
        for split, loader in loaders.items()
    }
    report = {
        "passed": True,
        "source": "completed_from_existing_best_model_checkpoint",
        "checkpoint_path": str(CHECKPOINT),
        "split_metrics": split_metrics,
        "latent_exports": latent_paths,
        "figures": figure_paths,
        "horizon_steps": artifacts.horizon_steps,
        "split_sizes": {
            "train": int(len(artifacts.train_indices)),
            "val": int(len(artifacts.val_indices)),
            "test": int(len(artifacts.test_indices)),
        },
    }
    (STAGE2_DIR / "stage2_completion_report.json").write_text(
        json.dumps(report, indent=2, default=_json_default), encoding="utf-8"
    )
    return report


def _load_latent_npz(path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    loaded = np.load(path, allow_pickle=True)
    return loaded["Z"], loaded["times_ms"], pd.DataFrame(loaded["metadata"].item())


def _align_eeg(metadata: pd.DataFrame) -> np.ndarray:
    eeg = np.load(DATASET_DIR / "eeg_cpp_trials.npy")
    dataset_meta = pd.read_csv(DATASET_DIR / "metadata.csv")
    trial_to_idx = {trial_id: idx for idx, trial_id in enumerate(dataset_meta["trial_id"])}
    indices = [trial_to_idx[trial_id] for trial_id in metadata["trial_id"]]
    return eeg[np.asarray(indices, dtype=int)]


def _cpp_features(eeg: np.ndarray, times_ms: np.ndarray) -> pd.DataFrame:
    cpp = eeg.mean(axis=2)
    late = _window_mask(times_ms, -120, -50)
    pre = _window_mask(times_ms, -600, -50)
    rows = []
    for idx, waveform in enumerate(cpp):
        rows.append(
            {
                "trial_index": idx,
                "cpp_late_amplitude": float(waveform[late].mean()),
                "cpp_pre_response_slope": float(np.polyfit(times_ms[pre], waveform[pre], 1)[0]),
                "cpp_pre_response_auc": float(np.trapezoid(waveform[pre], times_ms[pre])),
            }
        )
    return pd.DataFrame(rows)


def _score_table(latents: np.ndarray, times_ms: np.ndarray, cpp: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    scores, _ = _global_pca_scores(latents, n_components=min(3, latents.shape[-1]))
    masks = {
        "late_score": _window_mask(times_ms, -120, -50),
        "pre_response_score": _window_mask(times_ms, -600, -50),
    }
    targets = ["cpp_late_amplitude", "cpp_pre_response_slope", "cpp_pre_response_auc"]
    rows = []
    for pc_idx in range(scores.shape[-1]):
        for score_name, mask in masks.items():
            score = scores[:, mask, pc_idx].mean(axis=1)
            for target in targets:
                r, r2 = _safe_corr_r2(score, cpp[target].to_numpy())
                rows.append(
                    {
                        "source": "observed",
                        "pc": f"PC{pc_idx + 1}",
                        "score": score_name,
                        "target": target,
                        "pearson_r": r,
                        "linear_r2": r2,
                    }
                )
    return pd.DataFrame(rows), scores


def _global_pca_scores(latents: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    flat = latents.reshape(-1, latents.shape[-1]).astype(np.float64)
    flat = flat - flat.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(flat, full_matrices=False)
    components = vt[:n_components]
    scores = flat @ components.T
    scores = scores.reshape(latents.shape[0], latents.shape[1], n_components)
    explained = singular_values ** 2
    explained_ratio = explained / explained.sum()
    return scores.astype(np.float32), explained_ratio[:n_components].astype(np.float64)


def _pca_metrics(snapshot: np.ndarray) -> tuple[float, float, np.ndarray]:
    snapshot = np.asarray(snapshot, dtype=np.float64)
    snapshot = snapshot - snapshot.mean(axis=0, keepdims=True)
    covariance = np.cov(snapshot, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) == 0:
        return float("nan"), float("nan"), np.array([], dtype=float)
    explained = eigenvalues / eigenvalues.sum()
    participation = float((eigenvalues.sum() ** 2) / np.sum(eigenvalues ** 2))
    return float(explained[0]), participation, explained


def _time_resolved_pca(latents: np.ndarray, times_ms: np.ndarray) -> pd.DataFrame:
    rows = []
    for time_idx, time_ms in enumerate(times_ms):
        pc1, participation, explained = _pca_metrics(latents[:, time_idx, :])
        cumulative = np.cumsum(explained)
        rows.append(
            {
                "time_index": int(time_idx),
                "time_ms": float(time_ms),
                "pc1_explained": pc1,
                "pc2_explained": float(explained[1]) if len(explained) > 1 else 0.0,
                "pc1_pc2_cumulative": float(cumulative[min(1, len(cumulative) - 1)]) if len(cumulative) else float("nan"),
                "n_pc_80": int(np.searchsorted(cumulative, 0.80) + 1) if len(cumulative) else 0,
                "n_pc_90": int(np.searchsorted(cumulative, 0.90) + 1) if len(cumulative) else 0,
                "participation_ratio": participation,
            }
        )
    return pd.DataFrame(rows)


def _windowed_pca(latents: np.ndarray, times_ms: np.ndarray) -> pd.DataFrame:
    windows = {
        "early_pre_response": (-600.0, -300.0),
        "mid_pre_response": (-300.0, -120.0),
        "late_pre_response": (-120.0, -50.0),
        "peri_response_contaminated": (-50.0, 100.0),
    }
    rows = []
    for label, (start, end) in windows.items():
        mask = _window_mask(times_ms, start, end)
        snapshot = latents[:, mask, :].reshape(-1, latents.shape[-1])
        pc1, participation, explained = _pca_metrics(snapshot)
        cumulative = np.cumsum(explained)
        rows.append(
            {
                "window": label,
                "start_ms": start,
                "end_ms": end,
                "n_timepoints": int(mask.sum()),
                "n_observations": int(snapshot.shape[0]),
                "pc1_explained": pc1,
                "pc2_explained": float(explained[1]) if len(explained) > 1 else float("nan"),
                "pc3_explained": float(explained[2]) if len(explained) > 2 else float("nan"),
                "pc4_explained": float(explained[3]) if len(explained) > 3 else float("nan"),
                "pc1_pc2_cumulative": float(cumulative[min(1, len(cumulative) - 1)]) if len(cumulative) else float("nan"),
                "participation_ratio": participation,
            }
        )
    return pd.DataFrame(rows)


def _save_pca_figures(time_pca: pd.DataFrame, windowed: pd.DataFrame, scores: np.ndarray, times_ms: np.ndarray) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(time_pca["time_ms"], time_pca["pc1_explained"], label="PC1")
    plt.plot(time_pca["time_ms"], time_pca["pc1_pc2_cumulative"], label="PC1+PC2")
    plt.axvspan(-50, 100, color="tab:red", alpha=0.12)
    plt.xlabel("time from response (ms)")
    plt.ylabel("explained variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(STAGE3_DIR / "stage3_pc_explained_over_time.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(time_pca["time_ms"], time_pca["participation_ratio"], label="participation ratio")
    plt.plot(time_pca["time_ms"], time_pca["n_pc_90"], label="n PC for 90%")
    plt.axvspan(-50, 100, color="tab:red", alpha=0.12)
    plt.xlabel("time from response (ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(STAGE3_DIR / "stage3_effective_dimensionality_over_time.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(windowed["window"], windowed["pc1_explained"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("PC1 explained variance")
    plt.tight_layout()
    plt.savefig(STAGE3_DIR / "stage3_windowed_pca_bar.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(windowed["window"], windowed["participation_ratio"], marker="o")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("participation ratio")
    plt.tight_layout()
    plt.savefig(STAGE3_DIR / "stage3_window_participation_ratio.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(times_ms, scores[:, :, 0].mean(axis=0), label="PC1")
    if scores.shape[2] > 1:
        plt.plot(times_ms, scores[:, :, 1].mean(axis=0), label="PC2")
    plt.axvspan(-50, 100, color="tab:red", alpha=0.12)
    plt.xlabel("time from response (ms)")
    plt.ylabel("mean score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(STAGE3_DIR / "stage3_pc1_pc2_trajectory.png", dpi=160)
    plt.close()


def _control_score_table(latents: np.ndarray, times_ms: np.ndarray, cpp: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    controls = {}
    trial_shuffled = latents.copy()
    trial_shuffled = trial_shuffled[rng.permutation(trial_shuffled.shape[0])]
    controls["trial_shuffled_latent"] = trial_shuffled

    time_shuffled = latents.copy()
    for trial_idx in range(time_shuffled.shape[0]):
        time_shuffled[trial_idx] = time_shuffled[trial_idx, rng.permutation(time_shuffled.shape[1]), :]
    controls["time_shuffled_latent"] = time_shuffled

    direction = rng.normal(size=latents.shape[-1])
    direction = direction / np.linalg.norm(direction)
    projected = np.einsum("ntd,d->nt", latents, direction)
    targets = ["cpp_late_amplitude", "cpp_pre_response_slope", "cpp_pre_response_auc"]
    rows = []
    for label, control_latents in controls.items():
        table, _ = _score_table(control_latents, times_ms, cpp)
        table["source"] = label
        rows.append(table)
    for score_name, mask in {
        "late_score": _window_mask(times_ms, -120, -50),
        "pre_response_score": _window_mask(times_ms, -600, -50),
    }.items():
        score = projected[:, mask].mean(axis=1)
        for target in targets:
            r, r2 = _safe_corr_r2(score, cpp[target].to_numpy())
            rows.append(
                pd.DataFrame(
                    [
                        {
                            "source": "random_latent_direction",
                            "pc": "random_direction",
                            "score": score_name,
                            "target": target,
                            "pearson_r": r,
                            "linear_r2": r2,
                        }
                    ]
                )
            )
    return pd.concat(rows, ignore_index=True)


def _save_scatter_plots(scores: np.ndarray, times_ms: np.ndarray, cpp: pd.DataFrame) -> None:
    late_score = scores[:, _window_mask(times_ms, -120, -50), 0].mean(axis=1)
    targets = [
        ("cpp_late_amplitude", "CPP late amplitude"),
        ("cpp_pre_response_slope", "CPP pre-response slope"),
        ("cpp_pre_response_auc", "CPP AUC"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for axis, (column, title) in zip(axes, targets):
        axis.scatter(late_score, cpp[column], s=16, alpha=0.7)
        r, r2 = _safe_corr_r2(late_score, cpp[column].to_numpy())
        axis.set_title(f"{title}\nr={r:.3f}, R2={r2:.3f}")
        axis.set_xlabel("PC1 late score")
    axes[0].set_ylabel("CPP feature")
    fig.tight_layout()
    fig.savefig(STAGE3_DIR / "test_pc1_late_score_cpp_scatter.png", dpi=160)
    plt.close(fig)


def _latent_score_groups(scores: np.ndarray, eeg: np.ndarray, metadata: pd.DataFrame, times_ms: np.ndarray) -> pd.DataFrame:
    late_score = scores[:, _window_mask(times_ms, -120, -50), 0].mean(axis=1)
    group = pd.qcut(late_score, q=3, labels=["low", "mid", "high"], duplicates="drop")
    cpp = eeg.mean(axis=2)
    plt.figure(figsize=(8, 4))
    for label in ["low", "mid", "high"]:
        mask = np.asarray(group == label)
        plt.plot(times_ms, cpp[mask].mean(axis=0), label=f"{label} PC1 late")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("time from response (ms)")
    plt.ylabel("real CPP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(STAGE3_DIR / "latent_score_groups_real_cpp_waveform.png", dpi=160)
    plt.close()

    out = metadata.copy()
    out["pc1_late_score"] = late_score
    out["latent_score_group"] = group.astype(str)
    summary = out.groupby("latent_score_group", observed=True).agg(
        n_trials=("trial_id", "size"),
        mean_RT_ms=("RT_ms", "mean"),
        median_RT_ms=("RT_ms", "median"),
        correctness_mean=("correctness", "mean"),
        pc1_late_score_mean=("pc1_late_score", "mean"),
    ).reset_index()
    summary.to_csv(STAGE3_DIR / "latent_score_group_behavior_summary.csv", index=False)
    out[["trial_id", "subject_id", "RT_ms", "correctness", "pc1_late_score", "latent_score_group"]].to_csv(
        STAGE3_DIR / "latent_score_group_assignments.csv", index=False
    )
    return summary


def run_stage3_extra() -> dict[str, Any]:
    STAGE3_DIR.mkdir(parents=True, exist_ok=True)
    latent_path = STAGE2_DIR / "latents_test.npz"
    latents, times_ms, metadata = _load_latent_npz(latent_path)
    eeg = _align_eeg(metadata)
    cpp = _cpp_features(eeg, times_ms)
    cpp.to_csv(STAGE3_DIR / "test_cpp_trial_features.csv", index=False)
    observed_mapping, scores = _score_table(latents, times_ms, cpp)
    _, explained_ratio = _global_pca_scores(latents, n_components=min(3, latents.shape[-1]))
    time_pca = _time_resolved_pca(latents, times_ms)
    windowed = _windowed_pca(latents, times_ms)
    time_pca.to_csv(STAGE3_DIR / "stage3_time_resolved_pca.csv", index=False)
    time_pca.to_csv(STAGE3_DIR / "stage3_time_varying_dimensionality.csv", index=False)
    windowed.to_csv(STAGE3_DIR / "stage3_window_pca_summary.csv", index=False)
    windowed.to_csv(STAGE3_DIR / "stage3_windowed_pca_summary.csv", index=False)
    np.savez_compressed(
        STAGE3_DIR / "stage3_global_pca_scores.npz",
        scores=scores,
        explained_variance_ratio=explained_ratio,
        times_ms=times_ms,
    )
    _save_pca_figures(time_pca, windowed, scores, times_ms)
    controls = _control_score_table(latents, times_ms, cpp)
    mapping = pd.concat([observed_mapping, controls], ignore_index=True)
    mapping.to_csv(STAGE3_DIR / "test_latent_cpp_linking_with_controls.csv", index=False)
    observed_mapping.to_csv(STAGE3_DIR / "stage3_pc_cpp_mapping.csv", index=False)
    _save_scatter_plots(scores, times_ms, cpp)
    group_summary = _latent_score_groups(scores, eeg, metadata, times_ms)
    best = observed_mapping.iloc[observed_mapping["pearson_r"].abs().idxmax()].to_dict()
    control_best = controls.groupby("source")["pearson_r"].apply(lambda s: float(s.abs().max())).reset_index()
    control_best.to_csv(STAGE3_DIR / "stage3_control_summary.csv", index=False)
    report = {
        "passed": True,
        "analysis_split": "test",
        "test_trials": int(latents.shape[0]),
        "latent_shape": list(latents.shape),
        "explained_variance_ratio": explained_ratio.tolist(),
        "best_observed_latent_cpp_link": best,
        "control_max_abs_correlations": control_best.to_dict(orient="records"),
        "behavior_group_summary": group_summary.to_dict(orient="records"),
        "ddm_evidence": "not_run",
    }
    (STAGE3_DIR / "test_latent_cpp_behavior_report.json").write_text(
        json.dumps(report, indent=2, default=_json_default), encoding="utf-8"
    )
    return report


def main() -> None:
    stage2 = finish_stage2()
    stage3 = run_stage3_extra()
    combined = {"stage2": stage2, "stage3": stage3}
    (OUTPUT_ROOT / "dataset_fixed_stage2_stage3_report.json").write_text(
        json.dumps(combined, indent=2, default=_json_default), encoding="utf-8"
    )
    print(json.dumps(combined, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
