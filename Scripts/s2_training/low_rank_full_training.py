from __future__ import annotations

"""Train and validate the formal full-data rank-5 low-rank RNN CPP latent model.

Input:
    Processed response-locked EEG trials in Data/ProcessedData.
Output:
    Reports, tables, figures, logs, checkpoints, and latent exports under the
    requested Results/low_rank_full_training_rank5* directory.
Example:
    python Scripts/s2_training/low_rank_full_training.py \
      --dataset-dir Data/ProcessedData \
      --output-dir Results/low_rank_full_training_rank5 \
      --seeds 0,1,2,3,4 \
      --max-epochs 60 \
      --early-stopping-patience 10 \
      --batch-size 256
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, ttest_rel
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from s1_modeling.config import LossWeights, TrainingConfig
from s1_modeling.dataset import load_stage2_dataset, make_dataloaders
from s1_modeling.low_rank_model import CPPLowRankRNN, LowRankRNNConfig, low_rank_self_supervised_loss
from s1_modeling.utils import set_global_seed


WINDOWS = {
    "late": (-120.0, -50.0),
    "late_robust": (-150.0, -50.0),
    "buildup": (-300.0, -50.0),
    "buildup_broad": (-500.0, -50.0),
    "early": (-400.0, -300.0),
    "peak": (-600.0, 0.0),
}
CPP_FEATURES = ["CPP_late_mean", "CPP_slope", "CPP_peak", "CPP_peak_time"]
RIDGE_ALPHAS = np.logspace(-3, 5, 30)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    x, y = a[ok], b[ok]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(pearsonr(x, y).statistic)


def window_mean(y: np.ndarray, times_ms: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (times_ms >= lo) & (times_ms <= hi)
    return y[:, mask].mean(axis=1)


def slope_feature(y: np.ndarray, times_ms: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (times_ms >= lo) & (times_ms <= hi)
    x = times_ms[mask].astype(float)
    x = x - x.mean()
    denom = np.sum(x**2)
    yy = y[:, mask].astype(float)
    return ((yy - yy.mean(axis=1, keepdims=True)) @ x) / denom


def peak_and_time(y: np.ndarray, times_ms: np.ndarray, lo: float = -600.0, hi: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    mask = (times_ms >= lo) & (times_ms <= hi)
    yy = y[:, mask]
    tt = times_ms[mask]
    idx = np.argmax(yy, axis=1)
    return yy[np.arange(len(yy)), idx], tt[idx]


def safe_rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y, pred)))


def run_epoch(
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
            loss, metrics = low_rank_self_supervised_loss(out, x, target_future, mask, times_ms, config.loss)
            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            n_batches += 1
    return {key: value / max(n_batches, 1) for key, value in totals.items()}


def predict_all(model: CPPLowRankRNN, eeg: np.ndarray, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    recons, latents = [], []
    with torch.no_grad():
        for start in range(0, len(eeg), batch_size):
            x = torch.as_tensor(eeg[start : start + batch_size], dtype=torch.float32, device=device)
            out = model(x)
            recons.append(out.reconstructed.cpu().numpy())
            latents.append(out.latents.cpu().numpy())
    return np.concatenate(recons, axis=0), np.concatenate(latents, axis=0)


def save_loss_plot(history: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["epoch"], history["train_total_loss"], label="train")
    ax.plot(history["epoch"], history["val_total_loss"], label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def train_seed(
    dataset_dir: Path,
    output_dir: Path,
    seed: int,
    rank: int,
    population_dim: int,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, Any]:
    seed_dir = output_dir / "checkpoints" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)
    seed_config = replace(config, seed=seed)
    eeg, targets, mask, times_ms, metadata = load_stage2_dataset(dataset_dir, seed_config)
    train_loader, val_loader, test_loader, splits = make_dataloaders(eeg, targets, mask, times_ms, seed_config)
    model = CPPLowRankRNN(eeg.shape[-1], LowRankRNNConfig(rank=rank, population_dim=population_dim)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=seed_config.learning_rate, weight_decay=seed_config.weight_decay)

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = 0
    patience_left = seed_config.early_stopping_patience
    rows = []
    for epoch in range(seed_config.max_epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, seed_config, device, True)
        val_metrics = run_epoch(model, val_loader, None, seed_config, device, False)
        row = {"seed": seed, "rank": rank, "epoch": epoch + 1}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        rows.append(row)
        val_loss = val_metrics["total_loss"]
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = seed_config.early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history = pd.DataFrame(rows)
    history.to_csv(output_dir / "logs" / f"training_history_seed_{seed}.csv", index=False)
    save_loss_plot(history, output_dir / "figures" / f"training_loss_seed_{seed}.png")

    test_metrics = run_epoch(model, test_loader, None, seed_config, device, False)
    recon, latents = predict_all(model, eeg, seed_config.batch_size, device)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "rank": rank,
        "population_dim": population_dim,
        "seed": seed,
        "config": asdict(seed_config),
        "splits": {k: v.tolist() for k, v in splits.items()},
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "test_metrics": test_metrics,
    }
    torch.save(ckpt, seed_dir / "best_model.pt")
    (seed_dir / "config.json").write_text(json.dumps(ckpt["config"], indent=2), encoding="utf-8")
    np.savez_compressed(
        output_dir / "latent_exports" / f"latents_seed_{seed}.npz",
        eeg=eeg.astype(np.float32),
        recon=recon.astype(np.float32),
        latents=latents.astype(np.float32),
        times_ms=times_ms.astype(np.float32),
        train_idx=splits["train"],
        val_idx=splits["val"],
        test_idx=splits["test"],
    )
    return {
        "seed": seed,
        "rank": rank,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val),
        "test_total_loss": float(test_metrics["total_loss"]),
        "n_epochs_run": len(history),
        "checkpoint": str(seed_dir / "best_model.pt"),
    }


def valid_trial_mask(eeg_raw: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
    valid = np.isfinite(eeg_raw).all(axis=(1, 2))
    rt = metadata["RT_ms"].to_numpy(float)
    valid &= np.isfinite(rt) & (rt > 100.0) & (rt < 5000.0)
    if "artifact_rejection_flag" in metadata.columns:
        valid &= metadata["artifact_rejection_flag"].fillna(0).to_numpy() == 0
    if "alignment" in metadata.columns:
        valid &= metadata["alignment"].astype(str).eq("response_locked").to_numpy()
    return valid


def write_data_summary(dataset_dir: Path, output_dir: Path) -> pd.DataFrame:
    eeg_raw = np.load(dataset_dir / "eeg_cpp_trials.npy")
    times_ms = np.load(dataset_dir / "times_ms.npy")
    metadata = pd.read_csv(dataset_dir / "metadata.csv")
    valid = valid_trial_mask(eeg_raw, metadata)
    meta = metadata.loc[valid].copy()
    rows = [
        ("n_subjects", meta["subject_id"].nunique()),
        ("n_total_trials_available", len(metadata)),
        ("n_valid_trials_after_exclusion", int(valid.sum())),
        ("n_excluded_trials", int((~valid).sum())),
        ("n_correct_trials", int((meta["correctness"] == 1).sum())),
        ("n_error_trials", int((meta["correctness"] == 0).sum())),
        ("RT_ms_mean", float(meta["RT_ms"].mean())),
        ("RT_ms_sd", float(meta["RT_ms"].std())),
        ("RT_ms_min", float(meta["RT_ms"].min())),
        ("RT_ms_max", float(meta["RT_ms"].max())),
        ("alignment", ",".join(sorted(meta["alignment"].astype(str).unique())) if "alignment" in meta else "unknown"),
        ("response_time_zero_ms", 0),
        ("time_unit", "milliseconds"),
        ("sampling_rate_hz", float(1000.0 / np.mean(np.diff(times_ms)))),
        ("input_window_start_ms", float(times_ms.min())),
        ("input_window_end_ms", float(times_ms.max())),
        ("n_timepoints", len(times_ms)),
        ("model_input_channels", (dataset_dir / "channel_names.txt").read_text(encoding="utf-8").strip().replace("\n", ",")),
        ("cpp_channels_in_model_input", True),
    ]
    summary = pd.DataFrame(rows, columns=["metric", "value"])
    summary.to_csv(output_dir / "tables" / "data_summary_full.csv", index=False)
    meta.groupby("subject_id").size().reset_index(name="n_trials").to_csv(output_dir / "tables" / "trials_per_subject.csv", index=False)
    for col in ["condition", "difficulty", "evidence_strength"]:
        if col in meta.columns:
            meta.groupby(col).size().reset_index(name="n_trials").to_csv(output_dir / "tables" / f"trials_per_{col}.csv", index=False)
    return summary


def extract_features(
    metadata: pd.DataFrame,
    eeg: np.ndarray,
    z: np.ndarray,
    times_ms: np.ndarray,
    seed: int,
    prefix: str = "aligned_z",
) -> pd.DataFrame:
    cpp = eeg.mean(axis=2)
    out: dict[str, Any] = {
        "seed": seed,
        "subject_id": metadata["subject_id"].astype(str).to_numpy(),
        "trial_id": metadata["trial_id"].astype(str).to_numpy(),
        "RT_ms": metadata["RT_ms"].to_numpy(float),
        "log_RT_ms": np.log(metadata["RT_ms"].to_numpy(float)),
        "correctness": metadata["correctness"].to_numpy(float),
    }
    for col in ["condition", "difficulty", "evidence_strength", "choice", "response_hand"]:
        if col in metadata.columns:
            out[col] = metadata[col].to_numpy()
    out["CPP_late_mean"] = window_mean(cpp, times_ms, *WINDOWS["late"])
    out["CPP_late_mean_robust"] = window_mean(cpp, times_ms, *WINDOWS["late_robust"])
    out["CPP_slope"] = slope_feature(cpp, times_ms, *WINDOWS["buildup"])
    out["CPP_slope_broad"] = slope_feature(cpp, times_ms, *WINDOWS["buildup_broad"])
    out["CPP_early_mean"] = window_mean(cpp, times_ms, *WINDOWS["early"])
    out["CPP_peak"], out["CPP_peak_time"] = peak_and_time(cpp, times_ms, *WINDOWS["peak"])
    for dim in range(z.shape[2]):
        name = f"{prefix}{dim + 1}"
        zz = z[:, :, dim]
        out[f"{name}_late_mean"] = window_mean(zz, times_ms, *WINDOWS["late"])
        out[f"{name}_late_mean_robust"] = window_mean(zz, times_ms, *WINDOWS["late_robust"])
        out[f"{name}_slope"] = slope_feature(zz, times_ms, *WINDOWS["buildup"])
        out[f"{name}_slope_broad"] = slope_feature(zz, times_ms, *WINDOWS["buildup_broad"])
        out[f"{name}_early_mean"] = window_mean(zz, times_ms, *WINDOWS["early"])
        out[f"{name}_peak"], out[f"{name}_peak_time"] = peak_and_time(zz, times_ms, *WINDOWS["peak"])
    return pd.DataFrame(out)


def load_seed_export(output_dir: Path, seed: int) -> dict[str, np.ndarray]:
    return dict(np.load(output_dir / "latent_exports" / f"latents_seed_{seed}.npz", allow_pickle=True))


def align_latents(output_dir: Path, seeds: list[int], rank: int, metadata: pd.DataFrame) -> tuple[dict[int, np.ndarray], pd.DataFrame, int]:
    metrics = pd.read_csv(output_dir / "tables" / "training_metrics_by_seed.csv")
    ref_seed = int(metrics.sort_values("best_val_loss").iloc[0]["seed"])
    ref = load_seed_export(output_dir, ref_seed)
    ref_z = ref["latents"].astype(np.float32)
    ref_cpp = ref["eeg"].mean(axis=2)
    ref_late = window_mean(ref_cpp, ref["times_ms"], *WINDOWS["late"])
    ref_feat = np.column_stack([window_mean(ref_z[:, :, d], ref["times_ms"], *WINDOWS["late"]) for d in range(rank)])

    aligned: dict[int, np.ndarray] = {ref_seed: ref_z.copy()}
    rows = []
    for seed in seeds:
        cur = load_seed_export(output_dir, seed)
        z = cur["latents"].astype(np.float32)
        feat = np.column_stack([window_mean(z[:, :, d], cur["times_ms"], *WINDOWS["late"]) for d in range(rank)])
        sim = np.zeros((rank, rank), dtype=float)
        for i in range(rank):
            for j in range(rank):
                sim[i, j] = abs(corr(feat[:, i], ref_feat[:, j]))
        raw_ind, aligned_ind = linear_sum_assignment(-sim)
        z_aligned = np.zeros_like(z)
        for raw, ali in zip(raw_ind, aligned_ind):
            sign_corr = corr(feat[:, raw], ref_feat[:, ali])
            sign = -1.0 if np.isfinite(sign_corr) and sign_corr < 0 else 1.0
            z_candidate = z[:, :, raw] * sign
            cpp_late = window_mean(cur["eeg"].mean(axis=2), cur["times_ms"], *WINDOWS["late"])
            cpp_slope = slope_feature(cur["eeg"].mean(axis=2), cur["times_ms"], *WINDOWS["buildup"])
            z_late = window_mean(z_candidate, cur["times_ms"], *WINDOWS["late"])
            if corr(z_late, cpp_late) < 0:
                sign *= -1.0
                z_candidate *= -1.0
                z_late *= -1.0
            z_aligned[:, :, ali] = z_candidate
            rows.append(
                {
                    "seed": seed,
                    "raw_latent_index": int(raw + 1),
                    "aligned_latent_index": int(ali + 1),
                    "correlation_with_reference_latent": float(sign_corr),
                    "correlation_with_CPP_late_mean": corr(z_late, cpp_late),
                    "correlation_with_CPP_slope": corr(slope_feature(z_candidate, cur["times_ms"], *WINDOWS["buildup"]), cpp_slope),
                    "sign_flipped": bool(sign < 0),
                }
            )
        aligned[seed] = z_aligned
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "tables" / "latent_alignment_table.csv", index=False)
    return aligned, table, ref_seed


def zscore_train_apply(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    include_correctness: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    x_train = train[cols].to_numpy(float) if cols else np.empty((len(train), 0))
    x_test = test[cols].to_numpy(float) if cols else np.empty((len(test), 0))
    covariate_candidates = ["difficulty"]
    if include_correctness:
        covariate_candidates.append("correctness")
    covs = [c for c in covariate_candidates if c in train.columns and c not in cols]
    if covs:
        x_train = np.column_stack([x_train, train[covs].to_numpy(float)])
        x_test = np.column_stack([x_test, test[covs].to_numpy(float)])
    return x_train, x_test


def subject_aware_rt_models(features: pd.DataFrame, z_selected: list[str], rank: int, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    z_all = []
    for d in range(1, rank + 1):
        z_all += [f"aligned_z{d}_late_mean", f"aligned_z{d}_slope", f"aligned_z{d}_peak", f"aligned_z{d}_peak_time"]
    model_specs = {
        "baseline": [],
        "CPP-only": CPP_FEATURES,
        "z-only": z_selected,
        "CPP + z": CPP_FEATURES + z_selected,
        "All-z": z_all,
        "Full": CPP_FEATURES + z_all,
    }
    y = features["log_RT_ms"].to_numpy(float)
    groups = features["subject_id"].to_numpy(str)
    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    rows, pred_rows = [], []
    for name, cols in model_specs.items():
        pred = np.full(len(features), np.nan)
        fold_rows = []
        for fold, (train_idx, test_idx) in enumerate(gkf.split(features, y, groups), start=1):
            train = features.iloc[train_idx]
            test = features.iloc[test_idx]
            x_train, x_test = zscore_train_apply(train, test, cols)
            model = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=RIDGE_ALPHAS, cv=5))])
            model.fit(x_train, y[train_idx])
            pred[test_idx] = model.predict(x_test)
            fold_rows.append(
                {
                    "seed": int(features["seed"].iloc[0]),
                    "model": name,
                    "fold": fold,
                    "cv_r2": float(r2_score(y[test_idx], pred[test_idx])),
                    "rmse": safe_rmse(y[test_idx], pred[test_idx]),
                    "mae": float(mean_absolute_error(y[test_idx], pred[test_idx])),
                    "n_test_subjects": len(np.unique(groups[test_idx])),
                    "n_test_trials": len(test_idx),
                }
            )
        rows.extend(fold_rows)
        rows.append(
            {
                "seed": int(features["seed"].iloc[0]),
                "model": name,
                "fold": "pooled",
                "cv_r2": float(r2_score(y, pred)),
                "rmse": safe_rmse(y, pred),
                "mae": float(mean_absolute_error(y, pred)),
                "n_test_subjects": len(np.unique(groups)),
                "n_test_trials": len(features),
            }
        )
        pred_rows.extend(
            {
                "seed": int(features["seed"].iloc[0]),
                "model": name,
                "trial_id": features.iloc[i]["trial_id"],
                "observed_log_RT": y[i],
                "predicted_log_RT": pred[i],
            }
            for i in range(len(features))
            if name in {"CPP-only", "CPP + z"}
        )
    comp = pd.DataFrame(rows)
    pooled_cpp = comp[(comp["fold"] == "pooled") & (comp["model"] == "CPP-only")]["cv_r2"].iloc[0]
    comp["delta_r2_vs_cpp_only"] = comp["cv_r2"] - pooled_cpp
    return comp, pd.DataFrame(pred_rows)


def correlation_tables(features: pd.DataFrame, rank: int, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    z_features = []
    for d in range(1, rank + 1):
        z_features += [f"aligned_z{d}_late_mean", f"aligned_z{d}_slope", f"aligned_z{d}_peak", f"aligned_z{d}_peak_time"]
    rows, within_rows, subject_rows = [], [], []
    for cpp_col in CPP_FEATURES:
        for z_col in z_features:
            rows.append({"seed": int(features["seed"].iloc[0]), "cpp_feature": cpp_col, "z_feature": z_col, "pooled_corr": corr(features[cpp_col].to_numpy(), features[z_col].to_numpy())})
            subj_corrs = []
            for subject, sub in features.groupby("subject_id"):
                r = corr(sub[cpp_col].to_numpy(), sub[z_col].to_numpy())
                if np.isfinite(r):
                    subj_corrs.append(r)
                    within_rows.append({"seed": int(features["seed"].iloc[0]), "subject_id": subject, "cpp_feature": cpp_col, "z_feature": z_col, "within_subject_corr": r})
            subj_mean = features.groupby("subject_id")[[cpp_col, z_col]].mean()
            subject_rows.append({"seed": int(features["seed"].iloc[0]), "cpp_feature": cpp_col, "z_feature": z_col, "subject_level_corr": corr(subj_mean[cpp_col].to_numpy(), subj_mean[z_col].to_numpy()), "mean_within_subject_corr": float(np.nanmean(subj_corrs)) if subj_corrs else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(within_rows), pd.DataFrame(subject_rows)


def select_cpp_like_latents(corr_table: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    mapping_rows = []
    for seed, sub in corr_table.groupby("seed"):
        amp = sub[sub["cpp_feature"] == "CPP_late_mean"].copy()
        slope = sub[sub["cpp_feature"] == "CPP_slope"].copy()
        amp = amp[amp["z_feature"].str.endswith("_late_mean")]
        slope = slope[slope["z_feature"].str.endswith("_slope")]
        amp_best = amp.iloc[amp["pooled_corr"].abs().argmax()]
        slope_best = slope.iloc[slope["pooled_corr"].abs().argmax()]
        mapping_rows.append({"seed": seed, "role": "amplitude_like_latent", "selected_feature": amp_best["z_feature"], "corr": amp_best["pooled_corr"], "aligned_latent": amp_best["z_feature"].split("_late_mean")[0]})
        mapping_rows.append({"seed": seed, "role": "slope_like_latent", "selected_feature": slope_best["z_feature"], "corr": slope_best["pooled_corr"], "aligned_latent": slope_best["z_feature"].split("_slope")[0]})
    mapping = pd.DataFrame(mapping_rows)
    selected_latents = sorted(mapping["aligned_latent"].unique())
    selected_cols = []
    for z in selected_latents:
        selected_cols += [f"{z}_late_mean", f"{z}_slope", f"{z}_peak", f"{z}_peak_time"]
    return mapping, selected_cols


def residual_and_accumulation(features: pd.DataFrame, z_cols: list[str], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rt_comp, _ = subject_aware_rt_models(features, z_cols, rank=5, output_dir=output_dir)
    pooled = rt_comp[rt_comp["fold"] == "pooled"].set_index("model")
    rows = [
        {
            "seed": int(features["seed"].iloc[0]),
            "test": "z_predicts_RT_residual_after_CPP",
            "approx_delta_cv_r2": float(pooled.loc["CPP + z", "cv_r2"] - pooled.loc["CPP-only", "cv_r2"]),
        },
        {
            "seed": int(features["seed"].iloc[0]),
            "test": "CPP_predicts_RT_residual_after_z",
            "approx_delta_cv_r2": float(pooled.loc["CPP + z", "cv_r2"] - pooled.loc["z-only", "cv_r2"]),
        },
    ]
    accum_rows = []
    for z_base in sorted({c[: -len("_slope")] for c in z_cols if c.endswith("_slope")}):
        accum_rows.append(
            {
                "seed": int(features["seed"].iloc[0]),
                "signal": z_base,
                "diagnostic": "slope_vs_log_RT",
                "pooled_corr": corr(features[f"{z_base}_slope"].to_numpy(), features["log_RT_ms"].to_numpy()),
                "within_subject_mean_corr": float(np.nanmean([corr(sub[f"{z_base}_slope"].to_numpy(), sub["log_RT_ms"].to_numpy()) for _, sub in features.groupby("subject_id")])),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(accum_rows)


def unique_variance_table(features: pd.DataFrame, z_cols: list[str]) -> pd.DataFrame:
    rows = []
    for target in ["CPP_late_mean", "CPP_slope"]:
        y = features[target].to_numpy(float)
        base_cols = [c for c in ["difficulty", "correctness"] if c in features.columns]
        full_cols = z_cols + base_cols
        x = features[full_cols].to_numpy(float)
        model = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=RIDGE_ALPHAS, cv=5))])
        model.fit(x, y)
        pred = model.predict(x)
        rows.append({"seed": int(features["seed"].iloc[0]), "target": target, "model": "selected_CPP_like_latents_plus_covariates", "full_sample_r2": float(r2_score(y, pred)), "rmse": safe_rmse(y, pred), "n_predictors": len(full_cols)})
    return pd.DataFrame(rows)


def correctness_models(features: pd.DataFrame, z_cols: list[str], rank: int) -> pd.DataFrame:
    if features["correctness"].nunique() < 2 or features["correctness"].value_counts().min() < 20:
        return pd.DataFrame([{"seed": int(features["seed"].iloc[0]), "note": "Too few correct/error trials for stable logistic comparison."}])
    z_all = []
    for d in range(1, rank + 1):
        z_all += [f"aligned_z{d}_late_mean", f"aligned_z{d}_slope", f"aligned_z{d}_peak", f"aligned_z{d}_peak_time"]
    specs = {"baseline": [], "CPP-only": CPP_FEATURES, "z-only": z_cols, "CPP + z": CPP_FEATURES + z_cols, "All-z": z_all, "Full": CPP_FEATURES + z_all}
    y = features["correctness"].astype(int).to_numpy()
    groups = features["subject_id"].to_numpy(str)
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    rows = []
    for name, cols in specs.items():
        pred = np.full(len(features), -1)
        for train_idx, test_idx in gkf.split(features, y, groups):
            train, test = features.iloc[train_idx], features.iloc[test_idx]
            x_train, x_test = zscore_train_apply(train, test, cols, include_correctness=False)
            model = Pipeline([("scaler", StandardScaler()), ("logit", LogisticRegression(max_iter=3000, class_weight="balanced"))])
            model.fit(x_train, y[train_idx])
            pred[test_idx] = model.predict(x_test)
        rows.append({"seed": int(features["seed"].iloc[0]), "model": name, "subject_aware_balanced_accuracy": float(balanced_accuracy_score(y, pred))})
    return pd.DataFrame(rows)


def make_plots(output_dir: Path, features_ref: pd.DataFrame, eeg: np.ndarray, z: np.ndarray, times_ms: np.ndarray, corr_all: pd.DataFrame, rt_all: pd.DataFrame, pred_all: pd.DataFrame, mapping: pd.DataFrame) -> None:
    fig_dir = output_dir / "figures"
    cpp = eeg.mean(axis=2)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times_ms, cpp.mean(axis=0), label="CPP", color="black", lw=2)
    for d in range(z.shape[2]):
        ax.plot(times_ms, z[:, :, d].mean(axis=0), label=f"aligned z{d+1}", lw=1)
    ax.axvline(0, color="black", ls=":", lw=0.8)
    ax.legend(ncol=3, fontsize=8)
    ax.set_xlabel("Time from response (ms)")
    ax.set_title("Mean response-aligned CPP and aligned rank-5 latents")
    fig.tight_layout()
    fig.savefig(fig_dir / "mean_response_aligned_cpp_and_rank5_latents.png", dpi=180)
    fig.savefig(fig_dir / "mean_response_aligned_cpp_and_rank5_latents.pdf")
    plt.close(fig)

    heat = corr_all.groupby(["cpp_feature", "z_feature"])["pooled_corr"].mean().reset_index().pivot(index="cpp_feature", columns="z_feature", values="pooled_corr")
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=75, ha="right", fontsize=7)
    ax.set_yticks(range(len(heat.index)), heat.index)
    fig.colorbar(im, ax=ax, label="mean pooled r")
    fig.tight_layout()
    fig.savefig(fig_dir / "cpp_feature_vs_latent_feature_correlation_heatmap.png", dpi=180)
    fig.savefig(fig_dir / "cpp_feature_vs_latent_feature_correlation_heatmap.pdf")
    plt.close(fig)

    pooled = rt_all[rt_all["fold"].astype(str) == "pooled"]
    fig, ax = plt.subplots(figsize=(8, 4))
    order = ["baseline", "CPP-only", "z-only", "CPP + z", "All-z", "Full"]
    data = [pooled[pooled["model"] == m]["cv_r2"].to_numpy(float) for m in order]
    ax.boxplot(data, labels=order, showmeans=True)
    ax.set_ylabel("Subject-aware CV R2")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(fig_dir / "subject_aware_rt_cv_r2_comparison.png", dpi=180)
    fig.savefig(fig_dir / "subject_aware_rt_cv_r2_comparison.pdf")
    plt.close(fig)

    for model_name, fname in [("CPP-only", "observed_vs_predicted_log_rt_cpp_only.png"), ("CPP + z", "observed_vs_predicted_log_rt_cpp_plus_z.png")]:
        sub = pred_all[pred_all["model"] == model_name]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(sub["observed_log_RT"], sub["predicted_log_RT"], s=7, alpha=0.25)
        ax.set_xlabel("Observed log RT")
        ax.set_ylabel("Predicted log RT")
        ax.set_title(model_name)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=180)
        fig.savefig((fig_dir / fname).with_suffix(".pdf"))
        plt.close(fig)

    features_ref = features_ref.copy()
    features_ref["rt_group"] = features_ref.groupby("subject_id")["RT_ms"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["fast", "medium", "slow"]))
    selected = mapping["aligned_latent"].value_counts().index[:2].tolist()
    signals = [("CPP", cpp)] + [(name, z[:, :, int(name.replace("aligned_z", "")) - 1]) for name in selected]
    for name, values in signals:
        fig, ax = plt.subplots(figsize=(8, 4))
        for group in ["fast", "medium", "slow"]:
            subj_means = []
            for _, sub in features_ref[features_ref["rt_group"].astype(str) == group].groupby("subject_id"):
                idx = sub.index.to_numpy()
                if len(idx):
                    subj_means.append(values[idx].mean(axis=0))
            if subj_means:
                y = np.vstack(subj_means).mean(axis=0)
                ax.plot(times_ms, y, label=group)
        ax.axvline(0, color="black", ls=":", lw=0.8)
        ax.set_title(f"{name}: fast / medium / slow")
        ax.set_xlabel("Time from response (ms)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"{name}_fast_medium_slow_trajectories.png", dpi=180)
        fig.savefig(fig_dir / f"{name}_fast_medium_slow_trajectories.pdf")
        plt.close(fig)

    if features_ref["correctness"].nunique() >= 2:
        for name, values in signals:
            fig, ax = plt.subplots(figsize=(8, 4))
            for val, label in [(1.0, "correct"), (0.0, "error")]:
                idx = features_ref.index[features_ref["correctness"] == val].to_numpy()
                if len(idx):
                    ax.plot(times_ms, values[idx].mean(axis=0), label=label)
            ax.axvline(0, color="black", ls=":", lw=0.8)
            ax.set_title(f"{name}: correct / error")
            ax.set_xlabel("Time from response (ms)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / f"{name}_correct_error_trajectories.png", dpi=180)
            fig.savefig(fig_dir / f"{name}_correct_error_trajectories.pdf")
            plt.close(fig)


def write_report(output_dir: Path, seeds: list[int], ref_seed: int, rank: int) -> None:
    data_summary = pd.read_csv(output_dir / "tables" / "data_summary_full.csv")
    train = pd.read_csv(output_dir / "tables" / "training_metrics_by_seed.csv")
    mapping = pd.read_csv(output_dir / "tables" / "seed_level_cpp_z_mapping.csv")
    rt = pd.read_csv(output_dir / "tables" / "rt_model_comparison_subject_aware.csv")
    pooled = rt[rt["fold"].astype(str) == "pooled"]
    cpp_only = pooled[pooled["model"] == "CPP-only"]["cv_r2"].mean()
    cpp_z = pooled[pooled["model"] == "CPP + z"]["cv_r2"].mean()
    delta = cpp_z - cpp_only
    stable = mapping.groupby("role")["aligned_latent"].nunique().max() <= 2
    if delta > 0.01:
        category = "Category B: CPP-like and improves behavior"
        conclusion = "full-data rank-5 latents look CPP-like and add a small amount of subject-aware RT prediction beyond standard CPP features."
    elif not stable:
        category = "Category C: CPP-like but unstable across seeds"
        conclusion = "CPP-like dynamics emerge, but interpretation should focus on the aligned latent subspace rather than raw z indices."
    else:
        category = "Category A: Strong CPP-like but no behavioral improvement"
        conclusion = "the model reliably recovers CPP-like coordinates, but they do not clearly improve RT prediction beyond conventional CPP features."
    lines = [
        "# Full-Data Rank-5 Low-Rank RNN Latent Training Report",
        "",
        "## Executive Summary",
        "",
        f"- Trained rank-{rank} low-rank RNN models on the complete processed dataset for seeds: {seeds}.",
        f"- Reference model for latent alignment: seed {ref_seed}.",
        f"- Mean subject-aware CV R2, CPP-only: {cpp_only:.4f}.",
        f"- Mean subject-aware CV R2, CPP + selected latents: {cpp_z:.4f}.",
        f"- Delta R2 CPP+z minus CPP-only: {delta:.4f}.",
        f"- Conclusion category: {category}.",
        "",
        "## Data Summary",
        "",
    ]
    for _, row in data_summary.iterrows():
        lines.append(f"- {row['metric']}: {row['value']}")
    lines.extend(
        [
            "",
            "The data are response-locked, with response at 0 ms. The model input channels are CP1, CP2, and CPz, so high CPP-latent correlation can partly reflect compression of CPP-related input activity.",
            "",
            "## Training Setup",
            "",
            "The training objective was the existing composite self-supervised loss: EEG reconstruction, one-step future prediction, derivative matching, variance alignment, CPP mean alignment, CPP shape prior terms, and latent smoothness.",
            f"Best validation losses ranged from {train['best_val_loss'].min():.4f} to {train['best_val_loss'].max():.4f}.",
            "",
            "## Validation Strategy",
            "",
            "The final models were trained on the complete valid dataset for descriptive latent interpretation. Behavioral claims use downstream GroupKFold validation by subject, using full-data model latents as extracted features. This tests downstream behavioral generalization, not full RNN retraining on held-out subjects.",
            "",
            "## Latent Alignment and CPP Mapping",
            "",
        ]
    )
    for _, row in mapping.iterrows():
        lines.append(f"- seed {row['seed']} {row['role']}: {row['aligned_latent']} ({row['selected_feature']}, r={row['corr']:.3f})")
    lines.extend(
        [
            "",
            "Raw z indices should not be interpreted before alignment. The report therefore uses labels such as amplitude-like and slope-like latent rather than assuming the smoke-model z1/z4 identities.",
            "",
            "## Behavioral Prediction",
            "",
            f"The key comparison is CPP + z versus CPP-only. Here the average delta R2 was {delta:.4f}. A positive but tiny value should be treated cautiously; a negative or near-zero value means the latents mostly recapitulate CPP information for RT prediction.",
            "",
            "## Accumulation-Like Diagnostics",
            "",
            "The generated figures compare CPP and selected latent trajectories across fast, medium, and slow RT groups, averaging within subject before grand averaging. These plots are descriptive support for response-proximal build-up dynamics.",
            "",
            "## Limitations",
            "",
            "- The available processed data are response-locked, not stimulus-locked.",
            "- CPP channels are model inputs, so CPP-like latents may reflect low-dimensional compression of input CPP activity.",
            "- Subject-aware behavioral validation is downstream validation using extracted latents; full RNN retraining inside each subject fold was not run by default.",
            "- These analyses support interpretability, not causal claims.",
            "",
            "## Final Interpretation",
            "",
            f"After full-data training, {conclusion} The safest wording is that the low-rank RNN learns CPP-like response-proximal latent dynamics, which are candidate low-dimensional accumulation-like coordinates rather than proven evidence-accumulation variables.",
            "",
            "## Recommended Next Step",
            "",
            "Run a smaller number of full subject-held-out RNN retraining folds for the selected configuration, then compare whether the same aligned CPP-like latent subspace appears on held-out subjects.",
        ]
    )
    (output_dir / "reports" / "full_data_rank5_latent_training_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def setup_output_dir(base: Path, no_timestamp: bool) -> Path:
    out = base if no_timestamp else base.with_name(f"{base.name}_{datetime.now().strftime('%Y%m%d_%H%M')}")
    for sub in ["checkpoints", "figures", "tables", "logs", "reports", "latent_exports"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    return out


def run_pipeline(args: argparse.Namespace) -> Path:
    output_dir = setup_output_dir(args.output_dir, args.no_timestamp)
    write_data_summary(args.dataset_dir, output_dir)
    loss = LossWeights(lambda_cpp_prior=args.lambda_cpp_prior, lambda_smooth=args.lambda_smooth)
    config = TrainingConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss=loss,
    )
    (output_dir / "tables" / "training_config_table.csv").write_text(pd.DataFrame([{"rank": args.rank, "population_dim": args.population_dim, **asdict(config), "loss": json.dumps(asdict(loss))}]).to_csv(index=False), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    metrics = [train_seed(args.dataset_dir, output_dir, seed, args.rank, args.population_dim, config, device) for seed in seeds]
    train_df = pd.DataFrame(metrics)
    train_df.to_csv(output_dir / "tables" / "training_metrics_by_seed.csv", index=False)
    train_df.to_csv(output_dir / "tables" / "seed_level_training_metrics.csv", index=False)

    metadata = pd.read_csv(args.dataset_dir / "metadata.csv").reset_index(drop=True)
    aligned, alignment_table, ref_seed = align_latents(output_dir, seeds, args.rank, metadata)
    all_features, all_corr, all_within, all_subject, all_rt, all_pred, all_resid, all_accum, all_unique, all_correct = [], [], [], [], [], [], [], [], [], []
    preliminary_corr = []
    for seed in seeds:
        cur = load_seed_export(output_dir, seed)
        feat = extract_features(metadata, cur["eeg"], aligned[seed], cur["times_ms"], seed)
        corr_table, within_table, subject_table = correlation_tables(feat, args.rank, output_dir)
        preliminary_corr.append(corr_table)
    mapping, selected_cols = select_cpp_like_latents(pd.concat(preliminary_corr, ignore_index=True))
    mapping.to_csv(output_dir / "tables" / "seed_level_cpp_z_mapping.csv", index=False)

    for seed in seeds:
        cur = load_seed_export(output_dir, seed)
        feat = extract_features(metadata, cur["eeg"], aligned[seed], cur["times_ms"], seed)
        feat.to_csv(output_dir / "latent_exports" / f"trial_level_features_seed_{seed}.csv", index=False)
        corr_table, within_table, subject_table = correlation_tables(feat, args.rank, output_dir)
        rt_comp, preds = subject_aware_rt_models(feat, selected_cols, args.rank, output_dir)
        resid, accum = residual_and_accumulation(feat, selected_cols, output_dir)
        unique = unique_variance_table(feat, selected_cols)
        correct = correctness_models(feat, selected_cols, args.rank)
        all_features.append(feat)
        all_corr.append(corr_table)
        all_within.append(within_table)
        all_subject.append(subject_table)
        all_rt.append(rt_comp)
        all_pred.append(preds)
        all_resid.append(resid)
        all_accum.append(accum)
        all_unique.append(unique)
        all_correct.append(correct)

    corr_all = pd.concat(all_corr, ignore_index=True)
    within_all = pd.concat(all_within, ignore_index=True)
    subject_all = pd.concat(all_subject, ignore_index=True)
    rt_all = pd.concat(all_rt, ignore_index=True)
    pred_all = pd.concat(all_pred, ignore_index=True)
    pd.concat(all_features, ignore_index=True).to_csv(output_dir / "latent_exports" / "trial_level_features_all_seeds.csv", index=False)
    corr_all.to_csv(output_dir / "tables" / "cpp_z_correlation_table_full.csv", index=False)
    within_all.to_csv(output_dir / "tables" / "within_subject_correlation_table.csv", index=False)
    subject_all.to_csv(output_dir / "tables" / "subject_level_correlation_table.csv", index=False)
    rt_all.to_csv(output_dir / "tables" / "rt_model_comparison_subject_aware.csv", index=False)
    pred_all.to_csv(output_dir / "tables" / "rt_cv_predictions_subject_aware.csv", index=False)
    pd.concat(all_resid, ignore_index=True).to_csv(output_dir / "tables" / "residual_increment_tests_full.csv", index=False)
    pd.concat(all_accum, ignore_index=True).to_csv(output_dir / "tables" / "accumulation_diagnostics_full.csv", index=False)
    pd.concat(all_unique, ignore_index=True).to_csv(output_dir / "tables" / "unique_variance_cpp_prediction_table.csv", index=False)
    pd.concat(all_correct, ignore_index=True).to_csv(output_dir / "tables" / "correctness_model_comparison_full.csv", index=False)

    robustness_rows = []
    for feature in ["CPP_late_mean", "CPP_late_mean_robust", "CPP_slope", "CPP_slope_broad"]:
        robustness_rows.append({"feature_or_window": feature, "available": True, "note": "Computed in trial-level feature exports; inspect seed-level correlations for robustness."})
    pd.DataFrame(robustness_rows).to_csv(output_dir / "tables" / "window_robustness_table.csv", index=False)

    ref_export = load_seed_export(output_dir, ref_seed)
    ref_feat = all_features[seeds.index(ref_seed)]
    make_plots(output_dir, ref_feat, ref_export["eeg"], aligned[ref_seed], ref_export["times_ms"], corr_all, rt_all, pred_all, mapping)
    write_report(output_dir, seeds, ref_seed, args.rank)
    (output_dir / "reports" / "run_manifest.json").write_text(
        json.dumps({"dataset_dir": str(args.dataset_dir), "output_dir": str(output_dir), "seeds": seeds, "rank": args.rank, "device": str(device)}, indent=2),
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-data low-rank RNN training and validation pipeline.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("Data/ProcessedData"))
    parser.add_argument("--output-dir", type=Path, default=Path("Results/low_rank_full_training_rank5"))
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--population-dim", type=int, default=64)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-cpp-prior", type=float, default=0.05)
    parser.add_argument("--lambda-smooth", type=float, default=0.01)
    args = parser.parse_args()
    out = run_pipeline(args)
    print(out)


if __name__ == "__main__":
    main()
