from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WINDOW_ORDER = [
    "early (-600 to -300 ms)",
    "mid (-300 to -120 ms)",
    "late (-120 to -50 ms)",
    "full (-600 to -50 ms)",
]
CONTRAST_ORDER = [
    "baseline+z - baseline",
    "baseline+cpp+z - baseline+cpp",
    "baseline+shuffled-z - baseline",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_figure(fig: plt.Figure, path_stem: Path) -> None:
    ensure_dir(path_stem.parent)
    fig.savefig(path_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".svg"), bbox_inches="tight")


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def run_paths(run_root: Path, label: str) -> dict[str, Path | str]:
    if label == "no_prior":
        tag = "low_rank_r5_no_cpp_prior"
        latent_dir = "latents_low_rank_r5_no_cpp_prior"
        latent_file = "latents_low_rank_r5_no_cpp_prior.npz"
    else:
        tag = "low_rank_r5"
        latent_dir = "latents_low_rank_r5"
        latent_file = "latents_low_rank_r5.npz"
    return {
        "model_version": label,
        "run_root": run_root,
        "regression": run_root / "Results" / f"{tag}_regression",
        "diagnostics": run_root / "Results" / f"{tag}_diagnostics",
        "latents": run_root / "Data" / "IntermediateData" / latent_dir / latent_file,
    }


def load_dual_tables(cpp_prior_run: Path, no_prior_run: Path) -> dict[str, pd.DataFrame]:
    specs = [
        run_paths(no_prior_run, "no_prior"),
        run_paths(cpp_prior_run, "cpp_prior"),
    ]
    perf_frames = []
    ci_frames = []
    rt_frames = []
    corr_frames = []
    for spec in specs:
        label = str(spec["model_version"])
        regression = Path(spec["regression"])
        diagnostics = Path(spec["diagnostics"])
        perf = pd.read_csv(regression / "ridge_rt_performance.csv")
        perf["model_version"] = label
        ci = pd.read_csv(regression / "ridge_delta_r2_ci.csv")
        ci["model_version"] = label
        rt = pd.read_csv(diagnostics / "z_rt_time_resolved_correlation.csv")
        rt["model_version"] = label
        corr = pd.read_csv(diagnostics / "z_cpp_behavior_correlation_matrix.csv")
        corr["model_version"] = label
        perf_frames.append(perf)
        ci_frames.append(ci)
        rt_frames.append(rt)
        corr_frames.append(corr)
    return {
        "performance": pd.concat(perf_frames, ignore_index=True),
        "delta_ci": pd.concat(ci_frames, ignore_index=True),
        "z_rt": pd.concat(rt_frames, ignore_index=True),
        "z_cpp_behavior": pd.concat(corr_frames, ignore_index=True),
    }


def summarize_delta_consistency(ci: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pivot = ci.pivot_table(index=["window", "contrast"], columns="model_version", values="mean_delta_r2", aggfunc="first")
    for (window, contrast), row in pivot.iterrows():
        no_prior = float(row.get("no_prior", np.nan))
        cpp_prior = float(row.get("cpp_prior", np.nan))
        same_direction = bool(np.sign(no_prior) == np.sign(cpp_prior)) if np.isfinite(no_prior) and np.isfinite(cpp_prior) else False
        rows.append(
            {
                "window": window,
                "contrast": contrast,
                "no_prior_delta_r2": no_prior,
                "cpp_prior_delta_r2": cpp_prior,
                "absolute_difference": float(abs(no_prior - cpp_prior)) if np.isfinite(no_prior) and np.isfinite(cpp_prior) else np.nan,
                "same_direction": same_direction,
            }
        )
    return pd.DataFrame(rows)


def summarize_strongest_z_rt(z_rt: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    analysis = z_rt[(z_rt["time_ms"] >= -600.0) & (z_rt["time_ms"] <= -50.0)].copy()
    for label, group in analysis.groupby("model_version"):
        row = group.loc[group["correlation_with_log_rt"].abs().idxmax()]
        rows.append(
            {
                "model_version": label,
                "latent": row["latent"],
                "time_ms": float(row["time_ms"]),
                "correlation_with_log_rt": float(row["correlation_with_log_rt"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_z_cpp(corr_df: pd.DataFrame) -> pd.DataFrame:
    target_cols = [c for c in ["CPP amplitude", "CPP slope", "Late CPP amplitude", "RT"] if c in corr_df.columns]
    rows: list[dict[str, Any]] = []
    for label, group in corr_df.groupby("model_version"):
        for target in target_cols:
            sub = group[["latent", target]].copy()
            row = sub.loc[sub[target].abs().idxmax()]
            rows.append(
                {
                    "model_version": label,
                    "target": target,
                    "latent": row["latent"],
                    "correlation": float(row[target]),
                }
            )
    return pd.DataFrame(rows)


def plot_delta_forest(ci: pd.DataFrame, output_dir: Path) -> None:
    models = ["no_prior", "cpp_prior"]
    colors = {"no_prior": "#0072B2", "cpp_prior": "#D55E00"}
    sub = ci[ci["contrast"].isin(["baseline+z - baseline", "baseline+cpp+z - baseline+cpp"])].copy()
    y_labels: list[str] = []
    positions: list[tuple[str, str, int]] = []
    pos = 0
    for window in WINDOW_ORDER:
        for contrast in ["baseline+z - baseline", "baseline+cpp+z - baseline+cpp"]:
            y_labels.append(f"{window}\n{contrast}")
            positions.append((window, contrast, pos))
            pos += 1
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for model_label, offset in [("no_prior", -0.11), ("cpp_prior", 0.11)]:
        for window, contrast, y in positions:
            row = sub[(sub["window"] == window) & (sub["contrast"] == contrast) & (sub["model_version"] == model_label)]
            if row.empty:
                continue
            row = row.iloc[0]
            ax.errorbar(
                row["mean_delta_r2"],
                y + offset,
                xerr=[[row["mean_delta_r2"] - row["ci_lower"]], [row["ci_upper"] - row["mean_delta_r2"]]],
                fmt="o",
                color=colors[model_label],
                elinewidth=1.0,
                capsize=2,
                markersize=4,
                label=model_label if y == 0 else None,
            )
    ax.axvline(0, color="0.2", linewidth=0.8, alpha=0.7)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Incremental cross-validated R², ΔR²")
    ax.set_title("Rank-5 no-prior vs CPP-prior RT prediction")
    clean_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_dir / "dual_prior_delta_r2_forestplot")
    plt.close(fig)


def plot_z_rt_overlay(z_rt: pd.DataFrame, output_dir: Path) -> None:
    colors = {"no_prior": "#0072B2", "cpp_prior": "#D55E00"}
    fig, axes = plt.subplots(5, 1, figsize=(5.4, 6.8), sharex=True, sharey=True)
    for k, ax in enumerate(axes, start=1):
        latent = f"z{k}"
        for label in ["no_prior", "cpp_prior"]:
            sub = z_rt[(z_rt["model_version"] == label) & (z_rt["latent"] == latent)]
            ax.plot(sub["time_ms"], sub["correlation_with_log_rt"], color=colors[label], linewidth=1.1, label=label if k == 1 else None)
        ax.axhline(0, color="0.8", linewidth=0.8)
        ax.axvspan(-600, -50, color="0.92", alpha=0.6)
        ax.axvline(0, color="0.2", linewidth=0.8, alpha=0.55)
        ax.set_ylabel(latent)
        clean_axes(ax)
    axes[0].legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel("Time from response (ms)")
    fig.suptitle("z-RT correlations: no-prior vs CPP-prior")
    fig.tight_layout()
    save_figure(fig, output_dir / "dual_prior_z_rt_time_resolved_overlay")
    plt.close(fig)


def plot_z_cpp_heatmaps(corr_df: pd.DataFrame, output_dir: Path) -> None:
    target_cols = [c for c in ["CPP amplitude", "CPP slope", "Late CPP amplitude", "RT", "difficulty", "evidence_strength", "correctness"] if c in corr_df.columns]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 2.9), sharey=True)
    for ax, label in zip(axes, ["no_prior", "cpp_prior"]):
        sub = corr_df[corr_df["model_version"] == label].set_index("latent")[target_cols]
        image = ax.imshow(sub.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_title(label)
        ax.set_xticks(np.arange(len(target_cols)))
        ax.set_xticklabels(target_cols, rotation=40, ha="right", fontsize=7)
        ax.set_yticks(np.arange(sub.shape[0]))
        ax.set_yticklabels(sub.index)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(image, ax=axes, shrink=0.75)
    cbar.set_label("Pearson r")
    fig.suptitle("z, CPP, and behaviour correlations")
    fig.tight_layout()
    save_figure(fig, output_dir / "dual_prior_z_cpp_behavior_heatmaps")
    plt.close(fig)


def build_interpretation(delta_summary: pd.DataFrame, strongest_rt: pd.DataFrame, strongest_cpp: pd.DataFrame) -> str:
    primary = delta_summary[delta_summary["contrast"].isin(["baseline+z - baseline", "baseline+cpp+z - baseline+cpp"])]
    same_direction_rate = float(primary["same_direction"].mean()) if len(primary) else float("nan")
    cpp_control = primary[primary["contrast"] == "baseline+cpp+z - baseline+cpp"].copy()
    both_positive_cpp_control = bool(((cpp_control["no_prior_delta_r2"] > 0) & (cpp_control["cpp_prior_delta_r2"] > 0)).any())
    shuffled = delta_summary[delta_summary["contrast"] == "baseline+shuffled-z - baseline"].copy()
    shuffled_near_zero = bool(
        (shuffled["no_prior_delta_r2"].abs().max() < 0.005)
        and (shuffled["cpp_prior_delta_r2"].abs().max() < 0.005)
    ) if len(shuffled) else False
    if same_direction_rate >= 0.75 and shuffled_near_zero:
        headline = "The two Rank-5 z versions show broadly consistent RT-prediction patterns, with shuffled-z behaving like a control."
    else:
        headline = "The two Rank-5 z versions are not uniformly consistent, so model-version-specific interpretation is needed."
    if both_positive_cpp_control:
        cpp_sentence = "At least one window shows positive baseline+CPP+z improvement over baseline+CPP in both versions."
    else:
        cpp_sentence = "CPP-controlled z increments are not consistently positive in both versions."
    lines = [
        "### Dual-prior Rank-5 interpretation",
        "",
        f"- Direction agreement across primary RT contrasts: `{same_direction_rate:.3f}`.",
        f"- Shuffled-z control near zero in both versions: `{shuffled_near_zero}`.",
        f"- {cpp_sentence}",
        f"- {headline}",
        "",
        "Cautious interpretation: use no-prior z as the cleaner representation and CPP-prior z as the theory-guided comparison. Emphasize findings that are directionally stable across both versions and survive CPP and shuffled-z controls.",
    ]
    return "\n".join(lines)


def run_comparison(cpp_prior_run: Path, no_prior_run: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    tables = load_dual_tables(cpp_prior_run, no_prior_run)
    for name, df in tables.items():
        save_dataframe(output_dir / f"dual_prior_{name}.csv", df)
    delta_summary = summarize_delta_consistency(tables["delta_ci"])
    strongest_rt = summarize_strongest_z_rt(tables["z_rt"])
    strongest_cpp = summarize_z_cpp(tables["z_cpp_behavior"])
    save_dataframe(output_dir / "dual_prior_delta_consistency.csv", delta_summary)
    save_dataframe(output_dir / "dual_prior_strongest_z_rt.csv", strongest_rt)
    save_dataframe(output_dir / "dual_prior_strongest_z_cpp_behavior.csv", strongest_cpp)
    plot_delta_forest(tables["delta_ci"], output_dir)
    plot_z_rt_overlay(tables["z_rt"], output_dir)
    plot_z_cpp_heatmaps(tables["z_cpp_behavior"], output_dir)
    interpretation = build_interpretation(delta_summary, strongest_rt, strongest_cpp)
    (output_dir / "dual_prior_interpretation.md").write_text(interpretation, encoding="utf-8")
    summary = {
        "cpp_prior_run": str(cpp_prior_run),
        "no_prior_run": str(no_prior_run),
        "output_dir": str(output_dir),
        "drift_rate_status": "not_run_no_drift_rate_column_found_in_current_metadata",
        "primary_windows": WINDOW_ORDER,
        "outputs": sorted(path.name for path in output_dir.glob("*")),
    }
    write_json(output_dir / "dual_prior_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Rank-5 no-prior and CPP-prior z analyses.")
    parser.add_argument("--cpp-prior-run", type=Path, required=True)
    parser.add_argument("--no-prior-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("Results/rank5_dual_prior_comparison"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_comparison(args.cpp_prior_run, args.no_prior_run, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
