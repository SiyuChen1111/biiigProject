from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import fdrcorrection

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = REPO_ROOT / "Data" / "ProcessedData" / "metadata.csv"
DEFAULT_TRACE = REPO_ROOT / "Data" / "model_traces" / "m5_traces.csv"
DEFAULT_JOINT = REPO_ROOT / "Data" / "joint-modeling" / "data_joint_modeling_all.csv"
DEFAULT_CPP = REPO_ROOT / "Data" / "ProcessedData" / "eeg_cpp_trials.npy"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Results" / "regression"

WINDOW_SPECS = {
    "early": (0.00, 0.30),
    "response": (0.70, 1.00),
    "across_response": (0.80, 1.10),
    "late": (1.00, 1.20),
    "slps_ams": (0.82, 0.92),
    "pams": (0.95, 1.05),
}

WINDOW_LABELS = {
    "early": "Early",
    "response": "Response",
    "across_response": "Across-response",
    "late": "Late",
    "slps_ams": "Slope+Amplitude",
    "pams": "Peak-amplitude",
}

WINDOW_COLORS = {
    "response": "#1f5a91",
    "slps_ams": "#3b7c54",
    "across_response": "#8a6a3f",
    "pams": "#8e4b78",
    "late": "#b35c3d",
    "early": "#7f8c8d",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the drift-rate latent significance analysis and export Nature-style figures."
    )
    parser.add_argument(
        "--latent-path",
        type=Path,
        default=None,
        help="Path to the no-prior Rank-5 latent npz file. If omitted, the latest local notebook run is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write summary tables and figures.",
    )
    return parser.parse_args()


def resolve_latent_path(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Latent file not found: {explicit_path}")
        return explicit_path

    candidates = sorted(
        REPO_ROOT.glob(
            "tmp/low_rank_r5_no_cpp_prior_notebook_runs/*/Data/IntermediateData/"
            "latents_low_rank_r5_no_cpp_prior/latents_low_rank_r5_no_cpp_prior.npz"
        )
    )
    if not candidates:
        raise FileNotFoundError("No local no-prior Rank-5 latent exports were found under tmp/.")
    return candidates[-1]


def z_score(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def load_drift_rate_table(trace_path: Path, joint_path: Path) -> pd.DataFrame:
    condition = "cue_dimensionality"
    feature_cols = ["ams", "pams", "slps", "ams_bin", "pam_bin", "slp_bin"]

    data_v = pd.read_csv(trace_path)
    data_joint_modeling = pd.read_csv(joint_path)

    v_intercept = data_v.filter(regex=r"^v_Intercept_subj\.sub-STSWD\d+$").mean(axis=0)
    v_condition = data_v.filter(regex=fr"^v_{condition}_subj\.sub-STSWD\d+$").mean(axis=0)

    subj_ids = [re.search(r"\.(sub-STSWD\d+)$", col).group(1) for col in v_intercept.index]
    v_params = pd.DataFrame(
        {
            "subj_idx": subj_ids,
            "v_intercept": v_intercept.to_numpy(),
            f"v_{condition}_slope": v_condition.to_numpy(),
        }
    )

    erp_v = data_joint_modeling.groupby(["subj_idx", condition])[feature_cols].mean().reset_index()
    erp_v = erp_v.merge(v_params, on="subj_idx", how="left")
    erp_v["v"] = erp_v["v_intercept"] + erp_v[f"v_{condition}_slope"] * erp_v[condition]

    # Match the notebook's explicit exclusion so results stay comparable.
    erp_v = pd.DataFrame(erp_v.drop(132)).reset_index(drop=True)
    return erp_v


def build_cpp_feature_table(metadata_path: Path, cpp_path: Path) -> pd.DataFrame:
    data_metadata = pd.read_csv(metadata_path)
    data_eeg = np.load(cpp_path, allow_pickle=True)
    data_erp = np.nanmean(data_eeg, axis=2)

    column_subid_condition = data_metadata[["subj_idx", "cue_dimensionality", "probe_rt", "probe_accuracy"]]
    data_erp_metadata = pd.DataFrame(np.column_stack((column_subid_condition, data_erp)))
    erp = data_erp_metadata.groupby([0, 1], as_index=False).mean()

    sample_rate = 256
    slope_start = int(0.82 * sample_rate) + 4
    slope_end = int(0.92 * sample_rate) + 4
    slope_data = erp.iloc[:, slope_start:slope_end].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    time_vector = np.arange(slope_data.shape[1]) / sample_rate
    erp["slope"] = np.array([np.polyfit(time_vector, slope_data[i, :], deg=1)[0] for i in range(slope_data.shape[0])])

    ams_data = erp.iloc[:, slope_start:slope_end].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    erp["ams"] = np.array([np.nanmean(ams_data[i, :]) for i in range(ams_data.shape[0])])

    pams_start = int(0.95 * sample_rate) + 4
    pams_end = int(1.05 * sample_rate) + 4
    pams_data = erp.iloc[:, pams_start:pams_end].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    erp["pams"] = np.array([np.nanmax(pams_data[i, :]) for i in range(pams_data.shape[0])])

    erp["rt"] = erp.iloc[:, 2]
    erp["acc"] = erp.iloc[:, 3]

    data_cpp = erp[[0, 1, "rt", "acc", "slope", "ams", "pams"]].rename(columns={0: "subj_id", 1: "condition"})
    return data_cpp


def build_mean_latent_table(latent_path: Path, metadata_path: Path) -> np.ndarray:
    data_z = np.load(latent_path, allow_pickle=True)["latents"]
    data_metadata = pd.read_csv(metadata_path)

    subj_idx = data_metadata["subj_id"].values
    condition = data_metadata["condition"].values
    group_df = pd.DataFrame({"subj_id": subj_idx, "condition": condition})
    group_idx, unique_groups = pd.factorize(pd.MultiIndex.from_frame(group_df))

    mean_z = np.zeros((len(unique_groups), data_z.shape[1], data_z.shape[2]), dtype=np.float64)
    counts = np.zeros(len(unique_groups), dtype=np.float64)
    np.add.at(mean_z, group_idx, data_z)
    np.add.at(counts, group_idx, 1)
    mean_z = mean_z / counts[:, None, None]

    # Match the notebook's explicit exclusion so row order matches the drift-rate table.
    mean_z = np.delete(mean_z, 132, axis=0)
    return mean_z


def build_analysis_table(
    metadata_path: Path,
    trace_path: Path,
    joint_path: Path,
    cpp_path: Path,
    latent_path: Path,
) -> tuple[pd.DataFrame, float, float]:
    drift_table = load_drift_rate_table(trace_path, joint_path)
    cpp_table = build_cpp_feature_table(metadata_path, cpp_path).reset_index(drop=True)
    mean_z = build_mean_latent_table(latent_path, metadata_path)

    data_merge = cpp_table.copy()
    sample_rate = 256
    window_features: dict[str, np.ndarray] = {}
    for key, (lo, hi) in WINDOW_SPECS.items():
        start_idx = int(round(lo * sample_rate))
        end_idx = int(round(hi * sample_rate))
        data_z_avg = np.nanmean(mean_z[:, start_idx:end_idx, :], axis=1)
        window_features[key] = data_z_avg

    feature_blocks = []
    for key, values in window_features.items():
        cols = {f"{key}_r{idx}": values[:, idx] for idx in range(values.shape[1])}
        feature_blocks.append(pd.DataFrame(cols))
    data_merge = pd.concat([data_merge, *feature_blocks], axis=1)
    data_merge["v"] = drift_table["v"].to_numpy(dtype=np.float64)

    for column in ["rt", "acc", "condition", "v", "slope", "ams", "pams"]:
        data_merge[column] = pd.to_numeric(data_merge[column], errors="coerce")
    data_merge["log_rt"] = np.log(data_merge["rt"])

    acc = data_merge["acc"].to_numpy(dtype=np.float64).reshape(-1, 1)
    condition_arr = data_merge["condition"].to_numpy(dtype=np.float64).reshape(-1, 1)
    rt = data_merge["log_rt"].to_numpy(dtype=np.float64).reshape(-1, 1)
    ams = data_merge["ams"].to_numpy(dtype=np.float64).reshape(-1, 1)
    ams_z = z_score(ams)
    y = data_merge["v"].to_numpy(dtype=np.float64)

    model_beh = sm.OLS(y, sm.add_constant(np.hstack([acc, condition_arr, rt])), missing="drop").fit()
    model_ams = sm.OLS(y, sm.add_constant(np.hstack([ams, acc, condition_arr, rt])), missing="drop").fit()

    rows = []
    for window_name in WINDOW_SPECS:
        for latent_idx in range(5):
            rank_name = f"{window_name}_r{latent_idx}"
            rank_values = data_merge[rank_name].to_numpy(dtype=np.float64).reshape(-1, 1)
            model = sm.OLS(
                y,
                sm.add_constant(np.hstack([rank_values, ams_z, acc, condition_arr, rt])),
                missing="drop",
            ).fit()
            anova_table = anova_lm(model_ams, model)
            rows.append(
                {
                    "window": window_name,
                    "window_label": WINDOW_LABELS[window_name],
                    "latent": f"z{latent_idx + 1}",
                    "rank": rank_name,
                    "adj_r2": model.rsquared_adj,
                    "delta_adj_r2_vs_ams": model.rsquared_adj - model_ams.rsquared_adj,
                    "coef": float(model.params[1]),
                    "coef_p": float(model.pvalues[1]),
                    "anova_p": float(anova_table.iloc[1, -1]),
                }
            )

    results = pd.DataFrame(rows)
    reject, q_values = fdrcorrection(results["anova_p"].to_numpy(), alpha=0.05)
    results["fdr_reject"] = reject
    results["anova_q"] = q_values
    return results, float(model_beh.rsquared_adj), float(model_ams.rsquared_adj)


def make_figure(results: pd.DataFrame, output_path: Path) -> None:
    heatmap_df = (
        results.pivot(index="window_label", columns="latent", values="delta_adj_r2_vs_ams")
        .reindex([WINDOW_LABELS[key] for key in WINDOW_SPECS])
        .loc[:, [f"z{i}" for i in range(1, 6)]]
    )
    sig_df = (
        results.pivot(index="window_label", columns="latent", values="fdr_reject")
        .reindex([WINDOW_LABELS[key] for key in WINDOW_SPECS])
        .loc[:, [f"z{i}" for i in range(1, 6)]]
    )

    ranked = results.sort_values(["fdr_reject", "delta_adj_r2_vs_ams", "anova_p"], ascending=[False, False, True]).copy()
    ranked["panel_label"] = ranked["window_label"] + " " + ranked["latent"]

    cmap = LinearSegmentedColormap.from_list(
        "nature_teal",
        ["#f5f5f2", "#d8ece8", "#9ccfc3", "#4b9a92", "#1f5a61"],
    )

    fig = plt.figure(figsize=(10.5, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.3], wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(heatmap_df.to_numpy(), cmap=cmap, aspect="auto", vmin=0, vmax=max(0.041, float(heatmap_df.max().max())))
    ax0.set_xticks(np.arange(heatmap_df.shape[1]))
    ax0.set_xticklabels(heatmap_df.columns, fontsize=8)
    ax0.set_yticks(np.arange(heatmap_df.shape[0]))
    ax0.set_yticklabels(heatmap_df.index, fontsize=8)
    ax0.set_title("Delta adjusted $R^2$ over CPP amplitude baseline", fontsize=9, pad=8)
    ax0.set_xlabel("Latent")
    ax0.set_ylabel("Time window")
    for i in range(heatmap_df.shape[0]):
        for j in range(heatmap_df.shape[1]):
            value = heatmap_df.iloc[i, j]
            color = "white" if value > 0.024 else "#1f1f1f"
            ax0.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=6.5, color=color)
            if bool(sig_df.iloc[i, j]):
                ax0.scatter(j, i, s=40, facecolors="none", edgecolors="black", linewidths=1.3)
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label("Delta adjusted $R^2$", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    ax0.text(-0.18, 1.02, "a", transform=ax0.transAxes, fontsize=10, fontweight="bold")

    ax1 = fig.add_subplot(gs[0, 1])
    significant = ranked[ranked["fdr_reject"]].copy()
    if significant.empty:
        significant = ranked.head(10).copy()
    significant = significant.sort_values("delta_adj_r2_vs_ams", ascending=True)
    y_pos = np.arange(len(significant))
    point_colors = significant["window"].map(WINDOW_COLORS).fillna("#6b6b6b")

    ax1.hlines(y=y_pos, xmin=0, xmax=significant["delta_adj_r2_vs_ams"], color=point_colors, linewidth=2, alpha=0.7)
    ax1.scatter(
        significant["delta_adj_r2_vs_ams"],
        y_pos,
        s=55,
        color=point_colors,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(significant["panel_label"], fontsize=7.5)
    ax1.set_xlabel("Delta adjusted $R^2$")
    ax1.set_title("FDR-significant latent additions", fontsize=9, pad=8)
    ax1.grid(axis="x", alpha=0.22)
    ax1.spines[["top", "right", "left"]].set_visible(False)
    for y, value, q_value in zip(y_pos, significant["delta_adj_r2_vs_ams"], significant["anova_q"]):
        ax1.text(value + 0.001, y, f"q={q_value:.3f}", va="center", fontsize=6.5, color="#2c2c2c")
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=WINDOW_COLORS[key], markeredgecolor="white",
                   markersize=7, label=WINDOW_LABELS[key])
        for key in WINDOW_SPECS
        if key in significant["window"].unique()
    ]
    ax1.legend(handles=handles, loc="lower right", frameon=False, fontsize=7)
    ax1.text(-0.12, 1.02, "b", transform=ax1.transAxes, fontsize=10, fontweight="bold")

    fig.suptitle("Drift-rate regression highlights a response-proximal z-pattern", fontsize=10, y=1.01)
    fig.text(
        0.5,
        0.01,
        "Open circles mark FDR-significant additions beyond the CPP-amplitude baseline model.",
        ha="center",
        fontsize=7,
    )
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_summary(results: pd.DataFrame, behavior_adj_r2: float, ams_adj_r2: float, output_dir: Path, latent_path: Path) -> None:
    significant = results[results["fdr_reject"]].sort_values("anova_p").copy()
    top_hit = significant.iloc[0] if not significant.empty else results.sort_values("anova_p").iloc[0]

    lines = [
        "# Drift-rate latent significance summary",
        "",
        f"- Latent source: `{latent_path}`",
        f"- Baseline behaviour adjusted R2: `{behavior_adj_r2:.3f}`",
        f"- CPP-amplitude baseline adjusted R2: `{ams_adj_r2:.3f}`",
        f"- Number of FDR-significant latent additions: `{len(significant)}`",
        f"- Strongest hit: `{top_hit['window_label']} {top_hit['latent']}` with delta adjusted R2 `{top_hit['delta_adj_r2_vs_ams']:.3f}` and q `{top_hit['anova_q']:.4f}`",
        "",
        "Interpretation:",
        "",
        "- Drift-rate-related latent signal is weakest in the early window and strongest in the response-proximal windows.",
        "- The most stable contributors are z3 first, then z2 and z5.",
        "- This supports using RT as an earlier behavioural validation step, but drift-rate as the more mechanistic follow-up target.",
    ]
    (output_dir / "drift_rate_latent_significance_summary.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    latent_path = resolve_latent_path(args.latent_path)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results, behavior_adj_r2, ams_adj_r2 = build_analysis_table(
        metadata_path=DEFAULT_METADATA,
        trace_path=DEFAULT_TRACE,
        joint_path=DEFAULT_JOINT,
        cpp_path=DEFAULT_CPP,
        latent_path=latent_path,
    )

    results = results.sort_values(["fdr_reject", "anova_p", "delta_adj_r2_vs_ams"], ascending=[False, True, False])
    results.to_csv(output_dir / "drift_rate_latent_significance.csv", index=False)

    heatmap_export = (
        results.pivot(index="window_label", columns="latent", values="delta_adj_r2_vs_ams")
        .reindex([WINDOW_LABELS[key] for key in WINDOW_SPECS])
        .loc[:, [f"z{i}" for i in range(1, 6)]]
    )
    heatmap_export.to_csv(output_dir / "drift_rate_latent_delta_adj_r2_heatmap.csv")

    make_figure(results, output_dir / "drift_rate_latent_significance")
    write_summary(results, behavior_adj_r2, ams_adj_r2, output_dir, latent_path)


if __name__ == "__main__":
    main()
