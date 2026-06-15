from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from run_neural_validation_audit import OUT_DIR, WINDOWS, load_model_predictions


FIG_DIR = OUT_DIR / "figures" / "publication_style"
PLAN_PATH = OUT_DIR / "publication_figure_plan.md"
SUMMARY_PATH = OUT_DIR / "publication_summary_note.md"

WINDOW_ORDER = [
    "minus600_to_minus300",
    "minus300_to_minus120",
    "minus120_to_minus50",
    "minus600_to_minus50",
]
WINDOW_LABELS = {
    "minus600_to_minus300": "-600 to -300 ms",
    "minus300_to_minus120": "-300 to -120 ms",
    "minus120_to_minus50": "-120 to -50 ms",
    "minus600_to_minus50": "-600 to -50 ms",
}
GROUP_ORDERS = {
    "condition": [1, 2, 3, 4],
    "correctness": [1, 0],
    "difficulty": [1, 2, 3, 4],
}
GROUP_LABELS = {
    "condition": {1: "Condition 1", 2: "Condition 2", 3: "Condition 3", 4: "Condition 4"},
    "correctness": {1: "Correct", 0: "Error"},
    "difficulty": {1: "Difficulty 1", 2: "Difficulty 2", 3: "Difficulty 3", 4: "Difficulty 4"},
}
GROUP_COLORS = {
    "condition": ["#1f3a5f", "#4f6d7a", "#9a6b4f", "#c08a52"],
    "correctness": ["#1f3a5f", "#b55d4c"],
    "difficulty": ["#264653", "#3f6f7a", "#8c6d46", "#c49a5a"],
}
TARGET_LABELS = {
    "choice": "Choice",
    "condition": "Condition",
    "correctness": "Correctness",
    "difficulty": "Difficulty",
    "rt_bin": "RT bin",
    "arrangement_probe_leftrightwin": "Arrangement",
}
REG_TARGET_LABELS = {
    "cpp_amp_minus600_to_minus50": "CPP amplitude",
    "cpp_slope_minus600_to_minus50": "CPP slope",
}
BEHAVIOR_ORDER = [
    "behavior_only",
    "cpp_features_only",
    "hidden_states_only",
    "behavior_plus_cpp",
    "behavior_plus_hidden",
    "behavior_plus_cpp_plus_hidden",
    "behavior_plus_shuffled_hidden",
]
BEHAVIOR_LABELS = {
    "behavior_only": "Behavior only",
    "cpp_features_only": "CPP only",
    "hidden_states_only": "Hidden only",
    "behavior_plus_cpp": "Behavior + CPP",
    "behavior_plus_hidden": "Behavior + hidden",
    "behavior_plus_cpp_plus_hidden": "Behavior + CPP + hidden",
    "behavior_plus_shuffled_hidden": "Behavior + shuffled hidden",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300)
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(colors="#333333")


def mean_sem(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    sem = np.nanstd(x, axis=0, ddof=1) / np.sqrt(max(x.shape[0], 1))
    return mean, sem


def response_locked_mask(times_ms: np.ndarray, lo: float = -600.0, hi: float = 0.0) -> np.ndarray:
    return (times_ms >= lo) & (times_ms <= hi)


def plot_waveform_with_band(
    ax: plt.Axes,
    times_ms: np.ndarray,
    data: np.ndarray,
    color: str,
    label: str,
    linestyle: str = "-",
    alpha_fill: float = 0.14,
) -> None:
    mean, sem = mean_sem(data)
    ax.plot(times_ms, mean, color=color, linewidth=2.0, linestyle=linestyle, label=label)
    ax.fill_between(times_ms, mean - sem, mean + sem, color=color, alpha=alpha_fill, linewidth=0)


def make_waveform_figure() -> tuple[float, float]:
    eeg, pred, metadata, times_ms, artifacts = load_model_predictions()
    test_idx = artifacts.test_indices
    empirical_cpp = eeg[test_idx].mean(axis=2)
    recon_cpp = pred[test_idx].mean(axis=2)
    meta = metadata.iloc[test_idx].reset_index(drop=True).copy()
    mask_main = response_locked_mask(times_ms, -600.0, 0.0)
    times_main = times_ms[mask_main]
    empirical_main = empirical_cpp[:, mask_main]
    recon_main = recon_cpp[:, mask_main]

    y_all = [empirical_main, recon_main]
    for group_name in GROUP_ORDERS:
        for group_value in GROUP_ORDERS[group_name]:
            idx = meta[group_name].to_numpy() == group_value
            if idx.sum() > 1:
                y_all.append(empirical_main[idx])
                y_all.append(recon_main[idx])
    y_min = min(float(np.nanmin(x)) for x in y_all)
    y_max = max(float(np.nanmax(x)) for x in y_all)
    pad = 0.08 * (y_max - y_min)
    y_limits = (y_min - pad, y_max + pad)

    fig = plt.figure(figsize=(11.0, 11.6))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1.0, 1.0, 1.0], hspace=0.4, wspace=0.22)

    ax_main = fig.add_subplot(gs[0, :])
    plot_waveform_with_band(ax_main, times_main, empirical_main, "#1f3a5f", "Empirical", "-")
    plot_waveform_with_band(ax_main, times_main, recon_main, "#c67c47", "Reconstruction", "--")
    ax_main.axvspan(-50, 0, color="#d9d9d9", alpha=0.25, zorder=0)
    ax_main.axvline(0, color="#4c4c4c", linestyle=(0, (3, 3)), linewidth=1.0)
    ax_main.set_xlim(-600, 0)
    ax_main.set_ylim(*y_limits)
    ax_main.set_xlabel("Time from response (ms)")
    ax_main.set_ylabel("CPP signal")
    ax_main.set_title("Grand-average response-locked waveform")
    ax_main.legend(frameon=False, ncol=2, loc="upper left")
    clean_axis(ax_main)
    add_panel_label(ax_main, "a")
    ax_main.text(
        0.99,
        0.04,
        "Shaded window: response-proximal region",
        transform=ax_main.transAxes,
        ha="right",
        va="bottom",
        color="#666666",
        fontsize=8,
    )

    inset = inset_axes(ax_main, width="28%", height="38%", loc="lower left", borderpad=1.4)
    plot_waveform_with_band(inset, times_ms, empirical_cpp, "#1f3a5f", "Empirical", "-")
    plot_waveform_with_band(inset, times_ms, recon_cpp, "#c67c47", "Reconstruction", "--")
    inset.axvspan(-1000, -600, color="#efefef", alpha=0.8, zorder=0)
    inset.axvspan(-50, 0, color="#d9d9d9", alpha=0.25, zorder=0)
    inset.axvline(0, color="#4c4c4c", linestyle=(0, (3, 3)), linewidth=0.8)
    inset.set_xlim(-1000, 200)
    inset.set_title("Full window", fontsize=8, pad=2)
    inset.tick_params(labelsize=7)
    clean_axis(inset)

    panel_axes = {
        "condition": (fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])),
        "correctness": (fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])),
        "difficulty": (fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])),
    }
    panel_labels = {"condition": "b", "correctness": "c", "difficulty": "d"}

    for group_name, (ax_emp, ax_model) in panel_axes.items():
        values = GROUP_ORDERS[group_name]
        colors = GROUP_COLORS[group_name]
        for value, color in zip(values, colors):
            idx = meta[group_name].to_numpy() == value
            if idx.sum() < 2:
                continue
            label = GROUP_LABELS[group_name][value]
            plot_waveform_with_band(ax_emp, times_main, empirical_main[idx], color, label, "-")
            plot_waveform_with_band(ax_model, times_main, recon_main[idx], color, label, "--")

        for ax in (ax_emp, ax_model):
            ax.axvspan(-50, 0, color="#d9d9d9", alpha=0.25, zorder=0)
            ax.axvline(0, color="#4c4c4c", linestyle=(0, (3, 3)), linewidth=1.0)
            ax.set_xlim(-600, 0)
            ax.set_ylim(*y_limits)
            ax.set_xlabel("Time from response (ms)")
            clean_axis(ax)

        ax_emp.set_title(f"{group_name.capitalize()}: empirical")
        ax_model.set_title(f"{group_name.capitalize()}: model")
        ax_emp.set_ylabel("CPP signal")
        ax_model.legend(frameon=False, loc="upper left")
        add_panel_label(ax_emp, panel_labels[group_name])

    fig.suptitle("Main Figure 1. Neural reconstruction in the pre-response window", x=0.06, y=0.99, ha="left", fontsize=12)
    save_figure(fig, "main_figure_1_neural_reconstruction")
    return y_limits


def make_shared_scale_windowed_waveform_figure() -> None:
    eeg, pred, _, times_ms, artifacts = load_model_predictions()
    test_idx = artifacts.test_indices
    real_mean = eeg[test_idx].mean(axis=0)
    recon_mean = pred[test_idx].mean(axis=0)
    ch_names = ["CP1", "CP2", "CPz"]
    real_colors = ["#1f3a5f", "#4f6d7a", "#8b6fb3"]
    recon_colors = ["#c67c47", "#d8574d", "#8f6a5a"]

    shared_scale_mask = (times_ms >= -600.0) & (times_ms <= 0.0)
    plot_mask = times_ms >= -600.0
    scale_values = np.concatenate([real_mean[shared_scale_mask, :].ravel(), recon_mean[shared_scale_mask, :].ravel()])
    y_min = float(np.min(scale_values))
    y_max = float(np.max(scale_values))
    pad = 0.08 * (y_max - y_min) if y_max > y_min else 1e-7

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    for idx, ch_name in enumerate(ch_names):
        ax.plot(times_ms[plot_mask], real_mean[plot_mask, idx], color=real_colors[idx], linewidth=2.0, label=f"real {ch_name}")
        ax.plot(
            times_ms[plot_mask],
            recon_mean[plot_mask, idx],
            color=recon_colors[idx],
            linewidth=1.9,
            linestyle="--",
            label=f"recon {ch_name}",
        )

    ax.axvline(0, color="#4c4c4c", linestyle=(0, (3, 3)), linewidth=1.0)
    ax.set_xlim(-600, 200)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlabel("Time from response (ms)")
    ax.set_ylabel("Mean amplitude")
    ax.set_title("Real vs reconstructed channel-average waveforms")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    clean_axis(ax)
    add_panel_label(ax, "a")
    ax.text(
        0.99,
        0.04,
        "Shared y-axis scale estimated from the -600 to 0 ms analysis window",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#666666",
        fontsize=8,
    )
    save_figure(fig, "supplementary_shared_scale_waveforms_from_minus600")


def make_hidden_state_figure() -> None:
    class_results = pd.read_csv(OUT_DIR / "hidden_state_classification_decoding.csv")
    reg_results = pd.read_csv(OUT_DIR / "hidden_state_neural_regression_decoding.csv")

    reg_obs = reg_results[
        (reg_results["split"] == "within_subject")
        & (reg_results["control"] == "observed_hidden")
        & (reg_results["target"].isin(REG_TARGET_LABELS))
        & (reg_results["window"].isin(WINDOW_ORDER))
    ].copy()
    reg_obs["target_label"] = reg_obs["target"].map(REG_TARGET_LABELS)
    reg_obs["window_label"] = pd.Categorical(reg_obs["window"], WINDOW_ORDER, ordered=True)
    reg_heat = (
        reg_obs.pivot(index="target_label", columns="window_label", values="r2")
        .reindex(index=[REG_TARGET_LABELS[k] for k in REG_TARGET_LABELS], columns=WINDOW_ORDER)
    )
    reg_heat.columns = [WINDOW_LABELS[c] for c in reg_heat.columns]

    class_within = class_results[(class_results["split"] == "within_subject") & (class_results["window"].isin(WINDOW_ORDER))].copy()
    class_obs = class_within[class_within["control"] == "observed_hidden"].copy()
    class_ctrl = (
        class_within[class_within["control"] != "observed_hidden"]
        .groupby(["target", "window"], as_index=False)["balanced_accuracy"]
        .max()
        .rename(columns={"balanced_accuracy": "control_max"})
    )
    class_obs = class_obs.merge(class_ctrl, on=["target", "window"], how="left")
    class_obs["margin"] = class_obs["balanced_accuracy"] - class_obs["control_max"]
    class_obs = class_obs[class_obs["target"].isin(TARGET_LABELS)].copy()
    class_obs["target_label"] = class_obs["target"].map(TARGET_LABELS)
    class_obs["window_label"] = pd.Categorical(class_obs["window"], WINDOW_ORDER, ordered=True)
    target_order = ["Choice", "Condition", "Correctness", "Difficulty", "RT bin", "Arrangement"]
    class_heat = (
        class_obs.pivot(index="target_label", columns="window_label", values="margin")
        .reindex(index=target_order, columns=WINDOW_ORDER)
    )
    class_heat.columns = [WINDOW_LABELS[c] for c in class_heat.columns]
    class_val_text = (
        class_obs.pivot(index="target_label", columns="window_label", values="balanced_accuracy")
        .reindex(index=target_order, columns=WINDOW_ORDER)
    )

    fig = plt.figure(figsize=(11.0, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.82, 1.18], wspace=0.32)
    ax_reg = fig.add_subplot(gs[0, 0])
    ax_cls = fig.add_subplot(gs[0, 1])

    reg_vals = reg_heat.to_numpy(dtype=float)
    im_reg = ax_reg.imshow(reg_vals, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=max(0.85, np.nanmax(reg_vals)))
    for i in range(reg_vals.shape[0]):
        for j in range(reg_vals.shape[1]):
            if np.isfinite(reg_vals[i, j]):
                ax_reg.text(j, i, f"{reg_vals[i, j]:.2f}", ha="center", va="center", color="#16324f", fontsize=8)
    ax_reg.set_xticks(range(reg_heat.shape[1]), reg_heat.columns, rotation=25, ha="right")
    ax_reg.set_yticks(range(reg_heat.shape[0]), reg_heat.index)
    ax_reg.set_title("Hidden states predicting empirical CPP features")
    clean_axis(ax_reg)
    add_panel_label(ax_reg, "a")
    cbar_reg = fig.colorbar(im_reg, ax=ax_reg, fraction=0.048, pad=0.03)
    cbar_reg.set_label("$R^2$")

    cls_vals = class_heat.to_numpy(dtype=float)
    vmax = max(0.14, np.nanmax(np.abs(cls_vals)))
    im_cls = ax_cls.imshow(cls_vals, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for i in range(cls_vals.shape[0]):
        for j in range(cls_vals.shape[1]):
            margin = cls_vals[i, j]
            observed = class_val_text.iloc[i, j]
            if np.isfinite(margin):
                text_color = "white" if abs(margin) > vmax * 0.45 else "#222222"
                ax_cls.text(j, i, f"{margin:+.02f}\n{observed:.2f}", ha="center", va="center", color=text_color, fontsize=7)
    ax_cls.set_xticks(range(class_heat.shape[1]), class_heat.columns, rotation=25, ha="right")
    ax_cls.set_yticks(range(class_heat.shape[0]), class_heat.index)
    ax_cls.set_title("Hidden-state task coding")
    clean_axis(ax_cls)
    add_panel_label(ax_cls, "b")
    cbar_cls = fig.colorbar(im_cls, ax=ax_cls, fraction=0.048, pad=0.03)
    cbar_cls.set_label("Balanced accuracy above best control")
    ax_cls.text(
        1.01,
        -0.14,
        "Cell text: margin over control\nand observed balanced accuracy",
        transform=ax_cls.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#666666",
    )

    fig.suptitle("Main Figure 2. Hidden-state relation to neural features and task variables", x=0.06, y=0.99, ha="left", fontsize=12)
    save_figure(fig, "main_figure_2_hidden_state_relations")


def make_behavior_figure() -> None:
    behavior = pd.read_csv(OUT_DIR / "behavioral_external_validation_rt.csv")
    behavior = behavior.set_index("model").reindex(BEHAVIOR_ORDER).reset_index()
    behavior["label"] = behavior["model"].map(BEHAVIOR_LABELS)
    colors = []
    for model in behavior["model"]:
        if model == "behavior_plus_cpp_plus_hidden":
            colors.append("#1f3a5f")
        elif "hidden" in model and "shuffled" not in model:
            colors.append("#54708a")
        elif "cpp" in model:
            colors.append("#b9824d")
        elif "shuffled" in model:
            colors.append("#bfbfbf")
        else:
            colors.append("#7a7a7a")

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ypos = np.arange(len(behavior))
    ax.barh(ypos, behavior["r2"], xerr=behavior["r2_sd"], color=colors, edgecolor="none", height=0.65, error_kw={"elinewidth": 0.9, "ecolor": "#444444", "capsize": 2})
    ax.axvline(0, color="#4c4c4c", linewidth=0.9)
    ax.set_yticks(ypos, behavior["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Cross-validated $R^2$ for log RT")
    ax.set_title("Supplementary Figure. Behavioral external validation")
    clean_axis(ax)
    add_panel_label(ax, "a")
    ax.text(
        0.01,
        -0.18,
        "Displayed as secondary evidence rather than the main validation criterion.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#666666",
    )
    save_figure(fig, "supplementary_figure_behavioral_external_validation")


def write_plan_and_summary() -> None:
    reg_results = pd.read_csv(OUT_DIR / "hidden_state_neural_regression_decoding.csv")
    behavior = pd.read_csv(OUT_DIR / "behavioral_external_validation_rt.csv")
    hidden_cpp = reg_results[
        (reg_results["split"] == "within_subject")
        & (reg_results["control"] == "observed_hidden")
        & (reg_results["target"] == "cpp_amp_minus600_to_minus50")
        & (reg_results["window"] == "minus600_to_minus50")
    ].iloc[0]
    beh_cpp_hidden = behavior.loc[behavior["model"] == "behavior_plus_cpp_plus_hidden"].iloc[0]
    beh_cpp = behavior.loc[behavior["model"] == "behavior_plus_cpp"].iloc[0]

    plan_text = """# Publication Figure Plan

## Main figures

### Main Figure 1. Neural reconstruction in the pre-response window
- **Panel a:** grand-average empirical vs reconstructed CPP waveform, restricted to the main response-locked analysis window (-600 to 0 ms), with the response onset marked and the response-proximal region lightly shaded.
- **Panel b:** condition-wise empirical and model waveforms in a compact paired layout.
- **Panel c:** correctness-wise empirical and model waveforms in the same layout.
- **Panel d:** difficulty-wise empirical and model waveforms in the same layout.
- **Rationale:** this figure puts the strongest and most interpretable evidence first: the model captures the overall response-locked CPP-like shape in the scientifically relevant window while keeping the unstable early edge region out of the main emphasis.

### Main Figure 2. Hidden-state relation to neural features and task variables
- **Panel a:** heatmap of hidden-state prediction quality for empirical CPP amplitude and slope across the four pre-defined time windows.
- **Panel b:** heatmap of hidden-state task coding, expressed as balanced-accuracy gain over the best shuffled or majority control.
- **Rationale:** this figure centers the question of whether hidden states carry meaningful neural information and treats task decoding as supporting evidence rather than the primary claim.

## Supplementary figures

### Supplementary Figure. Behavioral external validation
- Compact model-comparison plot for log RT prediction across behavior-only, CPP-only, hidden-only, combined, and shuffled-hidden baselines.
- **Rationale:** this is useful external validation, but it should remain secondary because it does not directly establish that the latent states are a mechanistic behavioral model.

## Omitted items

- Full-window waveform plots are not used as main panels because the unstable edge region near -1000 ms is visually distracting and not part of the main interpretation.
- CPP AUC is not included because it is not present in the current saved validation tables.
"""
    PLAN_PATH.write_text(plan_text, encoding="utf-8")

    summary_text = f"""# Publication Figure Summary Note

The redesigned figures support the cautious conclusion that the current model is good enough for exploratory hidden-to-CPP analysis, especially because hidden states predict empirical CPP amplitude well in the main -600 to -50 ms window ($R^2$ = {hidden_cpp['r2']:.2f}).

At the same time, the behavior figure should still be framed as secondary evidence. Even though the combined behavioral model with CPP and hidden states reaches a higher log-RT prediction score ($R^2$ = {beh_cpp_hidden['r2']:.2f}) than behavior plus CPP alone ($R^2$ = {beh_cpp['r2']:.2f}), the overall story is still stronger on neural reconstruction and hidden-to-CPP mapping than on a strong mechanistic behavioral claim.
"""
    SUMMARY_PATH.write_text(summary_text, encoding="utf-8")


def main() -> None:
    set_style()
    ensure_dirs()
    make_waveform_figure()
    make_shared_scale_windowed_waveform_figure()
    make_hidden_state_figure()
    make_behavior_figure()
    write_plan_and_summary()
    print(f"Publication-style figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
