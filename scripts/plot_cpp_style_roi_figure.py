from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

import analyze_roi_conditional_generator as roi_analysis
from src.features.labels_cpp import CPP_ROI_CHANNELS, PAMS_WINDOW


COMPARISON_FIGURE_TITLE = "Figure 1. CPP-style ROI comparison of held-out real and generated waveforms"
REAL_ONLY_FIGURE_TITLE = "Figure 2. Classic held-out real CPP target in the ROI"
DEFAULT_REAL_ONLY_FILENAME_STEM = "real_only_cpp_target"
MICROVOLTS_PER_VOLT = 1_000_000.0
REAL_COLOR = "#111111"
REAL_BAND_COLOR = "#BDBDBD"
GENERATED_COLOR = "#595959"
GENERATED_BAND_COLOR = "#D9D9D9"
MISMATCH_COLOR = "#8C2D04"
MISMATCH_BAND_COLOR = "#FDD0A2"
REAL_NEGATIVE_COLOR = "#6E6E6E"
REAL_NEGATIVE_BAND_COLOR = "#D9D9D9"
REAL_DIFFERENCE_COLOR = "#111111"
REAL_DIFFERENCE_BAND_COLOR = "#CFCFCF"
PAMS_BAND_COLOR = "#EFEFEF"
ZERO_LINE_COLOR = "#4D4D4D"
REFERENCE_LINE_COLOR = "#A6A6A6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a publication-style CPP ROI summary figure from an existing "
            "roi_conditional_generator output root."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="ROI generator contract root containing train/ and analysis/ directories.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Dataset root. If omitted, the script reuses train/train_run_config.json when available.",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Optional subject cap. If omitted, the script reuses train/train_run_config.json when available.",
    )
    parser.add_argument(
        "--filename-stem",
        type=str,
        default="cpp_style_roi_summary",
        help="Filename stem for the saved figure files inside the analysis directory.",
    )
    return parser.parse_args()


def _setup_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Calibri", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.2,
            "lines.linewidth": 2.0,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def _resolve_runtime_args(args: argparse.Namespace, train_output_dir: Path) -> tuple[Path, int | None, int | None]:
    train_config_path = train_output_dir / "train_run_config.json"
    train_config = roi_analysis._load_json(train_config_path) if train_config_path.exists() else {"args": {}}
    config_args = train_config.get("args", {})

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir is not None else Path(config_args.get("dataset_dir", roi_analysis.DATASET_DIR))
    max_subjects = args.max_subjects if args.max_subjects is not None else config_args.get("max_subjects")
    samples_per_class = config_args.get("samples_per_class")
    return dataset_dir, max_subjects, samples_per_class


def _sanitize_filename_stem(filename_stem: str) -> str:
    candidate = Path(filename_stem)
    if filename_stem.strip() == "" or filename_stem in {".", ".."}:
        raise ValueError("--filename-stem must be a non-empty filename stem.")
    if candidate.is_absolute() or candidate.parent != Path("."):
        raise ValueError("--filename-stem must be a plain filename stem without path separators.")
    return candidate.name


def _build_fold_analyses(
    train_output_dir: Path,
    dataset_dir: Path,
    max_subjects: int | None,
    samples_per_class_from_train: int | None,
) -> list[roi_analysis.FoldAnalysis]:
    subject_ids = roi_analysis.filter_subject_ids_with_paths(dataset_dir=dataset_dir)
    if max_subjects is not None:
        subject_ids = subject_ids[:max_subjects]
    if len(subject_ids) < 5:
        raise ValueError("Need at least 5 subjects for grouped 5-fold ROI plotting.")

    subject_blocks = {
        subject_id: roi_analysis._load_subject_dataset(subject_id=subject_id, dataset_dir=dataset_dir)
        for subject_id in subject_ids
    }
    fold_assignments = roi_analysis._build_fold_assignment(subject_ids=subject_ids, n_folds=5)

    fold_analyses: list[roi_analysis.FoldAnalysis] = []
    for fold_idx, test_subjects in enumerate(fold_assignments, start=1):
        sample_path = train_output_dir / f"fold_{fold_idx}" / "conditional_samples.npy"
        if not sample_path.exists():
            raise ValueError(f"Missing generated samples for fold {fold_idx}: {sample_path}")

        remaining_subjects = [subject_id for subject_id in subject_ids if subject_id not in test_subjects]
        train_subjects, _ = roi_analysis._split_train_val_subjects(remaining_subjects)

        train_block = roi_analysis._concatenate_subject_blocks([subject_blocks[subject_id] for subject_id in train_subjects])
        test_block = roi_analysis._concatenate_subject_blocks([subject_blocks[subject_id] for subject_id in test_subjects])
        transform = roi_analysis.fit_cpp_label_transform(train_block.features)
        test_ready = roi_analysis._apply_label_transform(test_block, transform)

        generated_epochs, generated_times, inferred_samples_per_class = roi_analysis._prepare_generated_samples(
            sample_path=sample_path,
            reference_times=test_ready.times,
        )
        samples_per_class = (
            int(samples_per_class_from_train)
            if samples_per_class_from_train is not None
            else inferred_samples_per_class
        )
        if samples_per_class != inferred_samples_per_class:
            raise ValueError(
                f"Fold {fold_idx} sample count mismatch: train config says {samples_per_class}, "
                f"but {sample_path} implies {inferred_samples_per_class} samples per class."
            )

        generated_features = roi_analysis.compute_cpp_features(
            epochs=generated_epochs,
            times=generated_times,
            channel_names=list(CPP_ROI_CHANNELS),
            roi_channels=CPP_ROI_CHANNELS,
        )
        generated_scores = roi_analysis.transform_cpp_scores(generated_features, transform)

        positive = roi_analysis._compare_condition(
            label=1,
            real_epochs=test_ready.epochs,
            real_scores=test_ready.scores,
            real_labels=test_ready.labels,
            generated_epochs=generated_epochs,
            generated_scores=generated_scores,
            samples_per_class=samples_per_class,
            times=test_ready.times,
            generated_times=generated_times,
        )
        negative = roi_analysis._compare_condition(
            label=0,
            real_epochs=test_ready.epochs,
            real_scores=test_ready.scores,
            real_labels=test_ready.labels,
            generated_epochs=generated_epochs,
            generated_scores=generated_scores,
            samples_per_class=samples_per_class,
            times=test_ready.times,
            generated_times=generated_times,
        )

        fold_analyses.append(
            roi_analysis.FoldAnalysis(
                fold=fold_idx,
                test_subject_ids=list(test_subjects),
                times=test_ready.times,
                generated_times=generated_times,
                positive=positive,
                negative=negative,
            )
        )

    if not fold_analyses:
        raise ValueError("No fold analyses were produced.")
    return fold_analyses


def _waveform_stats(waveforms: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.stack(waveforms, axis=0)
    mean = stacked.mean(axis=0)
    sem = stacked.std(axis=0, ddof=0) / np.sqrt(stacked.shape[0])
    return mean, sem


def _to_microvolts(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) * MICROVOLTS_PER_VOLT


def _collect_condition_stats(
    fold_analyses: list[roi_analysis.FoldAnalysis],
) -> dict[str, dict[str, np.ndarray | int]]:
    positive_real_waveforms = [fold.positive.real_waveform for fold in fold_analyses]
    positive_generated_waveforms = [fold.positive.generated_waveform for fold in fold_analyses]
    negative_real_waveforms = [fold.negative.real_waveform for fold in fold_analyses]
    negative_generated_waveforms = [fold.negative.generated_waveform for fold in fold_analyses]

    positive_real_mean, positive_real_sem = _waveform_stats(positive_real_waveforms)
    positive_generated_mean, positive_generated_sem = _waveform_stats(positive_generated_waveforms)
    negative_real_mean, negative_real_sem = _waveform_stats(negative_real_waveforms)
    negative_generated_mean, negative_generated_sem = _waveform_stats(negative_generated_waveforms)
    difference_real_waveforms = [
        fold.positive.real_waveform - fold.negative.real_waveform for fold in fold_analyses
    ]
    difference_generated_waveforms = [
        fold.positive.generated_waveform - fold.negative.generated_waveform for fold in fold_analyses
    ]
    mismatch_waveforms = [
        generated_waveform - real_waveform
        for real_waveform, generated_waveform in zip(difference_real_waveforms, difference_generated_waveforms)
    ]
    difference_real_mean, difference_real_sem = _waveform_stats(difference_real_waveforms)
    difference_generated_mean, difference_generated_sem = _waveform_stats(difference_generated_waveforms)
    mismatch_mean, mismatch_sem = _waveform_stats(mismatch_waveforms)

    return {
        "positive": {
            "times_ms": fold_analyses[0].times * 1000.0,
            "generated_times_ms": fold_analyses[0].generated_times * 1000.0,
            "real_mean": _to_microvolts(positive_real_mean),
            "real_sem": _to_microvolts(positive_real_sem),
            "generated_mean": _to_microvolts(positive_generated_mean),
            "generated_sem": _to_microvolts(positive_generated_sem),
            "real_count": int(sum(fold.positive.real_trial_count for fold in fold_analyses)),
            "generated_count": int(sum(fold.positive.generated_trial_count for fold in fold_analyses)),
        },
        "negative": {
            "times_ms": fold_analyses[0].times * 1000.0,
            "generated_times_ms": fold_analyses[0].generated_times * 1000.0,
            "real_mean": _to_microvolts(negative_real_mean),
            "real_sem": _to_microvolts(negative_real_sem),
            "generated_mean": _to_microvolts(negative_generated_mean),
            "generated_sem": _to_microvolts(negative_generated_sem),
            "real_count": int(sum(fold.negative.real_trial_count for fold in fold_analyses)),
            "generated_count": int(sum(fold.negative.generated_trial_count for fold in fold_analyses)),
        },
        "difference": {
            "times_ms": fold_analyses[0].times * 1000.0,
            "generated_times_ms": fold_analyses[0].generated_times * 1000.0,
            "real_mean": _to_microvolts(difference_real_mean),
            "real_sem": _to_microvolts(difference_real_sem),
            "generated_mean": _to_microvolts(difference_generated_mean),
            "generated_sem": _to_microvolts(difference_generated_sem),
            "real_count": int(sum(fold.positive.real_trial_count + fold.negative.real_trial_count for fold in fold_analyses)),
            "generated_count": int(
                sum(fold.positive.generated_trial_count + fold.negative.generated_trial_count for fold in fold_analyses)
            ),
        },
        "mismatch": {
            "times_ms": fold_analyses[0].times * 1000.0,
            "mismatch_mean": _to_microvolts(mismatch_mean),
            "mismatch_sem": _to_microvolts(mismatch_sem),
        },
    }


def _collect_comparison_y_limits(condition_stats: dict[str, dict[str, np.ndarray | int]]) -> tuple[float, float]:
    values: list[np.ndarray] = []
    for condition_name in ("positive", "negative", "difference"):
        condition = condition_stats[condition_name]
        real_mean = np.asarray(condition["real_mean"], dtype=np.float64)
        real_sem = np.asarray(condition["real_sem"], dtype=np.float64)
        generated_mean = np.asarray(condition["generated_mean"], dtype=np.float64)
        generated_sem = np.asarray(condition["generated_sem"], dtype=np.float64)
        values.extend(
            [
                real_mean - real_sem,
                real_mean + real_sem,
                generated_mean - generated_sem,
                generated_mean + generated_sem,
            ]
        )
    all_values = np.concatenate(values)
    y_min = float(all_values.min())
    y_max = float(all_values.max())
    margin = 0.08 * max(1e-6, y_max - y_min)
    return y_min - margin, y_max + margin


def _collect_mismatch_y_limits(mismatch_stats: dict[str, np.ndarray | int]) -> tuple[float, float]:
    mismatch_mean = np.asarray(mismatch_stats["mismatch_mean"], dtype=np.float64)
    mismatch_sem = np.asarray(mismatch_stats["mismatch_sem"], dtype=np.float64)
    bounds = np.concatenate([mismatch_mean - mismatch_sem, mismatch_mean + mismatch_sem])
    max_abs = float(np.max(np.abs(bounds)))
    limit = max(0.05, max_abs * 1.15)
    return -limit, limit


def _collect_real_overlay_y_limits(condition_stats: dict[str, dict[str, np.ndarray | int]]) -> tuple[float, float]:
    values: list[np.ndarray] = []
    for condition_name in ("positive", "negative"):
        condition = condition_stats[condition_name]
        real_mean = np.asarray(condition["real_mean"], dtype=np.float64)
        real_sem = np.asarray(condition["real_sem"], dtype=np.float64)
        values.extend([real_mean - real_sem, real_mean + real_sem])
    all_values = np.concatenate(values)
    y_min = float(all_values.min())
    y_max = float(all_values.max())
    margin = 0.1 * max(1e-6, y_max - y_min)
    return y_min - margin, y_max + margin


def _collect_real_difference_y_limits(difference_stats: dict[str, np.ndarray | int]) -> tuple[float, float]:
    real_mean = np.asarray(difference_stats["real_mean"], dtype=np.float64)
    real_sem = np.asarray(difference_stats["real_sem"], dtype=np.float64)
    bounds = np.concatenate([real_mean - real_sem, real_mean + real_sem])
    y_min = float(bounds.min())
    y_max = float(bounds.max())
    margin = 0.12 * max(1e-6, y_max - y_min)
    return y_min - margin, y_max + margin


def _condition_title(label: str) -> str:
    return "CPP+ condition" if label == "positive" else "CPP− condition"


def _condition_note(label: str) -> str:
    if label == "positive":
        return "High CPP-like"
    if label == "negative":
        return "Low CPP-like"
    return "Difference wave"


def _plot_condition_panel(
    ax: Axes,
    *,
    label: str,
    stats: dict[str, np.ndarray | int],
    y_limits: tuple[float, float],
    panel_letter: str,
) -> None:
    times_ms = np.asarray(stats["times_ms"], dtype=np.float64)
    generated_times_ms = np.asarray(stats["generated_times_ms"], dtype=np.float64)
    real_mean = np.asarray(stats["real_mean"], dtype=np.float64)
    real_sem = np.asarray(stats["real_sem"], dtype=np.float64)
    generated_mean = np.asarray(stats["generated_mean"], dtype=np.float64)
    generated_sem = np.asarray(stats["generated_sem"], dtype=np.float64)

    ax.axvspan(PAMS_WINDOW[0] * 1000.0, PAMS_WINDOW[1] * 1000.0, color=PAMS_BAND_COLOR, alpha=0.28, zorder=0)
    ax.axhline(0.0, color=REFERENCE_LINE_COLOR, linewidth=0.9, zorder=1)
    ax.axvline(0.0, color=ZERO_LINE_COLOR, linestyle="--", linewidth=1.1, zorder=1)

    ax.fill_between(times_ms, real_mean - real_sem, real_mean + real_sem, color=REAL_BAND_COLOR, alpha=0.18, linewidth=0)
    ax.fill_between(
        generated_times_ms,
        generated_mean - generated_sem,
        generated_mean + generated_sem,
        color=GENERATED_BAND_COLOR,
        alpha=0.22,
        linewidth=0,
    )
    ax.plot(times_ms, real_mean, color=REAL_COLOR, linewidth=2.3, label="Held-out real", zorder=3)
    ax.plot(
        generated_times_ms,
        generated_mean,
        color=GENERATED_COLOR,
        linewidth=2.0,
        linestyle="-",
        label="Generated",
        zorder=4,
    )

    ax.set_title(_condition_title(label), pad=10)
    ax.set_ylim(*y_limits)
    ax.set_xlim(min(times_ms.min(), generated_times_ms.min()), max(times_ms.max(), generated_times_ms.max()))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(direction="in", length=4, width=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        -0.14,
        1.05,
        panel_letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=REAL_COLOR,
    )
    ax.text(
        0.02,
        0.96,
        f"Real n={int(stats['real_count'])}\nGenerated n={int(stats['generated_count'])}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#4D4D4D",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )


def _plot_difference_panel(
    ax: Axes,
    *,
    stats: dict[str, np.ndarray | int],
    y_limits: tuple[float, float],
    panel_letter: str,
) -> None:
    times_ms = np.asarray(stats["times_ms"], dtype=np.float64)
    generated_times_ms = np.asarray(stats["generated_times_ms"], dtype=np.float64)
    real_mean = np.asarray(stats["real_mean"], dtype=np.float64)
    real_sem = np.asarray(stats["real_sem"], dtype=np.float64)
    generated_mean = np.asarray(stats["generated_mean"], dtype=np.float64)
    generated_sem = np.asarray(stats["generated_sem"], dtype=np.float64)

    ax.axvspan(PAMS_WINDOW[0] * 1000.0, PAMS_WINDOW[1] * 1000.0, color=PAMS_BAND_COLOR, alpha=0.28, zorder=0)
    ax.axhline(0.0, color=ZERO_LINE_COLOR, linewidth=1.1, zorder=1)
    ax.axvline(0.0, color=ZERO_LINE_COLOR, linestyle="--", linewidth=1.1, zorder=1)

    ax.fill_between(times_ms, real_mean - real_sem, real_mean + real_sem, color=REAL_BAND_COLOR, alpha=0.18, linewidth=0)
    ax.fill_between(
        generated_times_ms,
        generated_mean - generated_sem,
        generated_mean + generated_sem,
        color=GENERATED_BAND_COLOR,
        alpha=0.22,
        linewidth=0,
    )
    ax.plot(times_ms, real_mean, color=REAL_COLOR, linewidth=2.3, label="Held-out real CPP effect", zorder=3)
    ax.plot(
        generated_times_ms,
        generated_mean,
        color=GENERATED_COLOR,
        linewidth=2.0,
        linestyle="-",
        label="Generated CPP effect",
        zorder=4,
    )

    ax.set_title("CPP difference wave", pad=10)
    ax.set_ylim(*y_limits)
    ax.set_xlim(min(times_ms.min(), generated_times_ms.min()), max(times_ms.max(), generated_times_ms.max()))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(direction="in", length=4, width=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        -0.14,
        1.05,
        panel_letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=REAL_COLOR,
    )
    ax.text(
        0.02,
        0.96,
        "Positive − negative\nreal vs generated",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#4D4D4D",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )


def _plot_mismatch_panel(
    ax: Axes,
    *,
    stats: dict[str, np.ndarray | int],
    y_limits: tuple[float, float],
    panel_letter: str,
) -> None:
    times_ms = np.asarray(stats["times_ms"], dtype=np.float64)
    mismatch_mean = np.asarray(stats["mismatch_mean"], dtype=np.float64)
    mismatch_sem = np.asarray(stats["mismatch_sem"], dtype=np.float64)

    ax.axvspan(PAMS_WINDOW[0] * 1000.0, PAMS_WINDOW[1] * 1000.0, color=PAMS_BAND_COLOR, alpha=0.28, zorder=0)
    ax.axhline(0.0, color=ZERO_LINE_COLOR, linewidth=1.2, zorder=1)
    ax.axvline(0.0, color=ZERO_LINE_COLOR, linestyle="--", linewidth=1.1, zorder=1)

    ax.fill_between(
        times_ms,
        mismatch_mean - mismatch_sem,
        mismatch_mean + mismatch_sem,
        color=MISMATCH_BAND_COLOR,
        alpha=0.42,
        linewidth=0,
    )
    ax.plot(times_ms, mismatch_mean, color=MISMATCH_COLOR, linewidth=2.2, zorder=3)

    ax.set_title("CPP effect mismatch", pad=10)
    ax.set_ylim(*y_limits)
    ax.set_xlim(times_ms.min(), times_ms.max())
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(direction="in", length=4, width=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        -0.14,
        1.05,
        panel_letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=REAL_COLOR,
    )
    ax.text(
        0.02,
        0.96,
        "Generated CPP effect − real CPP effect\nNear zero = better match",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#4D4D4D",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
    )


def _plot_real_overlay_panel(
    ax: Axes,
    *,
    positive_stats: dict[str, np.ndarray | int],
    negative_stats: dict[str, np.ndarray | int],
    y_limits: tuple[float, float],
    panel_letter: str,
) -> None:
    times_ms = np.asarray(positive_stats["times_ms"], dtype=np.float64)
    positive_mean = np.asarray(positive_stats["real_mean"], dtype=np.float64)
    positive_sem = np.asarray(positive_stats["real_sem"], dtype=np.float64)
    negative_mean = np.asarray(negative_stats["real_mean"], dtype=np.float64)
    negative_sem = np.asarray(negative_stats["real_sem"], dtype=np.float64)

    ax.axvspan(PAMS_WINDOW[0] * 1000.0, PAMS_WINDOW[1] * 1000.0, color=PAMS_BAND_COLOR, alpha=0.28, zorder=0)
    ax.axhline(0.0, color=REFERENCE_LINE_COLOR, linewidth=0.9, zorder=1)
    ax.axvline(0.0, color=ZERO_LINE_COLOR, linestyle="--", linewidth=1.1, zorder=1)

    ax.fill_between(times_ms, positive_mean - positive_sem, positive_mean + positive_sem, color=REAL_BAND_COLOR, alpha=0.2, linewidth=0)
    ax.fill_between(times_ms, negative_mean - negative_sem, negative_mean + negative_sem, color=REAL_NEGATIVE_BAND_COLOR, alpha=0.28, linewidth=0)
    ax.plot(times_ms, positive_mean, color=REAL_COLOR, linewidth=2.4, label="High CPP-like", zorder=3)
    ax.plot(times_ms, negative_mean, color=REAL_NEGATIVE_COLOR, linewidth=2.0, linestyle="--", label="Low CPP-like", zorder=4)

    ax.set_title("Held-out real ROI waveforms", pad=10)
    ax.set_ylim(*y_limits)
    ax.set_xlim(times_ms.min(), times_ms.max())
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(direction="in", length=4, width=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        -0.14,
        1.05,
        panel_letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=REAL_COLOR,
    )
    ax.text(
        0.02,
        0.96,
        f"High n={int(positive_stats['real_count'])}\nLow n={int(negative_stats['real_count'])}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#4D4D4D",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
    )


def _plot_real_difference_target_panel(
    ax: Axes,
    *,
    difference_stats: dict[str, np.ndarray | int],
    y_limits: tuple[float, float],
    panel_letter: str,
) -> None:
    times_ms = np.asarray(difference_stats["times_ms"], dtype=np.float64)
    real_mean = np.asarray(difference_stats["real_mean"], dtype=np.float64)
    real_sem = np.asarray(difference_stats["real_sem"], dtype=np.float64)

    ax.axvspan(PAMS_WINDOW[0] * 1000.0, PAMS_WINDOW[1] * 1000.0, color=PAMS_BAND_COLOR, alpha=0.28, zorder=0)
    ax.axhline(0.0, color=ZERO_LINE_COLOR, linewidth=1.1, zorder=1)
    ax.axvline(0.0, color=ZERO_LINE_COLOR, linestyle="--", linewidth=1.1, zorder=1)

    ax.fill_between(times_ms, real_mean - real_sem, real_mean + real_sem, color=REAL_DIFFERENCE_BAND_COLOR, alpha=0.3, linewidth=0)
    ax.plot(times_ms, real_mean, color=REAL_DIFFERENCE_COLOR, linewidth=2.5, label="Held-out real CPP effect", zorder=3)

    ax.set_title("Held-out real CPP effect", pad=10)
    ax.set_ylim(*y_limits)
    ax.set_xlim(times_ms.min(), times_ms.max())
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(direction="in", length=4, width=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        -0.14,
        1.05,
        panel_letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=REAL_COLOR,
    )
    ax.text(
        0.02,
        0.96,
        "High − low condition mean\nheld-out real only",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#4D4D4D",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
    )


def _save_figure_outputs(fig: Figure, output_path_stem: Path) -> list[Path]:
    output_path_stem.parent.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for extension in (".png", ".pdf"):
        save_path = output_path_stem.with_suffix(extension)
        fig.savefig(save_path)
        saved_paths.append(save_path)
    plt.close(fig)
    return saved_paths


def _create_real_only_cpp_target_figure(
    *,
    fold_analyses: list[roi_analysis.FoldAnalysis],
    output_path_stem: Path,
) -> list[Path]:
    _setup_publication_style()
    condition_stats = _collect_condition_stats(fold_analyses)
    overlay_y_limits = _collect_real_overlay_y_limits(condition_stats)
    difference_y_limits = _collect_real_difference_y_limits(condition_stats["difference"])

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))

    _plot_real_overlay_panel(
        axes[0],
        positive_stats=condition_stats["positive"],
        negative_stats=condition_stats["negative"],
        y_limits=overlay_y_limits,
        panel_letter="A",
    )
    _plot_real_difference_target_panel(
        axes[1],
        difference_stats=condition_stats["difference"],
        y_limits=difference_y_limits,
        panel_letter="B",
    )

    axes[0].set_ylabel("Amplitude (µV)")
    axes[1].set_ylabel("Amplitude (µV)")
    axes[0].set_xlabel("Time relative to response (ms)")
    axes[1].set_xlabel("Time relative to response (ms)")

    legend_handles = [
        Line2D([0], [0], color=REAL_COLOR, linewidth=2.4, label="High CPP-like"),
        Line2D([0], [0], color=REAL_NEGATIVE_COLOR, linewidth=2.0, linestyle="--", label="Low CPP-like"),
        Line2D([0], [0], color=REAL_DIFFERENCE_COLOR, linewidth=2.5, label="High − low CPP effect"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.9),
        ncol=3,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.6,
    )
    fig.suptitle(REAL_ONLY_FIGURE_TITLE, y=0.98, fontsize=11.5, fontweight="bold")

    note = (
        "Note. This display figure shows only the held-out real ROI signal that defines the intended CPP target appearance. "
        f"Panel A overlays the high and low CPP-like held-out real waveforms; Panel B shows their response-locked difference wave. "
        f"Shaded bands show ±SEM across fold-level mean waveforms from {len(fold_analyses)} held-out folds. "
        f"The light gray band marks the existing PAMS window used in the ROI label transform for electrodes {', '.join(CPP_ROI_CHANNELS)}. "
        "The dashed vertical line marks time zero at the response."
    )
    fig.text(0.015, 0.02, note, ha="left", va="bottom", fontsize=9, style="italic")
    fig.subplots_adjust(left=0.085, right=0.99, top=0.78, bottom=0.2, wspace=0.22)
    return _save_figure_outputs(fig, output_path_stem)


def _create_cpp_style_figure(
    *,
    fold_analyses: list[roi_analysis.FoldAnalysis],
    output_path_stem: Path,
) -> list[Path]:
    _setup_publication_style()
    condition_stats = _collect_condition_stats(fold_analyses)
    comparison_y_limits = _collect_comparison_y_limits(condition_stats)
    mismatch_y_limits = _collect_mismatch_y_limits(condition_stats["mismatch"])

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.2), sharex=True)

    _plot_condition_panel(
        axes[0, 0],
        label="positive",
        stats=condition_stats["positive"],
        y_limits=comparison_y_limits,
        panel_letter="A",
    )
    _plot_condition_panel(
        axes[0, 1],
        label="negative",
        stats=condition_stats["negative"],
        y_limits=comparison_y_limits,
        panel_letter="B",
    )
    _plot_difference_panel(
        axes[1, 0],
        stats=condition_stats["difference"],
        y_limits=comparison_y_limits,
        panel_letter="C",
    )
    _plot_mismatch_panel(
        axes[1, 1],
        stats=condition_stats["mismatch"],
        y_limits=mismatch_y_limits,
        panel_letter="D",
    )

    axes[0, 0].set_ylabel("Amplitude (µV)")
    axes[1, 0].set_ylabel("Amplitude (µV)")
    axes[1, 1].set_ylabel("Mismatch (µV)")
    axes[1, 0].set_xlabel("Time relative to response (ms)")
    axes[1, 1].set_xlabel("Time relative to response (ms)")

    legend_handles = [
        Line2D([0], [0], color=REAL_COLOR, linewidth=2.4, label="Held-out real"),
        Line2D([0], [0], color=GENERATED_COLOR, linewidth=2.2, label="Generated"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.8,
    )
    fig.suptitle(COMPARISON_FIGURE_TITLE, y=0.975, fontsize=11.5, fontweight="bold")

    note = (
        "Note. Panels A and B compare held-out real and generated ROI waveforms for the high and low CPP-like conditions; "
        "Panel C compares the inferred CPP effect directly as positive minus negative; Panel D isolates the signed mismatch "
        "between generated and real CPP effects, so traces closer to zero indicate better agreement. "
        f"Dark solid lines are held-out real means and medium-gray solid lines are generated means across {len(fold_analyses)} folds; "
        "shaded bands show ±SEM across fold-level mean waveforms. The light gray band marks the existing PAMS window "
        f"used in the ROI label transform for electrodes {', '.join(CPP_ROI_CHANNELS)}. The dashed vertical line marks time zero."
    )
    fig.text(0.015, 0.02, note, ha="left", va="bottom", fontsize=9, style="italic")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.18, wspace=0.16, hspace=0.26)
    return _save_figure_outputs(fig, output_path_stem)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    train_output_dir = output_root / "train"
    analysis_output_dir = output_root / "analysis"
    filename_stem = _sanitize_filename_stem(args.filename_stem)

    dataset_dir, max_subjects, samples_per_class_from_train = _resolve_runtime_args(args, train_output_dir)
    fold_analyses = _build_fold_analyses(
        train_output_dir=train_output_dir,
        dataset_dir=dataset_dir,
        max_subjects=max_subjects,
        samples_per_class_from_train=samples_per_class_from_train,
    )
    comparison_saved_paths = _create_cpp_style_figure(
        fold_analyses=fold_analyses,
        output_path_stem=analysis_output_dir / filename_stem,
    )
    real_only_saved_paths = _create_real_only_cpp_target_figure(
        fold_analyses=fold_analyses,
        output_path_stem=analysis_output_dir / DEFAULT_REAL_ONLY_FILENAME_STEM,
    )

    for saved_path in comparison_saved_paths:
        print(f"Saved comparison ROI figure to {saved_path}")
    for saved_path in real_only_saved_paths:
        print(f"Saved real-only CPP target figure to {saved_path}")


if __name__ == "__main__":
    main()
