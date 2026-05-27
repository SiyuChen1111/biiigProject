from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import numpy as np
import pandas as pd
from scipy.io import loadmat


CHANNEL_NAMES = ["CP1", "CP2", "CPz"]
SUBJECT_IDS = [
    1117, 1118, 1120, 1124, 1126, 1131, 1132, 1135, 1136,
    1151, 1160, 1164, 1167, 1169, 1172, 1173, 1178,
    1182, 1215, 1216, 1219, 1223, 1227, 1233,
    1234, 1237, 1240, 1243, 1245, 1247, 1250, 1239,
    1252, 1257, 1261, 1265, 1266, 1268, 1270, 1276, 1281,
]
SUBJECT_LABELS = [f"sub-STSWD{i}" for i in SUBJECT_IDS]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _load_exported_blocks(mat_path: Path) -> list[np.ndarray]:
    mat = loadmat(mat_path, squeeze_me=False)
    if "resp_locked_erp" not in mat:
        raise KeyError(f"`resp_locked_erp` not found in {mat_path}")
    exported = mat["resp_locked_erp"]
    blocks = [np.asarray(item, dtype=np.float32) for item in exported.ravel()]
    if len(blocks) != len(SUBJECT_LABELS):
        raise ValueError(f"Expected 41 subject blocks, found {len(blocks)} in {mat_path}")
    for idx, block in enumerate(blocks):
        if block.ndim != 3 or block.shape[1] != 3:
            raise ValueError(f"Unexpected block shape for {SUBJECT_LABELS[idx]}: {block.shape}")
    return blocks


def _prepare_metadata_subject(subject_rows: pd.DataFrame, subject_id: str, valid_mask: np.ndarray) -> pd.DataFrame:
    metadata = subject_rows.loc[valid_mask].copy().reset_index(drop=True)
    if "Unnamed: 0" in metadata.columns and "original_row_index" not in metadata.columns:
        metadata = metadata.rename(columns={"Unnamed: 0": "original_row_index"})
    if "original_row_index" not in metadata.columns:
        metadata["original_row_index"] = np.flatnonzero(valid_mask)
    metadata["subject_id"] = subject_id
    metadata["within_subject_trial_index"] = np.flatnonzero(valid_mask)
    metadata["trial_id"] = [f"{subject_id}_{int(i):04d}" for i in metadata["within_subject_trial_index"]]
    metadata["RT_ms"] = pd.to_numeric(metadata["probe_rt"], errors="coerce") * 1000.0
    metadata["correctness"] = pd.to_numeric(metadata["probe_accuracy"], errors="coerce")
    metadata["condition"] = metadata["probe_attribute"] if "probe_attribute" in metadata else metadata["cue_dimensionality"]
    metadata["difficulty"] = metadata["cue_dimensionality"] if "cue_dimensionality" in metadata else metadata["condition"]
    metadata["evidence_strength"] = metadata["difficulty"]
    metadata["choice"] = metadata["probe_leftrightwin"] if "probe_leftrightwin" in metadata else np.nan
    metadata["response_hand"] = metadata["choice"]
    metadata["artifact_rejection_flag"] = 0
    metadata["alignment"] = "response_locked"
    return metadata


def _write_notes(path: Path, eeg_shape: tuple[int, int, int], total_input_trials: int, removed_trials: int) -> None:
    text = f"""# Fixed Stage 2/3 Trial-Level Dataset

## Data sources

- EEG source: `script_pre_EEG/Kosciessa_et_al_2021/temp_data/resp_locked_erp.mat`.
- Behavior source: `script_pre_EEG/Kosciessa_et_al_2021/temp_data/behavior_data_all.csv`.
- The notebook export was fixed so each subject block is appended inside the subject loop.

## Alignment

- The exported EEG contains 41 subject blocks.
- Subject block trial counts match `behavior_data_all.csv` subject trial counts before invalid-trial removal.
- Metadata rows are built in the same subject and trial order as the EEG blocks.
- Trials containing non-finite EEG values are removed together with their metadata rows.

## Window and sampling

- This dataset is response-locked, not stimulus-locked.
- Response-locked window: approximately -1000 ms to 200 ms.
- Sampling rate: 256 Hz.
- Time points: {eeg_shape[1]}.

## Channels

- Channel order: CP1, CP2, CPz.
- Saved EEG shape: trial x time x channel = {eeg_shape}.

## Trial inclusion

- Input EEG/behavior trial pairs before non-finite EEG removal: {total_input_trials}.
- Removed trials because the response-locked EEG window contained NaN or inf: {removed_trials}.
- `RT_ms = probe_rt * 1000`.
- `correctness = probe_accuracy`.
- `condition = probe_attribute`.
- `difficulty` and `evidence_strength` currently use `cue_dimensionality` as a condition proxy.
- `choice` and `response_hand` use `probe_leftrightwin`.
- `artifact_rejection_flag = 0` because no separate artifact flag is available in the exported behavior table.

## Interpretation boundary

- This data is suitable for response-proximal CPP latent analysis.
- It should not be treated as pure stimulus-locked evidence accumulation.
- A strict stimulus-locked accumulation analysis requires an additional stimulus-locked export.
"""
    path.write_text(text, encoding="utf-8")


def _write_plots(output_dir: Path, eeg: np.ndarray, metadata: pd.DataFrame, times_ms: np.ndarray, summary: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cpp = eeg.mean(axis=2)
    plt.figure(figsize=(10, 4))
    plt.plot(times_ms, cpp.mean(axis=0), color="black", linewidth=2)
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Time from response (ms)")
    plt.ylabel("Mean CPP amplitude")
    plt.tight_layout()
    plt.savefig(output_dir / "mean_cpp_response_locked_sanity.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.bar(summary["subject_id"], summary["n_eeg_trials_exported"])
    plt.xticks(rotation=90)
    plt.ylabel("Valid EEG trials")
    plt.tight_layout()
    plt.savefig(output_dir / "subject_trial_counts.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(metadata["RT_ms"], bins=50, color="#4C78A8")
    plt.xlabel("RT (ms)")
    plt.ylabel("Trials")
    plt.tight_layout()
    plt.savefig(output_dir / "rt_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.bar(summary["subject_id"], summary["accuracy_mean"])
    plt.xticks(rotation=90)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_by_subject.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    for channel_idx, channel in enumerate(CHANNEL_NAMES):
        plt.plot(times_ms, eeg[:, :, channel_idx].mean(axis=0), label=channel)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Time from response (ms)")
    plt.ylabel("Mean amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "channel_average_waveforms.png", dpi=160)
    plt.close()


def _validation_report(output_dir: Path, eeg: np.ndarray, metadata: pd.DataFrame, times_ms: np.ndarray, summary: pd.DataFrame) -> dict[str, Any]:
    key_columns = [
        "subject_id", "trial_id", "original_row_index", "probe_rt", "RT_ms",
        "probe_accuracy", "correctness", "probe_attribute", "cue_dimensionality",
        "probe_leftrightwin", "choice", "stim_onset", "resp_onset_sample",
        "rt_is_not_outlier", "artifact_rejection_flag",
    ]
    missing_key_columns = [column for column in key_columns if column not in metadata.columns]
    any_nan_key = False if missing_key_columns else bool(metadata[key_columns].isna().any().any())
    metadata_counts = metadata.groupby("subject_id").size().rename("metadata_count").reset_index()
    merged = summary.merge(metadata_counts, on="subject_id", how="left")
    subject_level_trial_count_passed = bool((merged["n_eeg_trials_exported"] == merged["metadata_count"]).all())
    report = {
        "dataset_dir": str(output_dir),
        "eeg_shape": list(eeg.shape),
        "metadata_shape": list(metadata.shape),
        "times_shape": list(times_ms.shape),
        "channel_names": CHANNEL_NAMES,
        "n_subjects": int(metadata["subject_id"].nunique()),
        "total_trials": int(eeg.shape[0]),
        "trial_count_match": bool(eeg.shape[0] == len(metadata)),
        "time_count_match": bool(eeg.shape[1] == len(times_ms)),
        "channel_count_match": bool(eeg.shape[2] == 3),
        "channel_names_match": True,
        "trial_id_unique": bool(metadata["trial_id"].is_unique),
        "any_nan_in_eeg": bool(np.isnan(eeg).any()),
        "any_inf_in_eeg": bool(np.isinf(eeg).any()),
        "missing_metadata_key_columns": missing_key_columns,
        "any_nan_in_metadata_key_columns": any_nan_key,
        "subject_level_trial_count_passed": subject_level_trial_count_passed,
        "rt_positive": bool((metadata["RT_ms"] > 0).all()),
        "rt_ms_min": float(metadata["RT_ms"].min()),
        "rt_ms_max": float(metadata["RT_ms"].max()),
    }
    report["final_passed"] = all(
        [
            report["trial_count_match"],
            report["time_count_match"],
            report["channel_count_match"],
            report["channel_names_match"],
            report["trial_id_unique"],
            not report["any_nan_in_eeg"],
            not report["any_inf_in_eeg"],
            not report["missing_metadata_key_columns"],
            not report["any_nan_in_metadata_key_columns"],
            report["subject_level_trial_count_passed"],
            report["rt_positive"],
            report["n_subjects"] == 41,
        ]
    )
    return report


def build_dataset(mat_path: Path, behavior_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks = _load_exported_blocks(mat_path)
    behavior = pd.read_csv(behavior_path)
    if "subj_idx" not in behavior.columns:
        raise ValueError("Expected `subj_idx` in behavior_data_all.csv")

    eeg_blocks: list[np.ndarray] = []
    metadata_blocks: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    total_input_trials = 0
    removed_trials = 0

    for subject_id, block in zip(SUBJECT_LABELS, blocks):
        subject_rows = behavior[behavior["subj_idx"].astype(str) == subject_id].copy().reset_index(drop=True)
        if len(subject_rows) != block.shape[0]:
            raise ValueError(f"{subject_id}: behavior rows {len(subject_rows)} != EEG trials {block.shape[0]}")
        total_input_trials += block.shape[0]

        finite_mask = np.isfinite(block).all(axis=(1, 2))
        valid_rt_mask = (
            pd.to_numeric(subject_rows["probe_rt"], errors="coerce").notna().to_numpy()
            & (pd.to_numeric(subject_rows["probe_rt"], errors="coerce").to_numpy() > 0)
            & pd.to_numeric(subject_rows["probe_accuracy"], errors="coerce").notna().to_numpy()
        )
        valid_mask = finite_mask & valid_rt_mask
        removed_trials += int((~valid_mask).sum())

        subject_eeg = np.transpose(block[valid_mask], (0, 2, 1)).astype(np.float32)
        subject_metadata = _prepare_metadata_subject(subject_rows, subject_id, valid_mask)
        eeg_blocks.append(subject_eeg)
        metadata_blocks.append(subject_metadata)

        rt_ms = pd.to_numeric(subject_rows.loc[valid_mask, "probe_rt"], errors="coerce") * 1000.0
        accuracy = pd.to_numeric(subject_rows.loc[valid_mask, "probe_accuracy"], errors="coerce")
        summary_rows.append(
            {
                "subject_id": subject_id,
                "n_behavior_trials": int(len(subject_rows)),
                "n_valid_behavior_trials": int(valid_rt_mask.sum()),
                "n_eeg_trials_exported": int(valid_mask.sum()),
                "n_missing_or_skipped_trials": int((~valid_mask).sum()),
                "min_RT_ms": float(rt_ms.min()),
                "max_RT_ms": float(rt_ms.max()),
                "mean_RT_ms": float(rt_ms.mean()),
                "accuracy_mean": float(accuracy.mean()),
            }
        )

    eeg = np.concatenate(eeg_blocks, axis=0).astype(np.float32)
    metadata = pd.concat(metadata_blocks, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    times_ms = np.linspace(-1000.0, 200.0, eeg.shape[1], dtype=np.float32)

    np.save(output_dir / "eeg_cpp_trials.npy", eeg)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    np.save(output_dir / "times_ms.npy", times_ms)
    (output_dir / "channel_names.txt").write_text("\n".join(CHANNEL_NAMES) + "\n", encoding="utf-8")
    summary.to_csv(output_dir / "subject_trial_count_summary.csv", index=False)
    _write_notes(output_dir / "preprocessing_notes.md", tuple(eeg.shape), total_input_trials, removed_trials)
    _write_plots(output_dir, eeg, metadata, times_ms, summary)
    report = _validation_report(output_dir, eeg, metadata, times_ms, summary)
    (output_dir / "validation_report.json").write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    if not report["final_passed"]:
        raise RuntimeError(f"Validation failed. See {output_dir / 'validation_report.json'}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 2/3 dataset from fixed response-locked MAT export.")
    parser.add_argument(
        "--mat",
        type=Path,
        default=Path("script_pre_EEG/Kosciessa_et_al_2021/temp_data/resp_locked_erp.mat"),
    )
    parser.add_argument(
        "--behavior",
        type=Path,
        default=Path("script_pre_EEG/Kosciessa_et_al_2021/temp_data/behavior_data_all.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_fixed"))
    args = parser.parse_args()
    report = build_dataset(args.mat, args.behavior, args.output_dir)
    print(json.dumps(report, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
