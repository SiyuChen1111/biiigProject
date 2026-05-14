from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .config import DataContractConfig, default_evidence_dir
from .utils import ensure_dir, write_json


def _resolve_script_pre_eeg_root(dataset_dir: Path, source_root: Path | None = None) -> Path:
    if source_root is not None:
        return source_root
    candidates = [
        dataset_dir.parent / "script_pre_EEG",
        dataset_dir / "script_pre_EEG",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate script_pre_EEG relative to {dataset_dir}")


def _build_preliminary_metadata(n_trials: int) -> pd.DataFrame:
    metadata = pd.DataFrame(
        {
            "trial_id": np.arange(n_trials, dtype=int),
            "source_dataset": ["Kosciessa_et_al_2021"] * n_trials,
            "alignment": ["response_locked"] * n_trials,
            "n_channels": [3] * n_trials,
            "time_axis_status": ["inferred_from_notebook_plot_window"] * n_trials,
            "confirmed_fields": ["trial_id,source_dataset,alignment,n_channels"] * n_trials,
            "missing_required_fields": [
                "subject_id,condition,evidence_strength,choice,correctness,RT_ms,response_hand,artifact_rejection_flag"
            ]
            * n_trials,
            "formal_training_blocker": [
                "Missing required trial-level metadata and source data are response-locked rather than stimulus-locked."
            ]
            * n_trials,
        }
    )
    return metadata


def _write_preprocessing_notes(path: Path) -> None:
    notes = """# Preliminary Stage 2 Dataset Notes

## Package level

- This directory is a **preliminary package**, not a formal training-ready package.
- The EEG array comes from `script_pre_EEG/Kosciessa_et_al_2021/temp_data/resp_locked_erp.mat`.
- The reference dataset `van_et_al_2019` is audited only as supporting context and is not used as the main EEG source.

## What is confirmed

- The main EEG tensor is available as three channels and can be arranged to `trial x time x channel`.
- The package channel order is fixed to `CP1`, `CP2`, `CPz` to match the active Stage 2 model contract.
- The available EEG source is response-locked, not stimulus-locked.

## What is inferred

- The exported time axis is inferred from the notebook plotting window `RESP_PRE=-1.0 s` and `RESP_POST=0.2 s`.
- The resulting `times_ms.npy` spans approximately `-1000 ms` to `200 ms`.

## Blocking issues for formal training

- Trial-level `subject_id` is not available in the current repository source files for the chosen EEG tensor.
- Required formal metadata fields are missing: `condition`, `evidence_strength`, `choice`, `correctness`, `RT_ms`, `response_hand`, `artifact_rejection_flag`.
- The current EEG tensor is response-locked, while the formal Stage 2 model contract expects stimulus-locked input.

## Reference-only audit

- `script_pre_EEG/van_et_al_2019/temp_data/data_beh_memory.csv` contains behavioral rows and RT-like values.
- `script_pre_EEG/van_et_al_2019/temp_data/data_resp_locked_memory.csv` is also response-locked and contains only one CPP waveform per trial rather than three channel-resolved signals.
"""
    path.write_text(notes, encoding="utf-8")


def prepare_stage2_dataset_package(
    dataset_dir: Path,
    output_dir: Path | None = None,
    source_root: Path | None = None,
) -> Dict[str, Any]:
    dataset_dir = ensure_dir(dataset_dir)
    script_root = _resolve_script_pre_eeg_root(dataset_dir, source_root)
    output_dir = ensure_dir(output_dir or default_evidence_dir(dataset_dir.parent) / "stage0_prepare")

    kosciessa_mat_path = script_root / "Kosciessa_et_al_2021" / "temp_data" / "resp_locked_erp.mat"
    van_behavior_path = script_root / "van_et_al_2019" / "temp_data" / "data_beh_memory.csv"
    van_eeg_path = script_root / "van_et_al_2019" / "temp_data" / "data_resp_locked_memory.csv"

    kosciessa_mat = loadmat(kosciessa_mat_path)
    kosciessa_resp_locked = np.asarray(kosciessa_mat["resp_locked_erp"], dtype=np.float32)
    if kosciessa_resp_locked.ndim != 3 or kosciessa_resp_locked.shape[1] != 3:
        raise ValueError(f"Unexpected Kosciessa tensor shape: {kosciessa_resp_locked.shape}")
    eeg = np.transpose(kosciessa_resp_locked, (0, 2, 1))
    times_ms = np.linspace(-1000.0, 200.0, num=eeg.shape[1], dtype=np.float32)
    metadata = _build_preliminary_metadata(eeg.shape[0])

    np.save(dataset_dir / "eeg_cpp_trials.npy", eeg.astype(np.float32))
    np.save(dataset_dir / "times_ms.npy", times_ms)
    metadata.to_csv(dataset_dir / "metadata.csv", index=False)
    (dataset_dir / "channel_names.txt").write_text("\n".join(DataContractConfig().expected_channel_order) + "\n", encoding="utf-8")
    _write_preprocessing_notes(dataset_dir / "preprocessing_notes.md")

    van_behavior = pd.read_csv(van_behavior_path)
    van_eeg = pd.read_csv(van_eeg_path)

    required_fields = list(DataContractConfig().required_metadata_columns)
    available_columns = set(metadata.columns)
    field_status = {
        field: {
            "available_in_preliminary_metadata": field in available_columns,
            "status": "confirmed" if field in available_columns else "missing",
        }
        for field in required_fields
    }

    blockers = [
        {
            "field": "response_hand",
            "reason": "Not confirmed anywhere in the repository materials used to build the preliminary package.",
        },
        {
            "field": "condition",
            "reason": "Not confirmed for the chosen Kosciessa EEG tensor in the repository materials.",
        },
        {
            "field": "evidence_strength",
            "reason": "No repository field can be safely promoted to formal evidence strength for the chosen EEG tensor.",
        },
        {
            "field": "choice",
            "reason": "Formal trial-wise choice labels are not confirmed for the chosen EEG tensor.",
        },
        {
            "field": "stimulus_locked_input",
            "reason": "The available Kosciessa tensor in the repository is response-locked rather than stimulus-locked.",
        },
    ]

    report: Dict[str, Any] = {
        "passed": True,
        "package_level": "preliminary_only",
        "formal_training_ready": False,
        "dataset_dir": str(dataset_dir),
        "main_source_dataset": "Kosciessa_et_al_2021",
        "main_source_path": str(kosciessa_mat_path),
        "reference_only_source_paths": [
            str(van_behavior_path),
            str(van_eeg_path),
        ],
        "shape_summary": {
            "n_trials": int(eeg.shape[0]),
            "n_timepoints": int(eeg.shape[1]),
            "n_channels": int(eeg.shape[2]),
        },
        "channel_order": list(DataContractConfig().expected_channel_order),
        "alignment": "response_locked",
        "times_ms_summary": {
            "start_ms": float(times_ms[0]),
            "end_ms": float(times_ms[-1]),
            "n_timepoints": int(len(times_ms)),
            "inference_basis": "Notebook plotting window implies RESP_PRE=-1.0 s and RESP_POST=0.2 s.",
        },
        "required_field_status": field_status,
        "blocking_issues": blockers,
        "reference_audit": {
            "van_et_al_2019_behavior_rows": int(len(van_behavior)),
            "van_et_al_2019_behavior_columns": van_behavior.columns.tolist(),
            "van_et_al_2019_eeg_rows": int(len(van_eeg)),
            "van_et_al_2019_waveform_points": int(
                len([column for column in van_eeg.columns if column not in {"Unnamed: 0", "subject_id"}])
            ),
            "van_et_al_2019_alignment": "response_locked",
            "van_et_al_2019_channel_resolution": "single_cpp_waveform_per_trial",
        },
        "generated_files": {
            "eeg_cpp_trials.npy": str(dataset_dir / "eeg_cpp_trials.npy"),
            "metadata.csv": str(dataset_dir / "metadata.csv"),
            "times_ms.npy": str(dataset_dir / "times_ms.npy"),
            "channel_names.txt": str(dataset_dir / "channel_names.txt"),
            "preprocessing_notes.md": str(dataset_dir / "preprocessing_notes.md"),
        },
    }
    write_json(output_dir / "stage0_preliminary_package_report.json", report)
    return report
