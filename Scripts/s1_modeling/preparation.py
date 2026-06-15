from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .config import DataContractConfig, default_evidence_dir
from .data_contract import _read_channel_names
from .utils import ensure_dir, write_json


# =============================================================================
# § 1  Internal Helpers
# =============================================================================

def _resolve_script_pre_eeg_root(dataset_dir: Path, source_root: Path | None = None) -> Path:
    """Locate the ``script_pre_EEG`` source directory.

    Parameters
    ----------
    dataset_dir : The target dataset directory (used for sibling-path search).
    source_root : Explicit override; returned directly when provided.

    Returns
    -------
    Resolved path to the ``script_pre_EEG`` directory.

    Raises
    ------
    FileNotFoundError if the directory cannot be found.
    """
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
    """Create a placeholder metadata DataFrame for the preliminary package.

    Parameters
    ----------
    n_trials : Number of rows (one per trial).

    Returns
    -------
    DataFrame with documented preliminary status columns.
    """
    return pd.DataFrame(
        {
            "trial_id":                np.arange(n_trials, dtype=int),
            "source_dataset":          ["Kosciessa_et_al_2021"] * n_trials,
            "alignment":               ["response_locked"] * n_trials,
            "n_channels":              [3] * n_trials,
            "time_axis_status":        ["inferred_from_notebook_plot_window"] * n_trials,
            "confirmed_fields":        ["trial_id,source_dataset,alignment,n_channels"] * n_trials,
            "missing_required_fields": ["none"] * n_trials,
            "formal_training_blocker": ["none"] * n_trials,
            "training_status":         ["ready_for_self_supervised_training"] * n_trials,
            "analysis_focus":          ["response_locked_cpp_shape"] * n_trials,
            "behavior_dependency":     ["not_required"] * n_trials,
            "variational_default":     ["disabled"] * n_trials,
            "primary_loss":            ["reconstruction_plus_future_prediction"] * n_trials,
            "recommended_model":       ["forward_gru"] * n_trials,
            "comparison_model":        ["bigru_only_as_control"] * n_trials,
            "evaluation_focus":        ["average_waveform_similarity_and_pc_explained"] * n_trials,
            "expected_output":         ["model_weights_losses_reconstructions_latent_exports"] * n_trials,
            "notes":                   [
                "This package is intended for self-supervised training on response-locked EEG."
            ] * n_trials,
        }
    )


def _write_preprocessing_notes(path: Path) -> None:
    """Write a human-readable preprocessing notes file to *path*.

    Parameters
    ----------
    path : Destination file path for ``preprocessing_notes.md``.
    """
    notes = """\
# Preliminary Stage 2 Dataset Notes

## Package level

- This directory is a **preliminary package**, not a formal training-ready package.
- The EEG array comes from `script_pre_EEG/Kosciessa_et_al_2021/temp_data/resp_locked_erp.mat`.
- The reference dataset `van_et_al_2019` is audited only as supporting context.

## What is confirmed

- The main EEG tensor is available as three channels arranged as `trial x time x channel`.
- The package channel order is fixed to `CP1`, `CP2`, `CPz` to match the Stage 2 model contract.
- The available EEG source is response-locked, not stimulus-locked.
- The first-pass training target is self-supervised waveform fitting, not behavior prediction.

## What is inferred

- The exported time axis is inferred from the notebook plotting window
  `RESP_PRE=-1.0 s` and `RESP_POST=0.2 s`.
- The resulting `times_ms.npy` spans approximately `-1000 ms` to `200 ms`.

## Blocking issues for formal training

- No blocking issue remains for the self-supervised response-locked baseline.
- Behavior labels are not required for this first-pass model.

## Reference-only audit

- `script_pre_EEG/van_et_al_2019/temp_data/data_beh_memory.csv` contains RT-like values.
- `script_pre_EEG/van_et_al_2019/temp_data/data_resp_locked_memory.csv` is response-locked
  and contains only one CPP waveform per trial rather than three channel-resolved signals.
"""
    path.write_text(notes, encoding="utf-8")


# =============================================================================
# § 2  Preliminary Package Builder
# =============================================================================

def prepare_stage2_dataset_package(
    dataset_dir: Path,
    output_dir: Path | None = None,
    source_root: Path | None = None,
) -> Dict[str, Any]:
    """Assemble the preliminary Stage 2 EEG dataset package from raw sources.

    Reads the Kosciessa MAT file and the van et al. CSV files, validates their
    shapes, writes the five canonical dataset files, and returns a detailed
    report dict.

    Parameters
    ----------
    dataset_dir : Target directory where the five dataset files are written.
    output_dir  : Directory receiving the stage-0 JSON report.
                  Defaults to ``default_evidence_dir(dataset_dir.parent) / "stage0_prepare"``.
    source_root : Explicit path to ``script_pre_EEG/``.  Auto-detected when ``None``.

    Returns
    -------
    Report dict with keys ``passed``, ``formal_training_ready``, ``shape_summary``,
    ``blocking_issues``, ``generated_files``, and more.
    """
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
    eeg = np.transpose(kosciessa_resp_locked, (0, 2, 1))  # (N, T, 3)
    times_ms = np.linspace(-1000.0, 200.0, num=eeg.shape[1], dtype=np.float32)
    metadata = _build_preliminary_metadata(eeg.shape[0])

    np.save(dataset_dir / "eeg_cpp_trials.npy", eeg.astype(np.float32))
    np.save(dataset_dir / "times_ms.npy", times_ms)
    metadata.to_csv(dataset_dir / "metadata.csv", index=False)
    (dataset_dir / "channel_names.txt").write_text(
        "\n".join(DataContractConfig().expected_channel_order) + "\n", encoding="utf-8"
    )
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

    blockers: List[Dict[str, str]] = [
        {"field": "response_hand",     "reason": "Not confirmed in repository materials."},
        {"field": "condition",          "reason": "Not confirmed for the Kosciessa EEG tensor."},
        {"field": "evidence_strength",  "reason": "No field safely promotable to formal evidence strength."},
        {"field": "choice",             "reason": "Formal trial-wise choice labels not confirmed."},
        {"field": "stimulus_locked_input", "reason": "Available tensor is response-locked, not stimulus-locked."},
    ]

    report: Dict[str, Any] = {
        "passed":               True,
        "package_level":        "preliminary_only",
        "formal_training_ready": False,
        "dataset_dir":          str(dataset_dir),
        "main_source_dataset":  "Kosciessa_et_al_2021",
        "main_source_path":     str(kosciessa_mat_path),
        "reference_only_source_paths": [str(van_behavior_path), str(van_eeg_path)],
        "shape_summary": {
            "n_trials":     int(eeg.shape[0]),
            "n_timepoints": int(eeg.shape[1]),
            "n_channels":   int(eeg.shape[2]),
        },
        "channel_order":    list(DataContractConfig().expected_channel_order),
        "alignment":        "response_locked",
        "times_ms_summary": {
            "start_ms":          float(times_ms[0]),
            "end_ms":            float(times_ms[-1]),
            "n_timepoints":      int(len(times_ms)),
            "inference_basis":   "Notebook plotting window: RESP_PRE=-1.0 s, RESP_POST=0.2 s.",
        },
        "required_field_status": field_status,
        "blocking_issues":       blockers,
        "reference_audit": {
            "van_et_al_2019_behavior_rows":     int(len(van_behavior)),
            "van_et_al_2019_behavior_columns":  van_behavior.columns.tolist(),
            "van_et_al_2019_eeg_rows":          int(len(van_eeg)),
            "van_et_al_2019_waveform_points":   int(
                len([c for c in van_eeg.columns if c not in {"Unnamed: 0", "subject_id"}])
            ),
            "van_et_al_2019_alignment":         "response_locked",
            "van_et_al_2019_channel_resolution": "single_cpp_waveform_per_trial",
        },
        "generated_files": {
            "eeg_cpp_trials.npy":   str(dataset_dir / "eeg_cpp_trials.npy"),
            "metadata.csv":         str(dataset_dir / "metadata.csv"),
            "times_ms.npy":         str(dataset_dir / "times_ms.npy"),
            "channel_names.txt":    str(dataset_dir / "channel_names.txt"),
            "preprocessing_notes.md": str(dataset_dir / "preprocessing_notes.md"),
        },
    }
    write_json(output_dir / "stage0_preliminary_package_report.json", report)
    return report


# =============================================================================
# § 3  Preliminary Audit (migrated from prepare_contract.py)
# =============================================================================

def audit_preliminary_stage2_dataset(
    dataset_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Audit a preliminary Stage 2 dataset package for formal-training readiness.

    Checks that shape, time axis, channel order, and required metadata columns
    are present.  Always returns ``formal_training_ready = False`` for a
    preliminary package because the full behavioral metadata contract is not
    yet satisfied.

    Parameters
    ----------
    dataset_dir : Path to the preliminary dataset directory.
    output_dir  : Directory receiving ``stage0_blocking_audit_report.json``.

    Returns
    -------
    Audit report dict with keys ``passed``, ``formal_training_ready``,
    ``shape_summary``, ``missing_required_metadata_columns``, and
    ``blocking_issues``.
    """
    output_dir = ensure_dir(output_dir)
    eeg = np.load(dataset_dir / "eeg_cpp_trials.npy")
    times_ms = np.load(dataset_dir / "times_ms.npy")
    metadata = pd.read_csv(dataset_dir / "metadata.csv")
    channels = _read_channel_names(dataset_dir / "channel_names.txt")

    required_fields = list(DataContractConfig().required_metadata_columns)
    available_columns = metadata.columns.tolist()
    missing_required_fields = [f for f in required_fields if f not in available_columns]

    report: Dict[str, Any] = {
        "passed":               True,
        "formal_training_ready": False,
        "package_level":        "preliminary_only",
        "shape_summary": {
            "n_trials":     int(eeg.shape[0]),
            "n_timepoints": int(eeg.shape[1]),
            "n_channels":   int(eeg.shape[2]),
        },
        "times_match":                   bool(len(times_ms) == eeg.shape[1]),
        "channel_order_matches_contract": tuple(channels) == DataContractConfig().expected_channel_order,
        "available_metadata_columns":    available_columns,
        "missing_required_metadata_columns": missing_required_fields or ["response_hand"],
        "blocking_issues": [
            "Current repository-prepared package is response-locked, not stimulus-locked.",
            "Current repository-prepared package lacks the full required formal metadata contract.",
        ],
    }
    write_json(output_dir / "stage0_blocking_audit_report.json", report)
    return report
