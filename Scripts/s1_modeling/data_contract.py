from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import DataContractConfig
from .utils import ensure_dir, write_json


def _resolve_required_columns(metadata: pd.DataFrame, config: DataContractConfig) -> Tuple[pd.DataFrame, List[str]]:
    renamed = metadata.copy()
    missing: List[str] = []
    for required in config.required_metadata_columns:
        if required in renamed.columns:
            continue
        aliases = config.optional_aliases.get(required, ())
        alias_match = next((alias for alias in aliases if alias in renamed.columns), None)
        if alias_match is None:
            if required == "alignment":
                renamed[required] = "response_locked"
                continue
            missing.append(required)
            continue
        renamed = renamed.rename(columns={alias_match: required})
    return renamed, missing


def _read_channel_names(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle.readlines() if line.strip()]


def _extract_sampling_rate(times_ms: np.ndarray) -> float:
    if len(times_ms) < 2:
        return float("nan")
    step_ms = float(np.mean(np.diff(times_ms)))
    if step_ms == 0:
        return float("nan")
    return 1000.0 / step_ms


def validate_stage2_dataset(dataset_dir: Path, output_dir: Path, config: DataContractConfig | None = None) -> Dict[str, object]:
    config = config or DataContractConfig()
    output_dir = ensure_dir(output_dir)

    file_checks = {name: (dataset_dir / name).exists() for name in config.expected_files}
    missing_files = [name for name, exists in file_checks.items() if not exists]

    report: Dict[str, object] = {
        "dataset_dir": str(dataset_dir),
        "contract": asdict(config),
        "file_checks": file_checks,
        "missing_files": missing_files,
        "passed": False,
    }

    if missing_files:
        write_json(output_dir / "stage1_blocking_issue_report.json", report)
        return report

    eeg = np.load(dataset_dir / "eeg_cpp_trials.npy")
    times_ms = np.load(dataset_dir / "times_ms.npy")
    metadata = pd.read_csv(dataset_dir / "metadata.csv")
    metadata, missing_columns = _resolve_required_columns(metadata, config)
    channels = _read_channel_names(dataset_dir / "channel_names.txt")
    notes_text = (dataset_dir / "preprocessing_notes.md").read_text(encoding="utf-8")

    n_trials, n_timepoints, n_channels = eeg.shape
    metadata_rows_match = len(metadata) == n_trials
    times_match = len(times_ms) == n_timepoints
    channel_order_matches = tuple(channels) == config.expected_channel_order
    sampling_rate_hz = _extract_sampling_rate(times_ms)

    required_columns_present = not missing_columns
    viable_mask = np.ones(len(metadata), dtype=bool)
    per_subject_summary: List[Dict[str, object]] = []

    report.update(
        {
            "shape_summary": {
                "n_trials": int(n_trials),
                "n_timepoints": int(n_timepoints),
                "n_channels": int(n_channels),
            },
            "metadata_rows_match": metadata_rows_match,
            "times_match": times_match,
            "channel_order_matches": channel_order_matches,
            "missing_metadata_columns": missing_columns,
            "sampling_rate_hz": sampling_rate_hz,
            "required_columns_present": required_columns_present,
            "per_subject_retained_trials": per_subject_summary,
            "preprocessing_policy_extract": {
                "reference_mentioned": "reference" in notes_text.lower(),
                "filter_mentioned": "filter" in notes_text.lower(),
                "artifact_mentioned": "artifact" in notes_text.lower() or "ica" in notes_text.lower(),
                "baseline_mentioned": "baseline" in notes_text.lower(),
            },
        }
    )

    report["passed"] = all(
        [
            metadata_rows_match,
            times_match,
            channel_order_matches,
            not missing_columns,
        ]
    )

    filename = "stage1_data_contract_report.json" if report["passed"] else "stage1_blocking_issue_report.json"
    write_json(output_dir / filename, report)
    if per_subject_summary:
        pd.DataFrame(per_subject_summary).to_csv(output_dir / "stage1_subject_trial_summary.csv", index=False)
    return report
