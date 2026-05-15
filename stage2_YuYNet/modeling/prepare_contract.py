from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .config import DataContractConfig
from .data_contract import _read_channel_names
from .utils import write_json


def audit_preliminary_stage2_dataset(dataset_dir: Path, output_dir: Path) -> Dict[str, Any]:
    eeg = np.load(dataset_dir / "eeg_cpp_trials.npy")
    times_ms = np.load(dataset_dir / "times_ms.npy")
    metadata = pd.read_csv(dataset_dir / "metadata.csv")
    channels = _read_channel_names(dataset_dir / "channel_names.txt")

    required_fields = list(DataContractConfig().required_metadata_columns)
    available_columns = metadata.columns.tolist()
    missing_required_fields = [field for field in required_fields if field not in available_columns]

    report: Dict[str, Any] = {
        "passed": True,
        "formal_training_ready": False,
        "package_level": "preliminary_only",
        "shape_summary": {
            "n_trials": int(eeg.shape[0]),
            "n_timepoints": int(eeg.shape[1]),
            "n_channels": int(eeg.shape[2]),
        },
        "times_match": bool(len(times_ms) == eeg.shape[1]),
        "channel_order_matches_contract": tuple(channels) == DataContractConfig().expected_channel_order,
        "available_metadata_columns": available_columns,
        "missing_required_metadata_columns": missing_required_fields or ["response_hand"],
        "blocking_issues": [
            "Current repository-prepared package is response-locked, not stimulus-locked.",
            "Current repository-prepared package lacks the full required formal metadata contract.",
        ],
    }
    write_json(output_dir / "stage0_blocking_audit_report.json", report)
    return report
