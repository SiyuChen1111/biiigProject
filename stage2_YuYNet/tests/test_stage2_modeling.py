import tempfile
import unittest
from typing import Any, Dict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.analysis import run_core_latent_analysis
from modeling.config import AnalysisConfig, DataContractConfig, TrainingConfig
from modeling.controls import run_minimal_controls
from modeling.data_contract import validate_stage2_dataset
from modeling.dataset import build_pre_response_mask
from modeling.train import train_stage2_pipeline


def _make_synthetic_dataset(root: Path) -> Path:
    dataset_dir = root / "synthetic_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    n_trials, n_timepoints, n_channels = 12, 40, 3
    times_ms = np.linspace(-200, 580, num=n_timepoints, dtype=np.float32)
    eeg = np.random.default_rng(4).normal(size=(n_trials, n_timepoints, n_channels)).astype(np.float32)
    metadata = pd.DataFrame(
        {
            "subject_id": ["S1"] * 4 + ["S2"] * 4 + ["S3"] * 4,
            "trial_id": list(range(n_trials)),
            "condition": ["task"] * n_trials,
            "evidence_strength": np.tile([0.2, 0.4, 0.6, 0.8], 3),
            "choice": [0, 1] * 6,
            "correctness": [1, 0, 1, 1] * 3,
            "RT_ms": [420, 450, 480, 510] * 3,
            "response_hand": ["left", "right", "left", "right"] * 3,
            "artifact_rejection_flag": [False] * n_trials,
        }
    )
    np.save(dataset_dir / "eeg_cpp_trials.npy", eeg)
    np.save(dataset_dir / "times_ms.npy", times_ms)
    metadata.to_csv(dataset_dir / "metadata.csv", index=False)
    (dataset_dir / "channel_names.txt").write_text("CP1\nCP2\nCPz\n", encoding="utf-8")
    (dataset_dir / "preprocessing_notes.md").write_text(
        "Reference: average reference\nFilter: 0.1-30 Hz\nArtifact: ICA\nBaseline: -200 to 0 ms",
        encoding="utf-8",
    )
    return dataset_dir


class Stage2ModelingTests(unittest.TestCase):
    def test_mask_respects_pre_response_window(self):
        times_ms = np.array([-200, 0, 100, 200, 300], dtype=np.float32)
        rt_ms = np.array([260], dtype=np.float32)
        mask = build_pre_response_mask(times_ms, rt_ms, min_mask_lead_ms=50)
        self.assertEqual(mask.tolist(), [[False, True, True, True, False]])

    def test_stage1_validator_passes_synthetic_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = _make_synthetic_dataset(Path(tmp))
            report: Dict[str, Any] = validate_stage2_dataset(dataset_dir, Path(tmp) / "reports", DataContractConfig())
            self.assertTrue(report["passed"])

    def test_training_analysis_and_controls_run_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = _make_synthetic_dataset(Path(tmp))
            train_report: Dict[str, Any] = train_stage2_pipeline(
                dataset_dir,
                Path(tmp) / "evidence" / "stage2",
                TrainingConfig(max_epochs=3, early_stopping_patience=2, batch_size=4),
            )
            self.assertTrue(train_report["passed"])
            latent_path = Path(train_report["latent_exports"]["val"])
            analysis_report: Dict[str, Any] = run_core_latent_analysis(latent_path, Path(tmp) / "evidence" / "stage3", AnalysisConfig())
            controls_report: Dict[str, Any] = run_minimal_controls(latent_path, Path(tmp) / "evidence" / "stage4")
            self.assertTrue(analysis_report["passed"])
            self.assertTrue(controls_report["passed"])


if __name__ == "__main__":
    unittest.main()
