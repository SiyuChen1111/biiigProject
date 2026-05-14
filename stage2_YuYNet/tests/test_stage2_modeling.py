import tempfile
import unittest
from typing import Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.analysis import run_core_latent_analysis
from modeling.config import AnalysisConfig, DataContractConfig, TrainingConfig
from modeling.controls import run_minimal_controls
from modeling.data_contract import validate_stage2_dataset
from modeling.dataset import build_pre_response_mask
from modeling.prepare_contract import audit_preliminary_stage2_dataset
from modeling.preparation import prepare_stage2_dataset_package
from modeling.train import train_stage2_pipeline
from scipy.io import savemat


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


def _make_preparation_sources(root: Path) -> Path:
    script_root = root / "script_pre_EEG"
    (script_root / "Kosciessa_et_al_2021" / "temp_data").mkdir(parents=True, exist_ok=True)
    (script_root / "van_et_al_2019" / "temp_data").mkdir(parents=True, exist_ok=True)

    kosciessa = np.arange(4 * 3 * 8, dtype=np.float32).reshape(4, 3, 8)
    savemat(script_root / "Kosciessa_et_al_2021" / "temp_data" / "resp_locked_erp.mat", {"resp_locked_erp": kosciessa})

    pd.DataFrame(
        {
            "subj_idx": ["ACC001", "ACC001"],
            "mode": ["mem", "mem"],
            "type": ["resp", "resp"],
            "rt": [500, 650],
            "acc": [1, 0],
            "item_1": ["a", "b"],
            "item_2": ["c", "d"],
            "item_cue": ["a", "d"],
        }
    ).to_csv(script_root / "van_et_al_2019" / "temp_data" / "data_beh_memory.csv", index=False)
    pd.DataFrame(
        {
            "0": [0.1, 0.2],
            "1": [0.3, 0.4],
            "2": [0.5, 0.6],
            "subject_id": ["ACC001", "ACC001"],
        }
    ).to_csv(script_root / "van_et_al_2019" / "temp_data" / "data_resp_locked_memory.csv", index=False)
    return script_root


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

    def test_prepare_generates_preliminary_package_and_blocking_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = _make_preparation_sources(root)
            dataset_dir = root / "dataset"
            prepare_report = prepare_stage2_dataset_package(dataset_dir, root / "evidence" / "stage0", source_root=source_root)
            audit_report = audit_preliminary_stage2_dataset(dataset_dir, root / "evidence" / "stage0")

            self.assertTrue(prepare_report["passed"])
            self.assertFalse(prepare_report["formal_training_ready"])
            self.assertTrue((dataset_dir / "eeg_cpp_trials.npy").exists())
            self.assertTrue((dataset_dir / "metadata.csv").exists())
            self.assertTrue((dataset_dir / "times_ms.npy").exists())
            self.assertTrue((dataset_dir / "channel_names.txt").exists())
            self.assertTrue((dataset_dir / "preprocessing_notes.md").exists())

            eeg = np.load(dataset_dir / "eeg_cpp_trials.npy")
            times_ms = np.load(dataset_dir / "times_ms.npy")
            metadata = pd.read_csv(dataset_dir / "metadata.csv")
            channels = (dataset_dir / "channel_names.txt").read_text(encoding="utf-8").strip().splitlines()

            self.assertEqual(eeg.shape, (4, 8, 3))
            self.assertEqual(len(times_ms), eeg.shape[1])
            self.assertEqual(channels, ["CP1", "CP2", "CPz"])
            self.assertNotIn("response_hand", metadata.columns)
            self.assertIn("missing_required_fields", metadata.columns)
            self.assertFalse(audit_report["formal_training_ready"])
            self.assertIn("response_hand", audit_report["missing_required_metadata_columns"])

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
