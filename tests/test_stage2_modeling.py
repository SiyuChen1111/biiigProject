"""
Test suite for the Stage 2 CPP latent-dynamics pipeline.

Covers:
  - Config dataclass structure (ModelConfig, LossWeights, TrainingConfig)
  - Dataset loading and mask construction
  - Forward pass and loss computation (CPPForwardGRU)
  - End-to-end training + latent export (train_model)
  - Minimal controls (run_minimal_controls)
  - Ridge RT regression (run_ridge_rt_analysis)
  - Preliminary dataset preparation and audit (prepare/audit API)
  - Hyperparameter sweep smoke test (run_small_cpp_prior_sweep)
"""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from scipy.io import savemat

from modeling.config import AnalysisConfig, DataContractConfig, LossWeights, ModelConfig, TrainingConfig
from training.controls import run_minimal_controls
from modeling.data_contract import validate_stage2_dataset
from modeling.dataset import EEGWindowDataset, load_stage2_dataset, make_dataloaders
from modeling.model import CPPForwardGRU, ForwardOutputs, masked_self_supervised_loss
from modeling.low_rank_model import CPPLowRankRNN, LowRankRNNConfig, low_rank_self_supervised_loss
from modeling.preparation import audit_preliminary_stage2_dataset, prepare_stage2_dataset_package
from analysis.rt_ridge import run_ridge_rt_analysis
from training.sweep import run_small_cpp_prior_sweep
from training.train import export_full_latents_from_checkpoint, train_model
from training.low_rank_smoke import run_low_rank_smoke


# =============================================================================
# Shared synthetic-data helpers
# =============================================================================

def _make_synthetic_dataset(root: Path, n_trials: int = 12, n_timepoints: int = 40) -> Path:
    """Create a minimal well-formed dataset directory for testing."""
    dataset_dir = root / "synthetic_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(4)
    times_ms = np.linspace(-200.0, 580.0, num=n_timepoints, dtype=np.float32)
    eeg = rng.normal(size=(n_trials, n_timepoints, 3)).astype(np.float32)
    metadata = pd.DataFrame({
        "subject_id":   (["S1"] * 4 + ["S2"] * 4 + ["S3"] * 4)[:n_trials],
        "trial_id":     list(range(n_trials)),
        "condition":    ["task"] * n_trials,
        "correctness":  [1, 0, 1, 1] * (n_trials // 4),
        "RT_ms":        [420, 450, 480, 510] * (n_trials // 4),
        "alignment":    ["response_locked"] * n_trials,
    })
    np.save(dataset_dir / "eeg_cpp_trials.npy", eeg)
    np.save(dataset_dir / "times_ms.npy", times_ms)
    metadata.to_csv(dataset_dir / "metadata.csv", index=False)
    (dataset_dir / "channel_names.txt").write_text("CP1\nCP2\nCPz\n", encoding="utf-8")
    (dataset_dir / "preprocessing_notes.md").write_text(
        "Reference: average reference\nFilter: 0.1-30 Hz\n", encoding="utf-8"
    )
    return dataset_dir


def _make_preparation_sources(root: Path) -> Path:
    """Create minimal script_pre_EEG source files for preparation tests."""
    script_root = root / "script_pre_EEG"
    (script_root / "Kosciessa_et_al_2021" / "temp_data").mkdir(parents=True, exist_ok=True)
    (script_root / "van_et_al_2019"        / "temp_data").mkdir(parents=True, exist_ok=True)

    kosciessa = np.arange(4 * 3 * 8, dtype=np.float32).reshape(4, 3, 8)
    savemat(
        script_root / "Kosciessa_et_al_2021" / "temp_data" / "resp_locked_erp.mat",
        {"resp_locked_erp": kosciessa},
    )
    pd.DataFrame({
        "subj_idx": ["ACC001", "ACC001"],
        "mode": ["mem", "mem"],
        "type": ["resp", "resp"],
        "rt":   [500, 650],
        "acc":  [1, 0],
        "item_1": ["a", "b"],
        "item_2": ["c", "d"],
        "item_cue": ["a", "d"],
    }).to_csv(script_root / "van_et_al_2019" / "temp_data" / "data_beh_memory.csv", index=False)
    pd.DataFrame({
        "0": [0.1, 0.2], "1": [0.3, 0.4], "2": [0.5, 0.6],
        "subject_id": ["ACC001", "ACC001"],
    }).to_csv(script_root / "van_et_al_2019" / "temp_data" / "data_resp_locked_memory.csv", index=False)
    return script_root


# =============================================================================
# § 1  Config dataclass tests
# =============================================================================

class TestConfigDataclasses(unittest.TestCase):
    """Verify that the three-way config split behaves as expected."""

    def test_default_construction(self) -> None:
        cfg = TrainingConfig()
        self.assertEqual(cfg.model.hidden_dim, 32)
        self.assertEqual(cfg.loss.lambda_recon, 1.0)
        self.assertTrue(cfg.loss.enable_cpp_shape_prior)

    def test_backward_compat_properties(self) -> None:
        """TrainingConfig.hidden_dim etc. should proxy to model sub-config."""
        cfg = TrainingConfig(model=ModelConfig(hidden_dim=64))
        self.assertEqual(cfg.hidden_dim, 64)
        self.assertEqual(cfg.model.hidden_dim, 64)

    def test_loss_weight_override(self) -> None:
        cfg = TrainingConfig(loss=LossWeights(lambda_smooth=0.05, enable_cpp_shape_prior=False))
        self.assertEqual(cfg.loss.lambda_smooth, 0.05)
        self.assertFalse(cfg.loss.enable_cpp_shape_prior)
        self.assertFalse(cfg.enable_cpp_shape_prior)   # shim

    def test_replace_nested(self) -> None:
        cfg = TrainingConfig()
        new_cfg = replace(cfg, loss=replace(cfg.loss, lambda_cpp_prior=0.5))
        self.assertEqual(new_cfg.loss.lambda_cpp_prior, 0.5)
        self.assertEqual(cfg.loss.lambda_cpp_prior, 0.1)  # original unchanged


# =============================================================================
# § 2  Dataset and mask tests
# =============================================================================

class TestDataset(unittest.TestCase):
    """Verify data loading, normalisation, and mask construction."""

    def test_load_stage2_dataset_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = _make_synthetic_dataset(Path(tmp))
            cfg = TrainingConfig(analysis_window_ms=(-200.0, 580.0))
            eeg, targets, mask, times_ms, metadata = load_stage2_dataset(dataset_dir, cfg)
            self.assertEqual(eeg.shape, (12, 40, 3))
            self.assertEqual(targets.shape, (12, 40, 3))
            self.assertEqual(mask.shape, (12, 40))
            self.assertTrue(mask.any(), "At least some time steps should be masked valid.")

    def test_make_dataloaders_split_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = _make_synthetic_dataset(Path(tmp))
            cfg = TrainingConfig(
                batch_size=4,
                train_fraction=0.7,
                val_fraction=0.15,
                test_fraction=0.15,
                analysis_window_ms=(-200.0, 580.0),
            )
            eeg, targets, mask, times_ms, _ = load_stage2_dataset(dataset_dir, cfg)
            tr_l, va_l, te_l, idx = make_dataloaders(eeg, targets, mask, times_ms, cfg)
            total = len(idx["train"]) + len(idx["val"]) + len(idx["test"])
            self.assertEqual(total, 12)

    def test_eeg_window_dataset_item_keys(self) -> None:
        eeg = np.zeros((4, 10, 3), dtype=np.float32)
        targets = np.zeros_like(eeg)
        mask = np.ones((4, 10), dtype=bool)
        times_ms = np.linspace(-100.0, 100.0, 10, dtype=np.float32)
        ds = EEGWindowDataset(eeg, targets, mask, times_ms, np.arange(4))
        item = ds[0]
        for key in ("eeg", "target_future", "mask", "times_ms", "trial_idx"):
            self.assertIn(key, item)


# =============================================================================
# § 3  Model forward pass and loss
# =============================================================================

class TestModelAndLoss(unittest.TestCase):
    """Verify CPPForwardGRU output shapes and loss computation."""

    def setUp(self) -> None:
        self.cfg = TrainingConfig(model=ModelConfig(hidden_dim=8, projection_dim=4))
        self.model = CPPForwardGRU(n_channels=3, model_config=self.cfg.model)
        self.B, self.T, self.C = 4, 20, 3

    def test_forward_output_shapes(self) -> None:
        x = torch.randn(self.B, self.T, self.C)
        out: ForwardOutputs = self.model(x)
        self.assertEqual(out.reconstructed.shape, (self.B, self.T, self.C))
        self.assertEqual(out.predicted.shape,     (self.B, self.T, self.C))
        self.assertEqual(out.latents.shape,       (self.B, self.T, 8))

    def test_loss_returns_scalar_and_metrics(self) -> None:
        x = torch.randn(self.B, self.T, self.C)
        out = self.model(x)
        mask = torch.ones(self.B, self.T, dtype=torch.bool)
        times_ms = torch.linspace(-200.0, 580.0, self.T)
        loss, metrics = masked_self_supervised_loss(out, x, x, mask, times_ms, self.cfg.loss)
        self.assertEqual(loss.shape, ())
        self.assertIn("total_loss", metrics)
        self.assertGreater(metrics["total_loss"], 0.0)

    def test_loss_with_shape_prior_disabled(self) -> None:
        cfg = replace(self.cfg, loss=replace(self.cfg.loss, enable_cpp_shape_prior=False))
        x = torch.randn(self.B, self.T, self.C)
        out = self.model(x)
        mask = torch.ones(self.B, self.T, dtype=torch.bool)
        times_ms = torch.linspace(-200.0, 580.0, self.T)
        loss, metrics = masked_self_supervised_loss(out, x, x, mask, times_ms, cfg.loss)
        self.assertEqual(metrics["monotonic_loss"], 0.0)
        self.assertEqual(metrics["slope_floor_loss"], 0.0)


class TestLowRankModelAndSmoke(unittest.TestCase):
    """Verify the exploratory low-rank RNN smoke path."""

    def test_low_rank_forward_and_loss(self) -> None:
        cfg = TrainingConfig(loss=LossWeights(enable_cpp_shape_prior=False))
        model = CPPLowRankRNN(n_channels=3, config=LowRankRNNConfig(rank=3, population_dim=12))
        x = torch.randn(4, 20, 3)
        out = model(x)
        self.assertEqual(out.reconstructed.shape, (4, 20, 3))
        self.assertEqual(out.predicted.shape, (4, 20, 3))
        self.assertEqual(out.latents.shape, (4, 20, 3))
        mask = torch.ones(4, 20, dtype=torch.bool)
        times_ms = torch.linspace(-200.0, 580.0, 20)
        loss, metrics = low_rank_self_supervised_loss(out, x, x, mask, times_ms, cfg.loss)
        self.assertEqual(loss.shape, ())
        self.assertIn("total_loss", metrics)

    def test_low_rank_smoke_runs_and_saves_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = _make_synthetic_dataset(root, n_trials=12, n_timepoints=30)
            summary = run_low_rank_smoke(
                dataset_dir=dataset_dir,
                output_dir=root / "low_rank_smoke",
                ranks=(2,),
                max_trials=12,
                max_epochs=1,
                batch_size=4,
                population_dim=8,
                seed=3,
            )
            self.assertEqual(summary["ranks"], [2])
            self.assertTrue((root / "low_rank_smoke" / "low_rank_smoke_metrics.csv").exists())
            self.assertTrue((root / "low_rank_smoke" / "rank_2" / "metrics.csv").exists())
            self.assertTrue((root / "low_rank_smoke" / "rank_2" / "cpp_average_reconstruction.png").exists())
            self.assertTrue((root / "low_rank_smoke" / "rank_2" / "latent_trajectories_by_group.png").exists())


# =============================================================================
# § 4  Data contract validation
# =============================================================================

class TestDataContract(unittest.TestCase):
    def test_validates_synthetic_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = _make_synthetic_dataset(Path(tmp))
            report = validate_stage2_dataset(dataset_dir, Path(tmp) / "reports", DataContractConfig())
            self.assertTrue(report["passed"])


# =============================================================================
# § 5  Preparation and audit
# =============================================================================

class TestPreparation(unittest.TestCase):
    def test_prepare_generates_package_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = _make_preparation_sources(root)
            dataset_dir = root / "dataset"

            prep_report = prepare_stage2_dataset_package(
                dataset_dir, root / "evidence" / "s0", source_root=source_root
            )
            audit_report = audit_preliminary_stage2_dataset(dataset_dir, root / "evidence" / "s0")

            self.assertTrue(prep_report["passed"])
            self.assertFalse(prep_report["formal_training_ready"])
            for fname in ("eeg_cpp_trials.npy", "metadata.csv", "times_ms.npy",
                          "channel_names.txt", "preprocessing_notes.md"):
                self.assertTrue((dataset_dir / fname).exists(), f"{fname} missing")
            self.assertFalse(audit_report["formal_training_ready"])
            self.assertIn("response_hand", audit_report["missing_required_metadata_columns"])


# =============================================================================
# § 6  End-to-end training + latent export
# =============================================================================

class TestTraining(unittest.TestCase):
    def test_train_model_runs_and_saves_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = _make_synthetic_dataset(Path(tmp))
            cfg = TrainingConfig(
                max_epochs=3,
                early_stopping_patience=2,
                batch_size=4,
                analysis_window_ms=(-200.0, 580.0),
                model=ModelConfig(hidden_dim=8, projection_dim=4),
            )
            report = train_model(dataset_dir, Path(tmp) / "s2_out", cfg)
            self.assertIn("best_val_loss", report)
            self.assertTrue(Path(report["checkpoint_path"]).exists())

    def test_export_full_latents_aligns_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = _make_synthetic_dataset(root)
            cfg = TrainingConfig(
                max_epochs=2, early_stopping_patience=1, batch_size=4,
                analysis_window_ms=(-200.0, 580.0),
                model=ModelConfig(hidden_dim=8, projection_dim=4),
            )
            train_report = train_model(dataset_dir, root / "s2_out", cfg)
            out_path = export_full_latents_from_checkpoint(
                checkpoint_path=Path(train_report["checkpoint_path"]),
                dataset_dir=dataset_dir,
                output_dir=root / "latents",
            )
            self.assertTrue(out_path.exists())
            loaded = np.load(out_path)
            self.assertEqual(loaded["latents"].shape[0], 12)  # n_trials
            self.assertEqual(loaded["latents"].shape[2], 8)   # hidden_dim


# =============================================================================
# § 7  Controls
# =============================================================================

class TestControls(unittest.TestCase):
    def test_run_minimal_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = _make_synthetic_dataset(Path(tmp))
            cfg = TrainingConfig(
                max_epochs=2, early_stopping_patience=1, batch_size=4,
                analysis_window_ms=(-200.0, 580.0),
                model=ModelConfig(hidden_dim=8, projection_dim=4),
            )
            result = run_minimal_controls(dataset_dir, Path(tmp) / "controls", cfg)
            self.assertIn("untrained", result)
            self.assertIn("shuffled", result)
            self.assertIn("total_loss", result["untrained"])


# =============================================================================
# § 8  Ridge RT regression
# =============================================================================

class TestRidgeRT(unittest.TestCase):
    def test_ridge_rt_runs_and_saves_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = _make_synthetic_dataset(root)
            metadata = pd.read_csv(dataset_dir / "metadata.csv")
            times_ms = np.load(dataset_dir / "times_ms.npy")
            rng = np.random.default_rng(11)
            latents = rng.normal(size=(len(metadata), len(times_ms), 4)).astype(np.float32)
            latent_path = root / "latents_full.npz"
            np.savez(latent_path, latents=latents, times_ms=times_ms)
            # Write metadata next to npz so _load_latents can find it
            metadata.to_csv(root / "metadata.csv", index=False)

            out = run_ridge_rt_analysis(
                latent_npz=latent_path,
                dataset_dir=dataset_dir,
                output_dir=root / "ridge_out",
                window_definitions={"full_pre": (-200.0, 580.0)},
            )
            self.assertIn("performance", out)
            self.assertTrue((root / "ridge_out" / "ridge_rt_performance.csv").exists())
            self.assertTrue((root / "ridge_out" / "ridge_rt_deltas.csv").exists())


# =============================================================================
# § 9  Sweep smoke test
# =============================================================================

class TestSweep(unittest.TestCase):
    def test_small_sweep_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = _make_synthetic_dataset(Path(tmp))
            cfg = TrainingConfig(
                max_epochs=2, early_stopping_patience=1, batch_size=4,
                analysis_window_ms=(-200.0, 580.0),
                model=ModelConfig(hidden_dim=8, projection_dim=4),
            )
            summary = run_small_cpp_prior_sweep(dataset_dir, Path(tmp) / "sweep", cfg)
            self.assertIn("score", summary)
            self.assertTrue((Path(tmp) / "sweep" / "sweep_results.csv").exists())


if __name__ == "__main__":
    unittest.main()
