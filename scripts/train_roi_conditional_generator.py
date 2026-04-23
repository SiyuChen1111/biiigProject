from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.optim.adam import Adam
from torch.utils.data import BatchSampler, DataLoader, Sampler, TensorDataset

from src.data.data_kosciessa import DATASET_DIR, build_subject_trial_table, filter_subject_ids_with_paths
from src.features.labels_cpp import CPP_ROI_CHANNELS, PAMS_WINDOW, compute_cpp_features, fit_cpp_label_transform, threshold_cpp_scores, transform_cpp_scores
from src.models.conditional_eeg_vae import ConditionalEEGVAE
from src.models.cpp_losses import cpp_difference_wave_loss, cpp_peak_loss, cpp_waveform_loss, extract_roi_cpp_waveform
from src.preprocessing.epoching import PHASE1_BASELINE, extract_response_locked_epochs, to_roi_epoched_subject_data


@dataclass(frozen=True)
class SubjectDataset:
    subject_id: str
    epochs: np.ndarray
    times: np.ndarray
    channel_names: list[str]
    trial_table: pd.DataFrame
    features: dict[str, np.ndarray]


@dataclass(frozen=True)
class SubjectBlock:
    epochs: np.ndarray
    trial_table: pd.DataFrame
    features: dict[str, np.ndarray]
    times: np.ndarray
    channel_names: list[str]


@dataclass(frozen=True)
class LabeledBlock:
    epochs: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    trial_table: pd.DataFrame
    times: np.ndarray
    channel_names: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ROI-only conditional EEG generator on CPP-like labels.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/roi_conditional_generator_smoke/train"),
        help=(
            "Training output directory. For retained ROI-only runs, prefer the output contract roots "
            "described in docs/protocol/roi-conditional-generator-output-contract.md "
            "(e.g. outputs/roi_conditional_generator_smoke/train)."
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--feature-weight", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=0.1)
    parser.add_argument("--lambda-cpp-wave", type=float, default=1.0)
    parser.add_argument("--lambda-cpp-peak", type=float, default=0.5)
    parser.add_argument("--lambda-cpp-diff", type=float, default=1.0)
    parser.add_argument("--label-weight", type=float, default=0.5)
    parser.add_argument("--samples-per-class", type=int, default=16)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _maybe_resolve_contract_root(train_output_dir: Path) -> Path | None:
    if train_output_dir.name != "train":
        return None
    candidate = train_output_dir.parent
    if candidate.name.startswith("roi_conditional_generator_"):
        return candidate
    return None


def _write_run_manifest(root: Path, train_dir: Path) -> None:
    manifest_path = root / "run_manifest.json"
    if manifest_path.exists():
        return
    manifest = {
        "contract_id": "roi_conditional_generator_v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "train_dir": str(train_dir.relative_to(root)),
        "analysis_dir": "analysis",
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _build_fold_assignment(subject_ids: list[str], n_folds: int = 5) -> list[list[str]]:
    folds = [[] for _ in range(n_folds)]
    for idx, subject_id in enumerate(sorted(subject_ids)):
        folds[idx % n_folds].append(subject_id)
    return folds


def _split_train_val_subjects(subject_ids: list[str]) -> tuple[list[str], list[str]]:
    sorted_ids = sorted(subject_ids)
    n_val = max(1, round(0.2 * len(sorted_ids)))
    val_ids = sorted_ids[:n_val]
    train_ids = sorted_ids[n_val:]
    if not train_ids:
        raise ValueError("Validation split consumed all training subjects.")
    return train_ids, val_ids


def _epochs_to_model_input(epochs: np.ndarray) -> np.ndarray:
    return epochs.reshape(epochs.shape[0], epochs.shape[1], epochs.shape[2], 1).astype(np.float32)


def _load_subject_dataset(subject_id: str, dataset_dir: Path) -> SubjectDataset:
    trial_table = build_subject_trial_table(subject_id=subject_id, dataset_dir=dataset_dir)
    epoched = extract_response_locked_epochs(subject_id=subject_id, trial_table=trial_table, dataset_dir=dataset_dir)
    roi_epoched = to_roi_epoched_subject_data(epoched)
    features = compute_cpp_features(
        epochs=roi_epoched.epochs,
        times=roi_epoched.times,
        channel_names=roi_epoched.channel_names,
        roi_channels=CPP_ROI_CHANNELS,
    )
    return SubjectDataset(
        subject_id=subject_id,
        epochs=roi_epoched.epochs,
        times=roi_epoched.times,
        channel_names=roi_epoched.channel_names,
        trial_table=roi_epoched.trial_table,
        features=features,
    )


def _concatenate_subject_blocks(blocks: list[SubjectDataset]) -> SubjectBlock:
    epochs = np.concatenate([block.epochs for block in blocks], axis=0)
    trial_table = pd.concat([block.trial_table for block in blocks], ignore_index=True)
    features = {key: np.concatenate([block.features[key] for block in blocks], axis=0) for key in ["ams", "pams", "slps"]}
    return SubjectBlock(
        epochs=epochs,
        trial_table=trial_table,
        features=features,
        times=blocks[0].times,
        channel_names=blocks[0].channel_names,
    )


def _apply_label_transform(block: SubjectBlock, transform) -> LabeledBlock:
    scores = transform_cpp_scores(block.features, transform)
    labels, keep_mask = threshold_cpp_scores(scores, transform)
    keep_mask = keep_mask.astype(bool)
    return LabeledBlock(
        epochs=block.epochs[keep_mask],
        labels=labels[keep_mask],
        scores=scores[keep_mask],
        trial_table=block.trial_table.iloc[keep_mask].reset_index(drop=True),
        times=block.times,
        channel_names=block.channel_names,
    )


def _validate_binary_block(block: LabeledBlock, *, split_name: str, fold_idx: int, require_both_classes: bool) -> None:
    n_trials = int(block.labels.shape[0])
    if n_trials == 0:
        raise ValueError(
            f"Fold {fold_idx} {split_name} split has zero retained trials after CPP thresholding. "
            "Adjust data selection or labeling policy before training."
        )
    present = set(np.unique(block.labels).tolist())
    if require_both_classes and present != {0, 1}:
        raise ValueError(
            f"Fold {fold_idx} {split_name} split must contain both CPP-like classes after thresholding; got labels {sorted(present)}."
        )


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.FloatTensor(x), torch.LongTensor(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class BalancedBatchSampler(BatchSampler):
    def __init__(self, labels: np.ndarray, batch_size: int) -> None:
        if batch_size < 2:
            raise ValueError("BalancedBatchSampler requires batch_size >= 2.")
        self.batch_size = batch_size
        self.labels = np.asarray(labels)
        self.neg_indices = np.flatnonzero(self.labels == 0)
        self.pos_indices = np.flatnonzero(self.labels == 1)
        if len(self.neg_indices) == 0 or len(self.pos_indices) == 0:
            raise ValueError("BalancedBatchSampler requires both label classes 0 and 1.")
        self.neg_per_batch = batch_size // 2
        self.pos_per_batch = batch_size - self.neg_per_batch
        if self.neg_per_batch == 0 or self.pos_per_batch == 0:
            raise ValueError("BalancedBatchSampler requires batch_size large enough to include both classes.")
        self.num_batches = int(np.ceil(max(len(self.neg_indices) / self.neg_per_batch, len(self.pos_indices) / self.pos_per_batch)))

    def __iter__(self):
        neg_pool = np.random.permutation(self.neg_indices)
        pos_pool = np.random.permutation(self.pos_indices)
        neg_cursor = 0
        pos_cursor = 0

        for _ in range(self.num_batches):
            if neg_cursor + self.neg_per_batch > len(neg_pool):
                neg_pool = np.concatenate([neg_pool[neg_cursor:], np.random.permutation(self.neg_indices)])
                neg_cursor = 0
            if pos_cursor + self.pos_per_batch > len(pos_pool):
                pos_pool = np.concatenate([pos_pool[pos_cursor:], np.random.permutation(self.pos_indices)])
                pos_cursor = 0

            batch = np.concatenate(
                [
                    neg_pool[neg_cursor : neg_cursor + self.neg_per_batch],
                    pos_pool[pos_cursor : pos_cursor + self.pos_per_batch],
                ]
            )
            neg_cursor += self.neg_per_batch
            pos_cursor += self.pos_per_batch
            np.random.shuffle(batch)
            yield batch.tolist()

    def __len__(self) -> int:
        return self.num_batches


def _make_class_balanced_loader(x: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    dataset = TensorDataset(torch.FloatTensor(x), torch.LongTensor(y))
    sampler = BalancedBatchSampler(labels=y, batch_size=batch_size)
    return DataLoader(dataset, batch_sampler=sampler)


def _compute_feature_consistency_loss(real_x: torch.Tensor, recon_x: torch.Tensor, sample_times: np.ndarray) -> torch.Tensor:
    real_epochs = real_x.squeeze(-1)
    recon_epochs = recon_x.squeeze(-1)
    real_times = np.linspace(float(sample_times[0]), float(sample_times[-1]), real_epochs.shape[-1], endpoint=True)
    recon_times = np.linspace(float(sample_times[0]), float(sample_times[-1]), recon_epochs.shape[-1], endpoint=True)
    real_features = compute_cpp_features(
        epochs=real_epochs.detach().cpu().numpy(),
        times=real_times,
        channel_names=list(CPP_ROI_CHANNELS),
        roi_channels=CPP_ROI_CHANNELS,
    )
    recon_features = compute_cpp_features(
        epochs=recon_epochs.detach().cpu().numpy(),
        times=recon_times,
        channel_names=list(CPP_ROI_CHANNELS),
        roi_channels=CPP_ROI_CHANNELS,
    )
    loss = torch.tensor(0.0, device=real_x.device, dtype=torch.float32)
    for key in ["ams", "pams", "slps"]:
        real_tensor = torch.tensor(real_features[key], device=real_x.device, dtype=torch.float32)
        recon_tensor = torch.tensor(recon_features[key], device=real_x.device, dtype=torch.float32)
        loss = loss + torch.mean((real_tensor - recon_tensor) ** 2)
    return loss / 3.0


def _compute_cpp_morphology_losses(
    real_x: torch.Tensor,
    recon_x: torch.Tensor,
    labels: torch.Tensor,
    sample_times: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    times_tensor = torch.as_tensor(sample_times, device=real_x.device, dtype=real_x.dtype)
    real_cpp = extract_roi_cpp_waveform(real_x, baseline_window=PHASE1_BASELINE, times=times_tensor)
    recon_cpp = extract_roi_cpp_waveform(recon_x, baseline_window=PHASE1_BASELINE, times=times_tensor)
    wave_loss = cpp_waveform_loss(real_cpp, recon_cpp)
    peak_loss = cpp_peak_loss(real_cpp, recon_cpp, times=times_tensor, peak_window=PAMS_WINDOW)
    diff_loss: torch.Tensor | None
    if torch.unique(labels).numel() < 2:
        diff_loss = None
    else:
        diff_loss = cpp_difference_wave_loss(real_cpp, recon_cpp, labels=labels)
    return wave_loss, peak_loss, diff_loss


def _evaluate_vae(
    model: ConditionalEEGVAE,
    loader: DataLoader,
    device: torch.device,
    beta: float,
    feature_weight: float,
    smoothness_weight: float,
    cpp_waveform_weight: float,
    cpp_peak_weight: float,
    cpp_difference_weight: float,
    label_weight: float,
    sample_times: np.ndarray,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    recon_losses: list[float] = []
    kl_losses: list[float] = []
    feature_losses: list[float] = []
    smoothness_losses: list[float] = []
    cpp_wave_losses: list[float] = []
    cpp_peak_losses: list[float] = []
    label_losses: list[float] = []
    accumulated_real_cpp: list[torch.Tensor] = []
    accumulated_recon_cpp: list[torch.Tensor] = []
    accumulated_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            recon_x, mu, logvar, label_logits = model(batch_x, batch_y)
            feature_loss = _compute_feature_consistency_loss(batch_x, recon_x, sample_times)
            cpp_wave_loss, cpp_peak_term, _ = _compute_cpp_morphology_losses(batch_x, recon_x, batch_y, sample_times)
            real_cpp = extract_roi_cpp_waveform(batch_x, baseline_window=PHASE1_BASELINE, times=torch.as_tensor(sample_times, device=batch_x.device, dtype=batch_x.dtype))
            recon_cpp = extract_roi_cpp_waveform(recon_x, baseline_window=PHASE1_BASELINE, times=torch.as_tensor(sample_times, device=batch_x.device, dtype=batch_x.dtype))
            loss, parts = model.loss_function(
                recon_x,
                batch_x,
                mu,
                logvar,
                label_logits=label_logits,
                labels=batch_y,
                beta=beta,
                feature_consistency_loss=feature_loss,
                smoothness_weight=smoothness_weight,
                feature_weight=feature_weight,
                cpp_waveform_loss=cpp_wave_loss,
                cpp_waveform_weight=cpp_waveform_weight,
                cpp_peak_loss=cpp_peak_term,
                cpp_peak_weight=cpp_peak_weight,
                label_weight=label_weight,
            )
            losses.append(float(loss.detach().cpu().item()))
            recon_losses.append(parts["recon_loss"])
            kl_losses.append(parts["kl_loss"])
            feature_losses.append(parts["feature_loss"])
            smoothness_losses.append(parts["smoothness_loss"])
            cpp_wave_losses.append(parts["cpp_waveform_loss"])
            cpp_peak_losses.append(parts["cpp_peak_loss"])
            label_losses.append(parts["label_loss"])
            accumulated_real_cpp.append(real_cpp.detach().cpu())
            accumulated_recon_cpp.append(recon_cpp.detach().cpu())
            accumulated_labels.append(batch_y.detach().cpu())

    if accumulated_real_cpp:
        dataset_real_cpp = torch.cat(accumulated_real_cpp, dim=0).to(device)
        dataset_recon_cpp = torch.cat(accumulated_recon_cpp, dim=0).to(device)
        dataset_labels = torch.cat(accumulated_labels, dim=0).to(device)
        dataset_cpp_diff_loss = cpp_difference_wave_loss(dataset_real_cpp, dataset_recon_cpp, dataset_labels)
    else:
        dataset_cpp_diff_loss = torch.tensor(float("nan"), device=device)

    recon_mean = float(np.mean(recon_losses)) if recon_losses else float("nan")
    kl_mean = float(np.mean(kl_losses)) if kl_losses else float("nan")
    feature_mean = float(np.mean(feature_losses)) if feature_losses else float("nan")
    smoothness_mean = float(np.mean(smoothness_losses)) if smoothness_losses else float("nan")
    cpp_wave_mean = float(np.mean(cpp_wave_losses)) if cpp_wave_losses else float("nan")
    cpp_peak_mean = float(np.mean(cpp_peak_losses)) if cpp_peak_losses else float("nan")
    cpp_diff_mean = float(dataset_cpp_diff_loss.detach().cpu().item()) if accumulated_real_cpp else float("nan")
    label_mean = float(np.mean(label_losses)) if label_losses else float("nan")
    total_mean = (
        recon_mean
        + beta * kl_mean
        + smoothness_weight * smoothness_mean
        + feature_weight * feature_mean
        + cpp_waveform_weight * cpp_wave_mean
        + cpp_peak_weight * cpp_peak_mean
        + cpp_difference_weight * cpp_diff_mean
        + label_weight * label_mean
    )
    return {
        "total_loss": total_mean,
        "recon_loss": recon_mean,
        "kl_loss": kl_mean,
        "feature_loss": feature_mean,
        "smoothness_loss": smoothness_mean,
        "cpp_waveform_loss": cpp_wave_mean,
        "cpp_peak_loss": cpp_peak_mean,
        "cpp_difference_loss": cpp_diff_mean,
        "label_loss": label_mean,
    }


def _save_condition_samples(model: ConditionalEEGVAE, output_dir: Path, device: torch.device, samples_per_class: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = torch.tensor([0] * samples_per_class + [1] * samples_per_class, dtype=torch.long, device=device)
    samples = model.sample(labels=labels, device=device).detach().cpu().numpy()
    np.save(output_dir / "conditional_samples.npy", samples)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    contract_root = _maybe_resolve_contract_root(output_dir)
    if contract_root is not None:
        contract_root.mkdir(parents=True, exist_ok=True)
        _write_run_manifest(root=contract_root, train_dir=output_dir)

    with open(output_dir / "train_run_config.json", "w", encoding="utf-8") as f:
        json.dump({"argv": sys.argv, "args": vars(args)}, f, indent=2, default=str)

    subject_ids = filter_subject_ids_with_paths(dataset_dir=args.dataset_dir)
    if args.max_subjects is not None:
        subject_ids = subject_ids[: args.max_subjects]
    if len(subject_ids) < 5:
        raise ValueError("Need at least 5 subjects for grouped 5-fold CV.")

    subject_blocks = {subject_id: _load_subject_dataset(subject_id=subject_id, dataset_dir=args.dataset_dir) for subject_id in subject_ids}
    fold_assignments = _build_fold_assignment(subject_ids=subject_ids, n_folds=5)
    fold_losses: list[dict[str, float | int]] = []

    reference_subject = next(iter(subject_blocks.values()))
    sample_times = reference_subject.times

    for fold_idx, test_subjects in enumerate(fold_assignments, start=1):
        remaining_subjects = [subject_id for subject_id in subject_ids if subject_id not in test_subjects]
        train_subjects, val_subjects = _split_train_val_subjects(remaining_subjects)

        train_block = _concatenate_subject_blocks([subject_blocks[s] for s in train_subjects])
        val_block = _concatenate_subject_blocks([subject_blocks[s] for s in val_subjects])
        test_block = _concatenate_subject_blocks([subject_blocks[s] for s in test_subjects])

        transform = fit_cpp_label_transform(train_block.features)
        train_ready = _apply_label_transform(train_block, transform)
        val_ready = _apply_label_transform(val_block, transform)
        test_ready = _apply_label_transform(test_block, transform)
        _validate_binary_block(train_ready, split_name="train", fold_idx=fold_idx, require_both_classes=True)
        _validate_binary_block(val_ready, split_name="validation", fold_idx=fold_idx, require_both_classes=True)
        _validate_binary_block(test_ready, split_name="test", fold_idx=fold_idx, require_both_classes=True)

        x_train = _epochs_to_model_input(train_ready.epochs)
        x_val = _epochs_to_model_input(val_ready.epochs)
        x_test = _epochs_to_model_input(test_ready.epochs)
        y_train = train_ready.labels
        y_val = val_ready.labels
        y_test = test_ready.labels

        train_loader = _make_class_balanced_loader(x_train, y_train, batch_size=args.batch_size)
        val_loader = _make_loader(x_val, y_val, batch_size=args.batch_size, shuffle=False)
        test_loader = _make_loader(x_test, y_test, batch_size=args.batch_size, shuffle=False)

        model = ConditionalEEGVAE.for_cpp_roi(samples=x_train.shape[2], latent_dim=args.latent_dim).to(device)
        optimizer = Adam(model.parameters(), lr=args.learning_rate)

        best_val_loss = float("inf")
        with tempfile.NamedTemporaryFile(suffix=f"_roi_fold{fold_idx}.pt", delete=False) as checkpoint_file:
            checkpoint_path = Path(checkpoint_file.name)
        torch.save(model.state_dict(), checkpoint_path)

        for _ in range(args.epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                recon_x, mu, logvar, label_logits = model(batch_x, batch_y)
                feature_loss = _compute_feature_consistency_loss(batch_x, recon_x, sample_times)
                cpp_wave_loss, cpp_peak_term, cpp_diff_term = _compute_cpp_morphology_losses(batch_x, recon_x, batch_y, sample_times)
                loss, _ = model.loss_function(
                    recon_x,
                    batch_x,
                    mu,
                    logvar,
                    label_logits=label_logits,
                    labels=batch_y,
                    beta=args.beta,
                    feature_consistency_loss=feature_loss,
                    smoothness_weight=args.smoothness_weight,
                    feature_weight=args.feature_weight,
                    cpp_waveform_loss=cpp_wave_loss,
                    cpp_waveform_weight=args.lambda_cpp_wave,
                    cpp_peak_loss=cpp_peak_term,
                    cpp_peak_weight=args.lambda_cpp_peak,
                    cpp_difference_loss=cpp_diff_term,
                    cpp_difference_weight=args.lambda_cpp_diff,
                    label_weight=args.label_weight,
                )
                loss.backward()
                optimizer.step()

            val_metrics = _evaluate_vae(
                model=model,
                loader=val_loader,
                device=device,
                beta=args.beta,
                feature_weight=args.feature_weight,
                smoothness_weight=args.smoothness_weight,
                cpp_waveform_weight=args.lambda_cpp_wave,
                cpp_peak_weight=args.lambda_cpp_peak,
                cpp_difference_weight=args.lambda_cpp_diff,
                label_weight=args.label_weight,
                sample_times=sample_times,
            )
            if val_metrics["total_loss"] < best_val_loss:
                best_val_loss = val_metrics["total_loss"]
                torch.save(model.state_dict(), checkpoint_path)

        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        checkpoint_path.unlink(missing_ok=True)

        test_metrics = _evaluate_vae(
            model=model,
            loader=test_loader,
            device=device,
            beta=args.beta,
            feature_weight=args.feature_weight,
            smoothness_weight=args.smoothness_weight,
            cpp_waveform_weight=args.lambda_cpp_wave,
            cpp_peak_weight=args.lambda_cpp_peak,
            cpp_difference_weight=args.lambda_cpp_diff,
            label_weight=args.label_weight,
            sample_times=sample_times,
        )
        test_metrics["fold"] = fold_idx
        test_metrics["n_test_subjects"] = len(test_subjects)
        test_metrics["n_test_trials"] = int(len(y_test))
        fold_losses.append({key: float(value) if isinstance(value, float) else int(value) for key, value in test_metrics.items()})

        fold_dir = output_dir / f"fold_{fold_idx}"
        _save_condition_samples(model=model, output_dir=fold_dir, device=device, samples_per_class=args.samples_per_class)

    summary = {
        "total_loss_mean": float(np.mean([float(item["total_loss"]) for item in fold_losses])),
        "total_loss_std": float(np.std([float(item["total_loss"]) for item in fold_losses], ddof=0)),
        "recon_loss_mean": float(np.mean([float(item["recon_loss"]) for item in fold_losses])),
        "recon_loss_std": float(np.std([float(item["recon_loss"]) for item in fold_losses], ddof=0)),
        "kl_loss_mean": float(np.mean([float(item["kl_loss"]) for item in fold_losses])),
        "kl_loss_std": float(np.std([float(item["kl_loss"]) for item in fold_losses], ddof=0)),
        "feature_loss_mean": float(np.mean([float(item["feature_loss"]) for item in fold_losses])),
        "feature_loss_std": float(np.std([float(item["feature_loss"]) for item in fold_losses], ddof=0)),
        "smoothness_loss_mean": float(np.mean([float(item["smoothness_loss"]) for item in fold_losses])),
        "smoothness_loss_std": float(np.std([float(item["smoothness_loss"]) for item in fold_losses], ddof=0)),
        "cpp_waveform_loss_mean": float(np.mean([float(item["cpp_waveform_loss"]) for item in fold_losses])),
        "cpp_waveform_loss_std": float(np.std([float(item["cpp_waveform_loss"]) for item in fold_losses], ddof=0)),
        "cpp_peak_loss_mean": float(np.mean([float(item["cpp_peak_loss"]) for item in fold_losses])),
        "cpp_peak_loss_std": float(np.std([float(item["cpp_peak_loss"]) for item in fold_losses], ddof=0)),
        "cpp_difference_loss_mean": float(np.mean([float(item["cpp_difference_loss"]) for item in fold_losses])),
        "cpp_difference_loss_std": float(np.std([float(item["cpp_difference_loss"]) for item in fold_losses], ddof=0)),
        "label_loss_mean": float(np.mean([float(item["label_loss"]) for item in fold_losses])),
        "label_loss_std": float(np.std([float(item["label_loss"]) for item in fold_losses], ddof=0)),
    }

    with open(output_dir / "fold_losses.json", "w", encoding="utf-8") as f:
        json.dump(fold_losses, f, indent=2)
    with open(output_dir / "summary_losses.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
