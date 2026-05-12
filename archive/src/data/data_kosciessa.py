from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_DATASET_RELATIVE_PATH = Path("data/raw/CPP_low-level-2_Kosciessa_et_al_2021")
LEGACY_DATASET_RELATIVE_PATH = Path("1_Data/CPP_low-level-2_Kosciessa_et_al_2021")


def resolve_dataset_dir(dataset_dir: Path | None = None) -> Path:
    if dataset_dir is not None:
        return dataset_dir
    if DEFAULT_DATASET_RELATIVE_PATH.exists():
        return DEFAULT_DATASET_RELATIVE_PATH
    return LEGACY_DATASET_RELATIVE_PATH


DATASET_DIR = resolve_dataset_dir()


@dataclass(frozen=True)
class SubjectPaths:
    subject_id: str
    behavior_csv: Path
    eeg_vhdr: Path
    events_tsv: Path
    channels_tsv: Path


def list_subject_ids(dataset_dir: Path = DATASET_DIR) -> list[str]:
    return sorted(path.name for path in dataset_dir.glob("sub-*") if path.is_dir())


def get_subject_paths(subject_id: str, dataset_dir: Path = DATASET_DIR) -> SubjectPaths:
    subject_dir = dataset_dir / subject_id
    return SubjectPaths(
        subject_id=subject_id,
        behavior_csv=subject_dir / "beh" / f"{subject_id}_task-dynamic_beh.csv",
        eeg_vhdr=subject_dir / "eeg" / f"{subject_id}_task-dynamic_eeg.vhdr",
        events_tsv=subject_dir / "eeg" / f"{subject_id}_task-dynamic_events.tsv",
        channels_tsv=subject_dir / "eeg" / f"{subject_id}_task-dynamic_channels.tsv",
    )


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_behavior_table(subject_id: str, dataset_dir: Path = DATASET_DIR) -> pd.DataFrame:
    paths = get_subject_paths(subject_id=subject_id, dataset_dir=dataset_dir)
    behavior = _read_csv(paths.behavior_csv).copy()
    behavior["subject_id"] = subject_id
    behavior["trial_index"] = range(len(behavior))
    return behavior


def load_events_table(subject_id: str, dataset_dir: Path = DATASET_DIR) -> pd.DataFrame:
    paths = get_subject_paths(subject_id=subject_id, dataset_dir=dataset_dir)
    events = _read_tsv(paths.events_tsv).copy()
    events["subject_id"] = subject_id
    return events


def load_channels_table(subject_id: str, dataset_dir: Path = DATASET_DIR) -> pd.DataFrame:
    paths = get_subject_paths(subject_id=subject_id, dataset_dir=dataset_dir)
    channels = _read_tsv(paths.channels_tsv).copy()
    channels["subject_id"] = subject_id
    return channels


def _has_nearby_event(events: pd.DataFrame, sample: int, tolerance_samples: int = 5) -> bool:
    if events.empty:
        return False
    distances = (events["sample"].astype(int) - int(sample)).abs()
    return bool((distances <= tolerance_samples).any())


def build_subject_trial_table(subject_id: str, dataset_dir: Path = DATASET_DIR) -> pd.DataFrame:
    behavior = load_behavior_table(subject_id=subject_id, dataset_dir=dataset_dir)
    events = load_events_table(subject_id=subject_id, dataset_dir=dataset_dir)

    behavior["resp_onset_sample"] = pd.to_numeric(behavior["resp_onset_sample"], errors="coerce")
    behavior["stim_onset"] = pd.to_numeric(behavior["stim_onset"], errors="coerce")
    behavior["probe_rt"] = pd.to_numeric(behavior["probe_rt"], errors="coerce")
    behavior["probe_accuracy"] = pd.to_numeric(behavior["probe_accuracy"], errors="coerce")

    filtered = behavior.loc[
        behavior["is_valid_trial"].astype(bool)
        & ~behavior["is_missing_response"].astype(bool)
        & ~behavior["is_rt_outlier"].astype(bool)
        & behavior["resp_onset_sample"].notna()
    ].copy()

    filtered["resp_onset_sample"] = filtered["resp_onset_sample"].astype(int)
    filtered["stim_onset"] = filtered["stim_onset"].astype(int)
    filtered["has_nearby_event_marker"] = filtered["resp_onset_sample"].apply(
        lambda sample: _has_nearby_event(events=events, sample=sample)
    )
    filtered["event_alignment_warning"] = ~filtered["has_nearby_event_marker"]

    return filtered.reset_index(drop=True)


def build_all_trial_tables(
    subject_ids: Iterable[str] | None = None,
    dataset_dir: Path = DATASET_DIR,
) -> pd.DataFrame:
    ids = list(subject_ids) if subject_ids is not None else list_subject_ids(dataset_dir=dataset_dir)
    tables = [build_subject_trial_table(subject_id=subject_id, dataset_dir=dataset_dir) for subject_id in ids]
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def filter_subject_ids_with_paths(dataset_dir: Path = DATASET_DIR) -> list[str]:
    valid_ids: list[str] = []
    for subject_id in list_subject_ids(dataset_dir=dataset_dir):
        paths = get_subject_paths(subject_id=subject_id, dataset_dir=dataset_dir)
        if all(
            path.exists()
            for path in [paths.behavior_csv, paths.eeg_vhdr, paths.events_tsv, paths.channels_tsv]
        ):
            valid_ids.append(subject_id)
    return valid_ids
