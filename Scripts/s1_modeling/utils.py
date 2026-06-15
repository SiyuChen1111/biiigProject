from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all global random seeds for reproducibility.

    Covers Python ``random``, NumPy, and PyTorch (CPU + CUDA).

    Parameters
    ----------
    seed : Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Backwards-compatibility alias (old code used set_global_seed).
set_global_seed = set_seed


def ensure_dir(path: Path) -> Path:
    """Create *path* and all parents if they do not exist, then return *path*.

    Parameters
    ----------
    path : Directory path to create.

    Returns
    -------
    The same *path* object (for chaining).
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Serialise *payload* as pretty-printed JSON and write to *path*.

    Creates any missing parent directories automatically.

    Parameters
    ----------
    path    : Destination file path (will be created or overwritten).
    payload : JSON-serialisable dict.
    """
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def safe_float(value: Any) -> float:
    """Safely convert *value* to float, returning NaN on failure.

    Parameters
    ----------
    value : Any value to convert.

    Returns
    -------
    float representation of *value*, or ``float("nan")`` if conversion fails.
    """
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
