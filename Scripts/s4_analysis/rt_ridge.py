from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap: allow running this file directly (python rt_ridge.py ...)
# without installing the package.  Import guard prevents double-insertion.
# ---------------------------------------------------------------------------
_s1 = str(Path(__file__).resolve().parent.parent / "s1_modeling")
if _s1 not in sys.path:
    sys.path.insert(0, _s1)

import csv
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# =============================================================================
# § 1  Data Loading & Input Validation
# =============================================================================

def _load_latents(latent_npz: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load latent hidden states and the shared time axis from a .npz file.

    Parameters
    ----------
    latent_npz : Path to ``latents_full.npz`` written by
                 :func:`export_full_latents_from_checkpoint`.

    Returns
    -------
    latents  : (N, T, H) float32 array of GRU hidden states.
    times_ms : (T,) float32 array of response-locked time values.

    Raises
    ------
    KeyError  : If required keys ``"latents"`` or ``"times_ms"`` are missing.
    ValueError: If array shapes are inconsistent.
    """
    data = np.load(latent_npz)
    if "latents" not in data or "times_ms" not in data:
        raise KeyError(f"latents_full.npz must contain 'latents' and 'times_ms'; got {list(data.keys())}")
    latents  = data["latents"].astype(np.float32)   # (N, T, H)
    times_ms = data["times_ms"].astype(np.float32)  # (T,)
    if latents.ndim != 3:
        raise ValueError(f"Expected latents shape (N, T, H), got {latents.shape}")
    if times_ms.ndim != 1 or times_ms.shape[0] != latents.shape[1]:
        raise ValueError(
            f"times_ms length {times_ms.shape[0]} does not match latents T={latents.shape[1]}"
        )
    return latents, times_ms


def _load_behaviour(dataset_dir: Path) -> pd.DataFrame:
    """Load and minimally validate the trial-level behavioural metadata.

    Parameters
    ----------
    dataset_dir : Directory containing ``metadata.csv``.

    Returns
    -------
    pd.DataFrame with at least columns ``RT_ms`` and ``subject_id``.

    Raises
    ------
    FileNotFoundError : If ``metadata.csv`` is absent.
    KeyError          : If required columns are missing.
    """
    csv_path = Path(dataset_dir) / "metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {dataset_dir}")
    df = pd.read_csv(csv_path)
    required = {"RT_ms", "subject_id"}
    missing  = required - set(df.columns)
    if missing:
        raise KeyError(f"metadata.csv is missing required columns: {missing}")
    return df


# =============================================================================
# § 2  Feature Engineering
# =============================================================================

# --- 2a. Time-window averaging of hidden states ----------------------------

def _window_features(
    latents: np.ndarray,
    times_ms: np.ndarray,
    window_ms: Tuple[float, float],
) -> np.ndarray:
    """Average hidden states within a pre-response time window.

    Parameters
    ----------
    latents   : (N, T, H) GRU hidden-state array.
    times_ms  : (T,) response-locked time axis.
    window_ms : (start_ms, end_ms) inclusive boundaries.

    Returns
    -------
    (N, H) float32 array — mean hidden state over the window.

    Raises
    ------
    ValueError : If the window contains no valid time steps.
    """
    mask = (times_ms >= window_ms[0]) & (times_ms <= window_ms[1])
    if not mask.any():
        raise ValueError(
            f"Window {window_ms} ms contains no time steps. "
            f"times_ms ranges from {times_ms.min():.1f} to {times_ms.max():.1f} ms."
        )
    return latents[:, mask, :].mean(axis=1).astype(np.float32)  # (N, H)


# --- 2b. Baseline design matrix -------------------------------------------

def _baseline_design(df: pd.DataFrame) -> np.ndarray:
    """Build a per-trial baseline feature matrix from behavioural metadata.

    Baseline features
    -----------------
    - One-hot subject dummies (mean-centred; first subject dropped to avoid
      multicollinearity).
    - Normalised difficulty (z-scored ``coherence`` or ``difficulty`` column
      when present; otherwise a column of zeros).
    - Accuracy flag (``correctness`` column as 0/1 when present).

    Parameters
    ----------
    df : Trial-level metadata DataFrame.

    Returns
    -------
    (N, K) float32 design matrix.
    """
    parts: List[np.ndarray] = []

    # Subject dummies (drop first to avoid perfect multicollinearity).
    subj_dummies = pd.get_dummies(df["subject_id"], drop_first=True).values.astype(np.float32)
    parts.append(subj_dummies)

    # Difficulty / coherence (optional).
    diff_col = next((c for c in ("coherence", "difficulty") if c in df.columns), None)
    if diff_col is not None:
        vals = df[diff_col].values.astype(np.float32)
        std  = vals.std()
        parts.append(((vals - vals.mean()) / (std if std > 0 else 1.0)).reshape(-1, 1))
    else:
        parts.append(np.zeros((len(df), 1), dtype=np.float32))

    # Accuracy (optional).
    if "correctness" in df.columns:
        parts.append(df["correctness"].values.astype(np.float32).reshape(-1, 1))

    return np.concatenate(parts, axis=1)  # (N, K)


# --- 2c. Hand-crafted CPP features (for external validation) ---------------

def _cpp_features(dataset_dir: Path, df: pd.DataFrame) -> Optional[np.ndarray]:
    """Extract hand-crafted CPP amplitude and slope features when available.

    These are used as an external-validation baseline to test whether hidden
    states provide information *beyond* directly measured CPP features.

    Parameters
    ----------
    dataset_dir : Dataset directory (may contain ``eeg_cpp_trials.npy``).
    df          : Metadata DataFrame (used for alignment checks).

    Returns
    -------
    (N, 2) float32 array with columns [late_amplitude, pre_response_slope],
    or ``None`` if the EEG file is not available.
    """
    eeg_path = Path(dataset_dir) / "eeg_cpp_trials.npy"
    if not eeg_path.exists():
        return None
    eeg = np.load(eeg_path).astype(np.float32)  # (N, T, C)
    if eeg.shape[0] != len(df):
        return None  # Alignment mismatch — skip rather than corrupt.

    times_ms = np.load(Path(dataset_dir) / "times_ms.npy").astype(np.float32)
    cpp = eeg.mean(axis=-1)  # (N, T) — mean across CP1/CP2/CPz

    # Late amplitude: mean CPP in [-120, -50] ms.
    late_mask = (times_ms >= -120.0) & (times_ms <= -50.0)
    late_amp  = cpp[:, late_mask].mean(axis=1) if late_mask.any() else np.zeros(len(df))

    # Pre-response slope: linear regression coefficient in [-300, -50] ms.
    slope_mask = (times_ms >= -300.0) & (times_ms <= -50.0)
    if slope_mask.sum() >= 2:
        t_slope = times_ms[slope_mask]
        t_norm  = (t_slope - t_slope.mean()) / t_slope.std()
        slopes  = np.array([np.polyfit(t_norm, cpp[i, slope_mask], 1)[0] for i in range(len(df))])
    else:
        slopes = np.zeros(len(df))

    return np.stack([late_amp, slopes], axis=1).astype(np.float32)


# =============================================================================
# § 3  Ridge Regression & Cross-Validation
#      Outer KFold (grouped by subject) + inner alpha selection
# =============================================================================

_ALPHA_GRID = np.logspace(-3, 4, 30)


def _fit_predict_outer_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_outer_folds: int = 5,
    scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Leave-group-out Ridge regression with inner alpha tuning.

    Uses grouped K-Fold (grouped by subject ID) for the outer loop so that
    test subjects never appear in the training fold.  Alpha is chosen by
    inner 5-fold CV on the training fold only.

    Parameters
    ----------
    X             : (N, K) design matrix (baseline or baseline + hidden).
    y             : (N,)   target variable (log RT).
    groups        : (N,)   subject group labels for outer fold assignment.
    n_outer_folds : Number of outer CV folds (default: 5).
    scale         : Whether to z-score X within each fold (default: True).

    Returns
    -------
    y_pred  : (N,) out-of-fold predictions.
    y_true  : (N,) corresponding true values (same order as X rows).
    """
    kf     = KFold(n_splits=n_outer_folds, shuffle=True, random_state=42)
    y_pred = np.empty_like(y)

    for train_idx, test_idx in kf.split(X, y, groups=groups):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te       = X[test_idx]

        if scale:
            scaler = StandardScaler()
            X_tr   = scaler.fit_transform(X_tr)
            X_te   = scaler.transform(X_te)

        ridge = RidgeCV(alphas=_ALPHA_GRID, cv=5)
        ridge.fit(X_tr, y_tr)
        y_pred[test_idx] = ridge.predict(X_te)

    return y_pred, y


# =============================================================================
# § 4  Summary Statistics & Delta Metrics
# =============================================================================

def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute coefficient of determination R²."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _summarise_performance(
    results: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """Convert a nested results dict into a tidy summary DataFrame.

    Parameters
    ----------
    results : {window_label: {model_label: r2_value}} mapping.

    Returns
    -------
    pd.DataFrame with columns ``window``, ``model``, ``r2``.
    """
    rows = [
        {"window": window, "model": model, "r2": r2}
        for window, models in results.items()
        for model, r2 in models.items()
    ]
    return pd.DataFrame(rows)


# =============================================================================
# § 5  Visualisation & Output
# =============================================================================

def _save_performance_figure(
    df_perf: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save a grouped bar chart of R² by time window and model.

    Parameters
    ----------
    df_perf    : Tidy DataFrame with columns ``window``, ``model``, ``r2``.
    output_dir : Directory to write ``ridge_rt_performance.png`` into.
    """
    try:
        import matplotlib.pyplot as plt
        windows = df_perf["window"].unique()
        models  = df_perf["model"].unique()
        x       = np.arange(len(windows))
        width   = 0.8 / max(len(models), 1)

        fig, ax = plt.subplots(figsize=(8, 4))
        for i, model in enumerate(models):
            vals = [
                df_perf.loc[(df_perf["window"] == w) & (df_perf["model"] == model), "r2"].values[0]
                if ((df_perf["window"] == w) & (df_perf["model"] == model)).any()
                else float("nan")
                for w in windows
            ]
            ax.bar(x + i * width, vals, width, label=model)

        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(windows, rotation=20, ha="right")
        ax.set_ylabel("R²  (out-of-fold)")
        ax.set_title("Ridge RT prediction: R² by time window and model")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "ridge_rt_performance.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass  # Non-critical visualisation.


def _save_delta_figure(
    df_delta: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save a bar chart of ΔR² (hidden states vs. baseline) per window.

    Parameters
    ----------
    df_delta   : DataFrame with columns ``window`` and ``delta_r2``.
    output_dir : Directory to write ``ridge_rt_deltas.png`` into.
    """
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(df_delta["window"], df_delta["delta_r2"])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("ΔR²  (baseline+hidden − baseline)")
        ax.set_title("Incremental RT predictability from hidden states")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(output_dir / "ridge_rt_deltas.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass


# =============================================================================
# § 6  Public Entry Point
# =============================================================================

def run_ridge_rt_analysis(
    latent_npz: Path,
    dataset_dir: Path,
    output_dir: Path,
    window_definitions: Optional[Dict[str, Tuple[float, float]]] = None,
    n_outer_folds: int = 5,
) -> Dict[str, Any]:
    """Run Ridge regression predicting log(RT) from GRU hidden states.

    For each time window, two nested-CV Ridge models are evaluated:
      * ``"baseline"``         — subject dummies + difficulty + accuracy.
      * ``"baseline+hidden"``  — baseline features + window-averaged hidden states.

    Optionally (when CPP EEG data is present):
      * ``"baseline+cpp"``     — baseline + hand-crafted CPP features.
      * ``"baseline+cpp+hidden"`` — all three.

    Parameters
    ----------
    latent_npz         : Path to ``latents_full.npz``.
    dataset_dir        : Dataset directory (for ``metadata.csv`` and optionally
                         ``eeg_cpp_trials.npy``).
    output_dir         : Directory for CSV and figure outputs.
    window_definitions : Dict mapping window label → (start_ms, end_ms).
                         Defaults to four canonical pre-response windows.
    n_outer_folds      : Number of outer CV folds (default: 5).

    Returns
    -------
    Dict with key ``"performance"`` — a nested dict
    ``{window_label: {model_label: r2_value}}`` — and key ``"delta_r2"``
    — a dict ``{window_label: delta_r2_value}`` for the hidden-state increment.

    Output files
    ------------
    ``ridge_rt_performance.csv`` : Tidy R² table.
    ``ridge_rt_deltas.csv``      : ΔR² (hidden increment) per window.
    ``ridge_rt_performance.png`` : Grouped bar chart.
    ``ridge_rt_deltas.png``      : Delta bar chart.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if window_definitions is None:
        window_definitions = {
            "early  (−600 to −300 ms)": (-600.0, -300.0),
            "mid    (−300 to −120 ms)": (-300.0, -120.0),
            "late   (−120 to  −50 ms)": (-120.0,  -50.0),
            "full   (−600 to  −50 ms)": (-600.0,  -50.0),
        }

    # --- Load inputs ----------------------------------------------------------
    latents,  times_ms = _load_latents(Path(latent_npz))
    df_beh             = _load_behaviour(Path(dataset_dir))
    n_trials           = latents.shape[0]

    if len(df_beh) != n_trials:
        raise ValueError(
            f"latents has {n_trials} trials but metadata has {len(df_beh)} rows. "
            "Ensure both were built from the same dataset."
        )

    # --- Target variable: log(RT) ------------------------------------------
    log_rt = np.log(df_beh["RT_ms"].values.astype(np.float64))

    # --- Baseline design matrix -------------------------------------------
    X_baseline = _baseline_design(df_beh)

    # --- Optional CPP features -------------------------------------------
    X_cpp = _cpp_features(Path(dataset_dir), df_beh)

    # --- Subject groups (for grouped K-Fold) ------------------------------
    groups = df_beh["subject_id"].values

    # --- Evaluate per window ----------------------------------------------
    performance: Dict[str, Dict[str, float]] = {}
    delta_r2:    Dict[str, float]             = {}

    for label, window_ms in window_definitions.items():
        try:
            X_hidden = _window_features(latents, times_ms, window_ms)
        except ValueError:
            continue  # Skip windows that fall outside the time axis.

        X_aug = np.concatenate([X_baseline, X_hidden], axis=1)

        preds_base,  y_true = _fit_predict_outer_cv(X_baseline, log_rt, groups, n_outer_folds)
        preds_aug,   _      = _fit_predict_outer_cv(X_aug,      log_rt, groups, n_outer_folds)

        r2_base = _r2_score(y_true, preds_base)
        r2_aug  = _r2_score(y_true, preds_aug)

        window_results: Dict[str, float] = {
            "baseline":        r2_base,
            "baseline+hidden": r2_aug,
        }

        if X_cpp is not None:
            X_cpp_aug  = np.concatenate([X_baseline, X_cpp], axis=1)
            X_all      = np.concatenate([X_baseline, X_cpp, X_hidden], axis=1)
            preds_cpp, _ = _fit_predict_outer_cv(X_cpp_aug, log_rt, groups, n_outer_folds)
            preds_all, _ = _fit_predict_outer_cv(X_all,     log_rt, groups, n_outer_folds)
            window_results["baseline+cpp"]        = _r2_score(y_true, preds_cpp)
            window_results["baseline+cpp+hidden"] = _r2_score(y_true, preds_all)

        performance[label] = window_results
        delta_r2[label]    = r2_aug - r2_base

    # --- Persist results --------------------------------------------------
    df_perf  = _summarise_performance(performance)
    df_delta = pd.DataFrame([
        {"window": w, "delta_r2": d} for w, d in delta_r2.items()
    ])

    df_perf.to_csv(output_dir  / "ridge_rt_performance.csv", index=False)
    df_delta.to_csv(output_dir / "ridge_rt_deltas.csv",      index=False)

    _save_performance_figure(df_perf,  output_dir)
    _save_delta_figure(df_delta, output_dir)

    return {"performance": performance, "delta_r2": delta_r2}
