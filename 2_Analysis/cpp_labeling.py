from __future__ import annotations

import numpy as np
import pandas as pd


CPP_SCORE_COLUMNS = [
    "pre_response_slope",
    "late_amplitude",
    "post_response_drop",
    "pre_response_monotonicity",
]


def _subjectwise_zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    std_is_na = bool(np.asarray(pd.isna(std)).any())
    std_value = 0.0 if std_is_na else float(std)
    if std_value == 0.0:
        return pd.Series(np.zeros(len(series), dtype=float), index=series.index)
    return (series - series.mean()) / std_value


def add_cpp_score_and_labels(
    metadata: pd.DataFrame,
    *,
    subject_column: str = "subject_id",
) -> pd.DataFrame:
    result = metadata.copy()

    for column in CPP_SCORE_COLUMNS:
        z_column = f"{column}_z"
        result[z_column] = result.groupby(subject_column, group_keys=False)[column].apply(_subjectwise_zscore)

    result["cpp_score"] = (
        result["pre_response_slope_z"]
        + result["late_amplitude_z"]
        + result["post_response_drop_z"]
        + 0.5 * result["pre_response_monotonicity_z"]
    ) / 3.5

    morphology_gate = (
        (result["pre_response_slope"] > 0)
        & (result["late_amplitude"] > 0)
        & (result["post_response_drop"] > 0)
    )
    result["morphology_gate"] = morphology_gate.astype(int)

    subject_threshold = result.groupby(subject_column)["cpp_score"].transform("median")
    result["cpp_label"] = (morphology_gate & (result["cpp_score"] >= subject_threshold)).astype(int)
    result["cpp_threshold"] = subject_threshold
    result["task_proxy_label"] = 1

    return result
