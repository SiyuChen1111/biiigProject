from __future__ import annotations

from pathlib import Path


ANALYSIS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_ROOT.parent

DATASET_ROOT = PROJECT_ROOT / "1_Data" / "CPP_low-level-2_Kosciessa_et_al_2021"

OUTPUT_ROOT = ANALYSIS_ROOT / "outputs"
CPP_OUTPUT_ROOT = OUTPUT_ROOT / "cpp_epochs"
ANN_OUTPUT_ROOT = OUTPUT_ROOT / "ann"


def ensure_output_dirs() -> None:
    for path in (OUTPUT_ROOT, CPP_OUTPUT_ROOT, ANN_OUTPUT_ROOT):
        path.mkdir(parents=True, exist_ok=True)
