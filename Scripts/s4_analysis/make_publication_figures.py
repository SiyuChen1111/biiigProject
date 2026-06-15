"""make_publication_figures.py

Assembles all pre-computed results into publication-ready figures.

Usage
-----
    python Scripts/s4_analysis/make_publication_figures.py \
        --results-dir Results \
        --output-dir  Results/validation/figures/publication_style

Inputs
------
    Results/validation/hidden_state_classification_decoding.csv
    Results/validation/hidden_state_neural_regression_decoding.csv

Outputs
-------
    Results/validation/figures/publication_style/main_figure_2_hidden_state_relations.pdf
    Results/validation/figures/publication_style/supplementary_figure_behavioral_external_validation.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path


def make_figures(results_dir: Path, output_dir: Path) -> None:
    """Generate Figure 2 and Supplementary S1 from pre-computed result tables."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── TODO: implement figure assembly ──────────────────────────────────────
    # This stub will be populated when the figure-generation code is migrated
    # from the Jupyter notebook (Scripts/s4_analysis/notebooks/2_beh_z_reg.ipynb)
    # into a standalone script.
    print(f"[make_publication_figures] Results dir : {results_dir}")
    print(f"[make_publication_figures] Output dir  : {output_dir}")
    print("[make_publication_figures] ⚠️  Stub — no figures written yet.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication figures from pre-computed result tables."
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir",  type=Path, required=True)
    args = parser.parse_args()
    make_figures(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
