"""pytest configuration: add project script directories to sys.path.

This file is picked up automatically by pytest when tests/ is the rootdir.
It inserts the three script package directories into sys.path so that tests
can import ``modeling``, ``training``, and ``analysis`` modules without
a package install step.

Import convention expected by tests
-------------------------------------
    from modeling.config  import TrainingConfig, ModelConfig, LossWeights
    from modeling.model   import CPPForwardGRU, masked_self_supervised_loss
    from modeling.dataset import load_stage2_dataset, make_dataloaders
    from training.train   import train_model, export_full_latents_from_checkpoint
    from training.controls import run_minimal_controls
    from training.sweep   import run_small_cpp_prior_sweep
    from analysis.rt_ridge import run_ridge_rt_analysis
"""

import sys
from pathlib import Path
import importlib

# Project root is one level above this file.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_SCRIPTS = _PROJECT_ROOT / "Scripts"
_SCRIPTS_STR = str(_SCRIPTS)
if _SCRIPTS_STR not in sys.path:
    sys.path.insert(0, _SCRIPTS_STR)

sys.modules.setdefault("modeling", importlib.import_module("s1_modeling"))
sys.modules.setdefault("training", importlib.import_module("s2_training"))
sys.modules.setdefault("analysis", importlib.import_module("s4_analysis"))
