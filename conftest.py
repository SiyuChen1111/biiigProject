"""Root-level pytest configuration.

Injects the three script-package directories into ``sys.path`` so that the
``modeling``, ``training``, and ``analysis`` namespaces are resolvable by
pytest *before* any test module is collected.

Import convention for tests
---------------------------
    from modeling.config   import TrainingConfig, ModelConfig, LossWeights
    from modeling.model    import CPPForwardGRU
    from modeling.dataset  import make_dataloaders
    from training.train    import train_model, export_full_latents_from_checkpoint
    from training.controls import run_minimal_controls
    from training.sweep    import run_small_cpp_prior_sweep
    from analysis.rt_ridge import run_ridge_rt_analysis
"""
import sys
from pathlib import Path
import importlib

_ROOT = Path(__file__).resolve().parent
_SCRIPTS = _ROOT / "Scripts"
_SCRIPTS_STR = str(_SCRIPTS)
if _SCRIPTS_STR not in sys.path:
    sys.path.insert(0, _SCRIPTS_STR)

sys.modules.setdefault("modeling", importlib.import_module("s1_modeling"))
sys.modules.setdefault("training", importlib.import_module("s2_training"))
sys.modules.setdefault("analysis", importlib.import_module("s4_analysis"))
