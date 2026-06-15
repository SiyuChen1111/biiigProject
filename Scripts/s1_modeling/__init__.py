"""modeling — CPP latent dynamics modelling package.

Public API
----------
Configuration
    DataContractConfig  : Dataset structural requirements.
    ModelConfig         : CPPForwardGRU architecture hyperparameters.
    LossWeights         : Self-supervised loss weights and shape-prior config.
    TrainingConfig      : Training loop, data pipeline, and split settings.
    AnalysisConfig      : Stage 3/4 latent readout settings.

Model
    CPPForwardGRU           : Causal GRU encoder for EEG latent dynamics.
    ForwardOutputs          : Named output container (reconstructed / predicted / latents).
    masked_self_supervised_loss : Composite self-supervised loss function.

Dataset
    EEGWindowDataset        : Trial-level PyTorch Dataset.
    Stage2SplitArtifacts    : Train/val/test split metadata.
    load_stage2_dataset     : Load, validate, and normalise the EEG dataset.
    make_dataloaders        : Wrap arrays into DataLoaders.

Utilities
    set_global_seed         : Set random seeds for reproducibility.
    default_evidence_dir    : Canonical output directory helper.
"""

from .config import (
    AnalysisConfig,
    DataContractConfig,
    LossWeights,
    ModelConfig,
    TrainingConfig,
    default_evidence_dir,
)
from .dataset import (
    EEGWindowDataset,
    Stage2SplitArtifacts,
    build_pre_response_mask,
    load_stage2_dataset,
    make_dataloaders,
)
from .model import (
    CPPForwardGRU,
    ForwardOutputs,
    masked_self_supervised_loss,
)
from .utils import set_global_seed

__all__ = [
    # config
    "AnalysisConfig",
    "DataContractConfig",
    "LossWeights",
    "ModelConfig",
    "TrainingConfig",
    "default_evidence_dir",
    # dataset
    "EEGWindowDataset",
    "Stage2SplitArtifacts",
    "build_pre_response_mask",
    "load_stage2_dataset",
    "make_dataloaders",
    # model
    "CPPForwardGRU",
    "ForwardOutputs",
    "masked_self_supervised_loss",
    # utils
    "set_global_seed",
]
