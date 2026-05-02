"""Central experiment configuration.

All modules import configuration from here to keep defaults consistent.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass(frozen=True)
class Config:
    """Experiment configuration container."""

    DATASET_NAME: str = "retinamnist"
    DATASET_SIZE: int = 224
    RESULTS_DIR: str = "results"

    DRY_RUN: bool = False
    SEED: int = 42
    BATCH_SIZE: int = 64

    AE_EPOCHS: int = 30
    AE_LR: float = 1e-3
    AE_LATENT_DIM: int = 128

    PCA_N_COMPONENTS: int = 128

    MLP_HIDDEN_DIMS: list[int] = field(default_factory=lambda: [256, 128])
    MLP_EPOCHS: int = 50
    MLP_LR: float = 1e-3

    MAE_MODEL_NAME: str = "vit_base_patch16_224"
    MAE_MODEL_NAMES: list[str] = field(
        default_factory=lambda: [
            "vit_base_patch16_224.mae",
            "vit_large_patch16_224.mae",
            "vit_huge_patch14_224.mae",
        ]
    )

    DEVICE: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    cfg = Config()
    set_global_seed(cfg.SEED)
    print("Config sanity check:", cfg)
    print("Device:", cfg.DEVICE)

