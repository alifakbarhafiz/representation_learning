"""Masked Autoencoder (MAE) / ViT feature extraction wrapper."""

from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np

from configs.config import Config, set_global_seed
from models.mae_encoder import build_mae_feature_extractor


def extract_mae_representations(
    x_train_img: np.ndarray,
    x_val_img: np.ndarray,
    x_test_img: np.ndarray,
    config: Config,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Extract MAE features for all splits.

    Returns:
        features dict and extraction times per split.
    """

    set_global_seed(config.SEED)
    extractor = build_mae_feature_extractor(config)

    times: Dict[str, float] = {}

    t0 = time.time()
    ftr = extractor(x_train_img)
    times["train"] = time.time() - t0

    t1 = time.time()
    fva = extractor(x_val_img)
    times["val"] = time.time() - t1

    t2 = time.time()
    fte = extractor(x_test_img)
    times["test"] = time.time() - t2

    return {"train": ftr, "val": fva, "test": fte}, times


if __name__ == "__main__":
    cfg = Config(DRY_RUN=True)
    xtr = np.random.rand(10, 1, 28, 28).astype(np.float32)
    xva = np.random.rand(5, 1, 28, 28).astype(np.float32)
    xte = np.random.rand(5, 1, 28, 28).astype(np.float32)
    feats, times = extract_mae_representations(xtr, xva, xte, cfg)
    print({k: v.shape for k, v in feats.items()}, times)

