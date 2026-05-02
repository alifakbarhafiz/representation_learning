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


def extract_mae_representations_multi(
    x_train_img: np.ndarray,
    x_val_img: np.ndarray,
    x_test_img: np.ndarray,
    config: Config,
    model_names: list[str] | None = None,
    batch_size: int = 256,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, float]]]:
    """Extract MAE/ViT features for multiple timm model names.

    Returns:
        features_by_model: model_name -> {'train','val','test'} -> features
        times_by_model: model_name -> {'train','val','test'} -> seconds
    """

    set_global_seed(config.SEED)
    names = config.MAE_MODEL_NAMES if model_names is None else list(model_names)

    features_by_model: Dict[str, Dict[str, np.ndarray]] = {}
    times_by_model: Dict[str, Dict[str, float]] = {}

    for name in names:
        extractor = build_mae_feature_extractor(config, model_name=name, batch_size=batch_size)
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

        features_by_model[name] = {"train": ftr, "val": fva, "test": fte}
        times_by_model[name] = times

    return features_by_model, times_by_model


if __name__ == "__main__":
    cfg = Config(DRY_RUN=True)
    xtr = np.random.rand(10, 1, 28, 28).astype(np.float32)
    xva = np.random.rand(5, 1, 28, 28).astype(np.float32)
    xte = np.random.rand(5, 1, 28, 28).astype(np.float32)
    feats, times = extract_mae_representations(xtr, xva, xte, cfg)
    print({k: v.shape for k, v in feats.items()}, times)

