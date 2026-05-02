"""PCA representation extraction."""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.decomposition import PCA

from configs.config import Config, set_global_seed


def fit_transform_pca(
    x_train_img: np.ndarray,
    x_val_img: np.ndarray,
    x_test_img: np.ndarray,
    config: Config,
) -> Tuple[Dict[str, np.ndarray], PCA, float, Dict[str, float]]:
    """Fit PCA on flattened *training* images and transform all splits.

    Args:
        x_*_img: images in shape (N,C,H,W) float32 in [0,1]

    Returns:
        features_dict: {'train','val','test'} -> (N, n_components)
        pca: fitted PCA object
        fit_time: seconds
        transform_times: per-split transform time in seconds
    """

    set_global_seed(config.SEED)

    def _flatten(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return x.reshape(x.shape[0], -1)

    xtr = _flatten(x_train_img)
    xva = _flatten(x_val_img)
    xte = _flatten(x_test_img)

    pca = PCA(n_components=config.PCA_N_COMPONENTS, random_state=config.SEED)

    t0 = time.time()
    pca.fit(xtr)
    fit_time = time.time() - t0

    transform_times: Dict[str, float] = {}

    t1 = time.time()
    ftr = pca.transform(xtr).astype(np.float32)
    transform_times["train"] = time.time() - t1

    t2 = time.time()
    fva = pca.transform(xva).astype(np.float32)
    transform_times["val"] = time.time() - t2

    t3 = time.time()
    fte = pca.transform(xte).astype(np.float32)
    transform_times["test"] = time.time() - t3

    return {"train": ftr, "val": fva, "test": fte}, pca, fit_time, transform_times


if __name__ == "__main__":
    cfg = Config(DRY_RUN=True)
    xtr = np.random.rand(200, 3, 224, 224).astype(np.float32)
    xva = np.random.rand(50, 3, 224, 224).astype(np.float32)
    xte = np.random.rand(50, 3, 224, 224).astype(np.float32)
    feats, pca, fit_t, tt = fit_transform_pca(xtr, xva, xte, cfg)
    print({k: v.shape for k, v in feats.items()}, "fit", fit_t, "transform", tt)

