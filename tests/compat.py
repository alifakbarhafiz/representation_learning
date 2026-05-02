"""Test-only wrappers matching documented experiment API helpers."""

from __future__ import annotations

import numpy as np

from configs.config import Config
from representations.ae_repr import train_and_extract_ae_representations
from representations.pca_repr import fit_transform_pca


def run_pca(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    n_components: int,
    config: Config,
) -> dict:
    """Fit PCA on training data only and transform all splits (no leakage).

    Accepts flattened (N, 784) arrays; reshapes internally to (N, 1, 28, 28)
    before delegating to the implementation.
    """

    cfg = config
    if cfg.PCA_N_COMPONENTS != n_components:
        from dataclasses import replace

        cfg = replace(config, PCA_N_COMPONENTS=n_components)

    x_train_img = np.asarray(X_train, dtype=np.float32).reshape(-1, 1, 28, 28)
    x_val_img = np.asarray(X_val, dtype=np.float32).reshape(-1, 1, 28, 28)
    x_test_img = np.asarray(X_test, dtype=np.float32).reshape(-1, 1, 28, 28)

    feats, pca, fit_time, tsplit = fit_transform_pca(x_train_img, x_val_img, x_test_img, cfg)
    transform_time = float(sum(tsplit.values()))
    return {
        "train": feats["train"],
        "val": feats["val"],
        "test": feats["test"],
        "pca": pca,
        "fit_time": float(fit_time),
        "transform_time": transform_time,
    }


def run_ae(
    loaders_dict: dict,
    arrays_dict: dict,
    config: Config,
):
    """Train AE via images loaders and extract latent features."""

    xt = arrays_dict["train"][0].reshape(-1, 1, 28, 28).astype(np.float32)
    xv = arrays_dict["val"][0].reshape(-1, 1, 28, 28).astype(np.float32)
    xs = arrays_dict["test"][0].reshape(-1, 1, 28, 28).astype(np.float32)

    feats, hist, _model = train_and_extract_ae_representations(
        loaders_dict["train"],
        loaders_dict["val"],
        xt,
        xv,
        xs,
        config,
    )
    return feats, hist
