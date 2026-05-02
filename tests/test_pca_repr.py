"""Tests for PCA feature extraction."""

from __future__ import annotations

import numpy as np
import pytest

from configs.config import Config
from tests.compat import run_pca


def test_pca_output_keys(pca_features: dict) -> None:
    """PCA pipeline should return consistent dict keys."""
    assert set(pca_features.keys()) == {"train", "val", "test", "pca", "fit_time", "transform_time"}


def test_pca_train_shape(pca_features: dict) -> None:
    """Dry-run training PCA features should be shaped (200, 128)."""
    assert pca_features["train"].shape == (200, 128)


def test_pca_val_shape(pca_features: dict) -> None:
    """Validation PCA embeddings should match dry-run sample count."""
    assert pca_features["val"].shape == (50, 128)


def test_pca_test_shape(pca_features: dict) -> None:
    """Test PCA embeddings should match dry-run sample count."""
    assert pca_features["test"].shape == (50, 128)


def test_pca_n_components(pca_features: dict) -> None:
    """PCA object should record 128 retained components."""
    assert int(pca_features["pca"].n_components_) == 128


def test_no_data_leakage(dataset: tuple[dict, dict], config: Config, pca_features: dict) -> None:
    """Sklearn centering must match the training mean only."""
    _loaders, arrays = dataset
    X_train = arrays["train"][0].astype(np.float32)
    # Recompute using the same standalone API to emphasize train-only fit semantics.
    reran = run_pca(X_train, arrays["val"][0], arrays["test"][0], n_components=128, config=config)
    assert np.allclose(reran["pca"].mean_, X_train.mean(axis=0))


def test_fit_time_positive(pca_features: dict) -> None:
    """PCA fit should consume measurable wall time."""
    assert pca_features["fit_time"] > 0.0


def test_transform_time_positive(pca_features: dict) -> None:
    """Transforming all splits should record positive total time."""
    assert pca_features["transform_time"] > 0.0


def test_pca_no_nan_output(pca_features: dict) -> None:
    """PCA coordinates must be finite without NaNs."""
    assert int(np.isnan(pca_features["train"]).sum()) == 0
