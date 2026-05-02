"""Tests for offline synthetic RetinaMNIST-like dataset fixtures."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from configs.config import Config


def test_dry_run_train_size(dataset: tuple[dict, dict]) -> None:
    """Train split must contain 200 samples in dry-run mode."""
    _loaders, arrays_dict = dataset
    assert len(arrays_dict["train"][0]) == 200


def test_dry_run_val_size(dataset: tuple[dict, dict]) -> None:
    """Validation split must contain 50 samples in dry-run mode."""
    _loaders, arrays_dict = dataset
    assert len(arrays_dict["val"][0]) == 50


def test_dry_run_test_size(dataset: tuple[dict, dict]) -> None:
    """Test split must contain 50 samples in dry-run mode."""
    _loaders, arrays_dict = dataset
    assert len(arrays_dict["test"][0]) == 50


def test_pixel_values_normalized(dataset: tuple[dict, dict]) -> None:
    """Training pixels must live in [0, 1]."""
    _loaders, arrays_dict = dataset
    x_train = arrays_dict["train"][0]
    assert x_train.min() >= 0.0
    assert x_train.max() <= 1.0


def test_flat_array_shape(dataset: tuple[dict, dict]) -> None:
    """Dry-run training images must be shaped (200, 1, 28, 28) in test fixture."""
    _loaders, arrays_dict = dataset
    x_train = arrays_dict["train"][0]
    assert x_train.shape == (200, 1, 28, 28)


def test_label_range(dataset: tuple[dict, dict]) -> None:
    """Labels must be ordinal class ids in {0..4}."""
    _loaders, arrays_dict = dataset
    y_train = arrays_dict["train"][1]
    assert set(np.unique(y_train).tolist()).issubset({0, 1, 2, 3, 4})


def test_dataloader_batch_shape(dataset: tuple[dict, dict]) -> None:
    """AE train loader batches must be shaped (B, 1, 28, 28)."""
    loaders_dict, _arrays = dataset
    batch = next(iter(loaders_dict["train"]))
    assert isinstance(batch, torch.Tensor)
    assert batch.ndim == 4
    assert batch.shape[1:] == (1, 28, 28)


def test_dataloader_returns_both_outputs(dataset: tuple[dict, dict]) -> None:
    """Loaders and arrays must expose train/val/test splits."""
    loaders_dict, arrays_dict = dataset
    assert set(loaders_dict.keys()) == {"train", "val", "test"}
    assert set(arrays_dict.keys()) == {"train", "val", "test"}


def test_no_nan_in_arrays(dataset: tuple[dict, dict]) -> None:
    """Training features must not contain NaNs."""
    _loaders, arrays_dict = dataset
    x_train = arrays_dict["train"][0]
    assert int(np.isnan(x_train).sum()) == 0


def test_load_retinamnist_signature_matches_config() -> None:
    """Real loader should accept (config, dry_run) when medmnist is installed."""
    import inspect

    from data.dataset import load_retinamnist

    sig = inspect.signature(load_retinamnist)
    params = list(sig.parameters.keys())
    assert "config" in params
    assert "dry_run" in params
