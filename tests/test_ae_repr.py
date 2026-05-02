"""Tests for autoencoder representation helpers."""

from __future__ import annotations

import numpy as np

from configs.config import Config


def test_ae_features_keys(ae_features: tuple[dict, dict]) -> None:
    """AE feature dict should cover all splits."""
    feats, _hist = ae_features
    assert set(feats.keys()) == {"train", "val", "test"}


def test_ae_train_feature_shape(ae_features: tuple[dict, dict]) -> None:
    """Training AE features should follow (200, latent_dim) in dry run."""
    feats, _hist = ae_features
    assert feats["train"].shape == (200, 128)


def test_ae_val_feature_shape(ae_features: tuple[dict, dict]) -> None:
    """Validation AE features should follow (50, latent_dim)."""
    feats, _hist = ae_features
    assert feats["val"].shape == (50, 128)


def test_ae_test_feature_shape(ae_features: tuple[dict, dict]) -> None:
    """Test AE features should follow (50, latent_dim)."""
    feats, _hist = ae_features
    assert feats["test"].shape == (50, 128)


def test_ae_history_keys(ae_features: tuple[dict, dict]) -> None:
    """AE trainer should expose loss/time traces."""
    _feats, hist = ae_features
    assert set(hist.keys()) == {"epoch_losses", "epoch_times"}


def test_ae_history_length(config: Config, ae_features: tuple[dict, dict]) -> None:
    """AE training length should mirror configured epochs."""
    _feats, hist = ae_features
    assert len(hist["epoch_losses"]) == config.AE_EPOCHS


def test_ae_epoch_times_positive(ae_features: tuple[dict, dict]) -> None:
    """Each AE epoch timing entry should be strictly positive."""
    _feats, hist = ae_features
    assert all(float(t) > 0.0 for t in hist["epoch_times"])


def test_ae_loss_is_finite(ae_features: tuple[dict, dict]) -> None:
    """Recorded reconstruction losses must be finite numbers."""
    _feats, hist = ae_features
    assert all(np.isfinite(loss) for loss in hist["epoch_losses"])


def test_ae_features_no_nan(ae_features: tuple[dict, dict]) -> None:
    """Latent AE features must not contain NaNs."""
    feats, _hist = ae_features
    assert int(np.isnan(feats["train"]).sum()) == 0
