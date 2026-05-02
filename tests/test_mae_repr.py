"""Tests for MAE feature extraction using stubs (offline-safe)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from configs.config import Config
from models import mae_encoder as mae_encoder_mod
from representations import mae_repr as mae_repr_mod


@pytest.fixture(scope="session")
def mocked_mae_splits(config: Config, dataset: tuple[dict, dict]) -> tuple[dict, dict]:
    """Run MAE extraction with a stub extractor to avoid pretrained downloads."""

    _loaders, arrays = dataset
    xt = arrays["train"][0].reshape(-1, 1, 28, 28).astype(np.float32)
    xv = arrays["val"][0].reshape(-1, 1, 28, 28).astype(np.float32)
    xs = arrays["test"][0].reshape(-1, 1, 28, 28).astype(np.float32)

    def fake_builder(_cfg: Config, model_name=None, batch_size: int = 256):  # noqa: ANN001
        def _extract(x: np.ndarray) -> np.ndarray:
            n = np.asarray(x).shape[0]
            # Default to 768-dim (ViT-Base); multi-model tests cover varied dims.
            return np.zeros((n, 768), dtype=np.float32)

        return _extract

    with patch.object(mae_repr_mod, "build_mae_feature_extractor", new=fake_builder):
        feats, times = mae_repr_mod.extract_mae_representations(xt, xv, xs, config)
    return feats, times


def test_mae_features_keys(mocked_mae_splits: tuple[dict, dict]) -> None:
    """features_dict covers train/val/test."""
    feats, _times = mocked_mae_splits
    assert set(feats.keys()) == {"train", "val", "test"}


def test_mae_train_feature_shape(mocked_mae_splits: tuple[dict, dict]) -> None:
    """Train split encodes into (200, 768) for ViT-Base placeholders."""
    feats, _times = mocked_mae_splits
    assert feats["train"].shape == (200, 768)


def test_mae_val_feature_shape(mocked_mae_splits: tuple[dict, dict]) -> None:
    """Val split retains ViT backbone width."""
    feats, _times = mocked_mae_splits
    assert feats["val"].shape == (50, 768)


def test_mae_test_feature_shape(mocked_mae_splits: tuple[dict, dict]) -> None:
    """Test split retains ViT backbone width."""
    feats, _times = mocked_mae_splits
    assert feats["test"].shape == (50, 768)


def test_mae_timing_keys(mocked_mae_splits: tuple[dict, dict]) -> None:
    """timing_dict should mirror dataset splits."""
    _feats, times = mocked_mae_splits
    assert set(times.keys()) == {"train", "val", "test"}


def test_mae_timing_positive(mocked_mae_splits: tuple[dict, dict]) -> None:
    """Timers should remain non-negative for mocked backends."""
    _feats, times = mocked_mae_splits
    assert all(float(v) >= 0.0 for v in times.values())


def _vit_stub() -> object:
    """Minimal ViT-like module returning token features."""
    import torch
    import torch.nn as nn

    class _Stub(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self._dim = int(dim)

        def forward_features(self, xt: torch.Tensor) -> torch.Tensor:
            b = xt.size(0)
            return torch.zeros(b, 1, self._dim, device=xt.device, dtype=xt.dtype)

    return _Stub(dim=768).eval()


def test_extract_mae_representations_multi_shapes(config: Config, dataset: tuple[dict, dict]) -> None:
    """Multi-model MAE extraction should return per-model split dicts with 2D features."""
    _loaders, arrays = dataset
    xt = arrays["train"][0].reshape(-1, 1, 28, 28).astype(np.float32)
    xv = arrays["val"][0].reshape(-1, 1, 28, 28).astype(np.float32)
    xs = arrays["test"][0].reshape(-1, 1, 28, 28).astype(np.float32)

    def fake_builder(_cfg: Config, model_name=None, batch_size: int = 256):  # noqa: ANN001
        name = str(model_name) if model_name is not None else "vit_base_patch16_224.mae"
        dim = 768
        if "vit_large" in name:
            dim = 1024
        if "vit_huge" in name:
            dim = 1280

        def _extract(x: np.ndarray) -> np.ndarray:
            n = np.asarray(x).shape[0]
            return np.zeros((n, dim), dtype=np.float32)

        return _extract

    names = ["vit_base_patch16_224.mae", "vit_large_patch16_224.mae", "vit_huge_patch14_224.mae"]
    with patch.object(mae_repr_mod, "build_mae_feature_extractor", new=fake_builder):
        feats_by_model, times_by_model = mae_repr_mod.extract_mae_representations_multi(
            xt, xv, xs, config, model_names=names, batch_size=64
        )

    assert set(feats_by_model.keys()) == set(names)
    assert set(times_by_model.keys()) == set(names)
    for name in names:
        feats = feats_by_model[name]
        times = times_by_model[name]
        assert set(feats.keys()) == {"train", "val", "test"}
        assert set(times.keys()) == {"train", "val", "test"}
        assert feats["train"].ndim == 2
        assert feats["val"].ndim == 2
        assert feats["test"].ndim == 2
        assert feats["train"].shape[0] == 200
        assert feats["val"].shape[0] == 50
        assert feats["test"].shape[0] == 50
        assert all(float(v) >= 0.0 for v in times.values())


def test_extract_mae_features_output_shape(config: Config) -> None:
    """Stubbed timm model should yield (32, 768) features."""
    pytest.importorskip("timm")
    import timm

    stub = _vit_stub()
    with patch.object(timm, "create_model", return_value=stub):
        extract = mae_encoder_mod.build_mae_feature_extractor(config, model_name="vit_base_patch16_224.mae", batch_size=16)
        feats = extract(np.zeros((32, 1, 224, 224), dtype=np.float32))
        assert feats.shape == (32, 768)


def test_extract_mae_features_no_nan(config: Config) -> None:
    """Stubbed MAE features must remain finite."""
    pytest.importorskip("timm")
    import timm

    stub = _vit_stub()
    with patch.object(timm, "create_model", return_value=stub):
        extract = mae_encoder_mod.build_mae_feature_extractor(config, model_name="vit_base_patch16_224.mae", batch_size=16)
        feats = extract(np.ones((17, 1, 224, 224), dtype=np.float32))
        assert int(np.isnan(feats).sum()) == 0


def test_use_official_mae_false_uses_timm(config: Config) -> None:
    """Encoder construction should invoke timm.create_model."""
    pytest.importorskip("timm")
    import timm

    stub = _vit_stub()
    with patch.object(timm, "create_model") as mocked:
        mocked.return_value = stub
        _ = mae_encoder_mod.build_mae_feature_extractor(config, model_name="vit_base_patch16_224.mae")
        mocked.assert_called_once()


def test_use_official_mae_true_loads_checkpoint() -> None:
    """Official checkpoint mode is not wired in Config for this codebase yet."""
    if not hasattr(Config, "USE_OFFICIAL_MAE"):
        pytest.skip("Config lacks USE_OFFICIAL_MAE; official checkpoint branches are not tested here.")
    raise AssertionError("Extend Config before enabling this assertion path.")  # pragma: no cover
