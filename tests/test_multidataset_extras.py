"""Additional tests for multi-dataset upgrade."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from models.classifier import train_mlp
from utils.io import collect_environment, save_json


def test_save_json_writes_file(tmp_path) -> None:
    """save_json should create directories and write valid JSON."""
    path = tmp_path / "nested" / "out.json"
    save_json(str(path), {"a": 1, "b": {"c": True}})
    assert path.exists()
    assert path.read_text(encoding="utf-8").strip().startswith("{")


def test_train_mlp_uses_weighted_loss_runs(config) -> None:
    """train_mlp should run with balanced class weights without error."""
    # Extremely imbalanced labels to exercise weight calculation
    xtr = np.random.randn(200, 16).astype(np.float32)
    ytr = np.array([0] * 190 + [1] * 10, dtype=np.int64)
    xva = np.random.randn(50, 16).astype(np.float32)
    yva = np.random.randint(0, 2, size=(50,), dtype=np.int64)

    model, hist = train_mlp(xtr, ytr, xva, yva, config, input_dim=16, num_classes=2)
    assert hasattr(model, "forward")
    assert len(hist["train_loss"]) == config.MLP_EPOCHS
    assert isinstance(hist.get("class_weights"), dict)
    assert float(hist.get("train_seconds", 0.0)) >= 0.0


def test_collect_environment_has_required_keys() -> None:
    env = collect_environment()
    for k in ["python_version", "torch_version", "timm_version", "medmnist_version", "cuda_version"]:
        assert k in env


def test_mae_grayscale_repeats_channel_to_three(config) -> None:
    """MAE extractor should repeat grayscale channel to 3 before forward_features."""
    pytest.importorskip("timm")
    import timm

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_shape = None

        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            self.seen_shape = tuple(x.shape)
            b = x.size(0)
            return torch.zeros(b, 1, 768, device=x.device, dtype=x.dtype)

    stub = Stub().eval()

    from models.mae_encoder import build_mae_feature_extractor

    with patch.object(timm, "create_model", return_value=stub):
        extract = build_mae_feature_extractor(config, model_name="vit_base_patch16_224.mae", batch_size=4)
        _ = extract(np.zeros((5, 1, 224, 224), dtype=np.float32))

    assert stub.seen_shape is not None
    assert stub.seen_shape[1] == 3


def test_load_medmnist_dispatch_validation_error() -> None:
    """Invalid dataset name should raise ValueError early."""
    from configs.config import Config
    from data.dataset import load_medmnist

    cfg = Config(DRY_RUN=True, DATASET_SIZE=28, DEVICE="cpu")
    with pytest.raises(ValueError):
        _ = load_medmnist(cfg, dataset_name="not_a_dataset", dry_run=True, num_workers=0)

