"""Tests for the MLP classifier and training helpers."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from configs.config import Config
from models.classifier import MLPClassifier, evaluate, train_epoch, train_mlp


def test_mlp_output_shape_128() -> None:
    """Hidden MLP maps 128 inputs to five logits."""
    model = MLPClassifier(input_dim=128, hidden_dims=[256, 128], num_classes=5)
    x = torch.randn(16, 128)
    y = model(x)
    assert y.shape == (16, 5)


def test_mlp_output_shape_768() -> None:
    """ViT-sized inputs should remain compatible with identical head layout."""
    model = MLPClassifier(input_dim=768, hidden_dims=[256, 128], num_classes=5)
    x = torch.randn(16, 768)
    y = model(x)
    assert y.shape == (16, 5)


def test_mlp_output_is_logits() -> None:
    """Raw logits must not resemble probabilities on an unconstrained Gaussian."""
    model = MLPClassifier(input_dim=128, hidden_dims=[256, 128], num_classes=5)
    x = torch.randn(16, 128)
    y = model(x)
    assert not torch.all((y >= 0) & (y <= 1)).item()


@pytest.mark.parametrize(
    "hidden_dims",
    [[128], [256, 128], [512, 256, 128]],
)
def test_mlp_different_hidden_dims(hidden_dims: list[int]) -> None:
    """Varying widths should preserve the logits shape."""
    model = MLPClassifier(input_dim=128, hidden_dims=hidden_dims, num_classes=5)
    x = torch.randn(16, 128)
    y = model(x)
    assert y.shape == (16, 5)


def test_train_epoch_returns_loss_and_acc() -> None:
    """train_epoch reports positive loss and accuracy in-range."""
    device = "cpu"
    ds = TensorDataset(torch.randn(32, 128), torch.randint(0, 5, (32,), dtype=torch.long))
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    model = MLPClassifier(128, [64], num_classes=5).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    crit = nn.CrossEntropyLoss()

    loss, acc = train_epoch(model, dl, opt, crit, device)
    assert isinstance(loss, float)
    assert loss > 0.0
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_evaluate_returns_loss_and_acc() -> None:
    """evaluate mirrors train_epoch bookkeeping on a loader."""
    device = "cpu"
    ds = TensorDataset(torch.randn(32, 128), torch.randint(0, 5, (32,), dtype=torch.long))
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    model = MLPClassifier(128, [64], num_classes=5).to(device)
    crit = nn.CrossEntropyLoss()

    loss, acc = evaluate(model, dl, crit, device)
    assert isinstance(loss, float)
    assert loss > 0.0
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_train_mlp_history_keys(config: Config) -> None:
    """train_mlp must emit the expected history keys."""
    torch.manual_seed(0)
    xtr = torch.randn(64, 128).numpy().astype("float32")
    ytr = torch.randint(0, 5, (64,)).numpy().astype("int64")
    xva = torch.randn(32, 128).numpy().astype("float32")
    yva = torch.randint(0, 5, (32,)).numpy().astype("int64")

    _model, hist = train_mlp(xtr, ytr, xva, yva, config)
    assert set(hist.keys()) == {"train_loss", "val_loss", "train_acc", "val_acc", "epoch_times"}


def test_train_mlp_history_length(config: Config) -> None:
    """History length should match configured MLP epochs."""
    torch.manual_seed(0)
    xtr = torch.randn(64, 128).numpy().astype("float32")
    ytr = torch.randint(0, 5, (64,)).numpy().astype("int64")
    xva = torch.randn(32, 128).numpy().astype("float32")
    yva = torch.randint(0, 5, (32,)).numpy().astype("int64")

    _model, hist = train_mlp(xtr, ytr, xva, yva, config)
    assert len(hist["train_loss"]) == config.MLP_EPOCHS


def test_epoch_times_are_positive(config: Config) -> None:
    """Recorded epoch timings should accumulate positive wall-clock time."""
    torch.manual_seed(0)
    xtr = torch.randn(64, 128).numpy().astype("float32")
    ytr = torch.randint(0, 5, (64,)).numpy().astype("int64")
    xva = torch.randn(32, 128).numpy().astype("float32")
    yva = torch.randint(0, 5, (32,)).numpy().astype("int64")

    _model, hist = train_mlp(xtr, ytr, xva, yva, config)
    assert all(t >= 0.0 for t in hist["epoch_times"])
    assert sum(hist["epoch_times"]) > 0.0


def test_val_acc_in_valid_range(config: Config) -> None:
    """Validation accuracy must remain a valid fraction."""
    torch.manual_seed(0)
    xtr = torch.randn(64, 128).numpy().astype("float32")
    ytr = torch.randint(0, 5, (64,)).numpy().astype("int64")
    xva = torch.randn(32, 128).numpy().astype("float32")
    yva = torch.randint(0, 5, (32,)).numpy().astype("int64")

    _model, hist = train_mlp(xtr, ytr, xva, yva, config)
    assert all(0.0 <= a <= 1.0 for a in hist["val_acc"])
