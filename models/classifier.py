"""MLP classifier and training utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from configs.config import Config, set_global_seed


class MLPClassifier(nn.Module):
    """Configurable MLP classifier with BatchNorm and Dropout."""

    def __init__(self, input_dim: int, hidden_dims: list[int], num_classes: int = 5):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                ]
            )
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == y).float().mean().item())


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """Train one epoch. Returns avg loss and accuracy."""

    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        total_acc += _accuracy(logits.detach(), yb) * bs
        n += bs
    return total_loss / max(1, n), total_acc / max(1, n)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """Evaluate. Returns avg loss and accuracy."""

    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        total_acc += _accuracy(logits, yb) * bs
        n += bs
    return total_loss / max(1, n), total_acc / max(1, n)


def _make_loader(
    x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
    ds = TensorDataset(xt, yt)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_mlp(
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_val: np.ndarray,
    labels_val: np.ndarray,
    config: Config,
    input_dim: int | None = None,
    num_classes: int = 5,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train an MLP classifier on precomputed features.

    Returns:
        model, history with keys:
            ['train_loss','val_loss','train_acc','val_acc','epoch_times']
    """

    set_global_seed(config.SEED)
    device = config.DEVICE

    xtr = np.asarray(features_train, dtype=np.float32)
    ytr = np.asarray(labels_train, dtype=np.int64).reshape(-1)
    xva = np.asarray(features_val, dtype=np.float32)
    yva = np.asarray(labels_val, dtype=np.int64).reshape(-1)

    in_dim = int(input_dim) if input_dim is not None else int(xtr.shape[1])

    train_loader = _make_loader(xtr, ytr, config.BATCH_SIZE, shuffle=True)
    val_loader = _make_loader(xva, yva, config.BATCH_SIZE, shuffle=False)

    model = MLPClassifier(in_dim, hidden_dims=config.MLP_HIDDEN_DIMS, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.MLP_LR)
    criterion = nn.CrossEntropyLoss()

    history: Dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "epoch_times": [],
    }

    pbar = tqdm(range(1, config.MLP_EPOCHS + 1), desc="MLP training", leave=True)
    for _epoch in pbar:
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["epoch_times"].append(dt)

        pbar.set_postfix(
            {
                "tr_loss": f"{tr_loss:.4f}",
                "va_loss": f"{va_loss:.4f}",
                "tr_acc": f"{tr_acc:.3f}",
                "va_acc": f"{va_acc:.3f}",
                "t": f"{dt:.2f}s",
            }
        )

    return model, history


if __name__ == "__main__":
    cfg = Config(DRY_RUN=True, MLP_EPOCHS=2)
    set_global_seed(cfg.SEED)
    xtr = np.random.randn(200, 128).astype(np.float32)
    ytr = np.random.randint(0, 5, size=(200,))
    xva = np.random.randn(50, 128).astype(np.float32)
    yva = np.random.randint(0, 5, size=(50,))
    model, hist = train_mlp(xtr, ytr, xva, yva, cfg)
    print("History keys:", hist.keys())

