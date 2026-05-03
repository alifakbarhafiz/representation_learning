"""Autoencoder representation training + extraction."""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from configs.config import Config, set_global_seed
from models.autoencoder import build_autoencoder_for_images, get_encoder


def train_autoencoder(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train AE with MSE reconstruction loss.

    Returns:
        trained_autoencoder, history dict with:
            - epoch_losses: list[float]
            - epoch_times: list[float]
    """

    set_global_seed(config.SEED)
    device = config.DEVICE

    # Infer input shape from the loader (supports both images-only and (images, labels) loaders)
    first_batch = next(iter(train_loader))
    xb0 = first_batch[0] if isinstance(first_batch, (tuple, list)) else first_batch
    if xb0.ndim != 4:
        raise ValueError(f"Expected AE loader to yield 4D tensors, got shape {tuple(xb0.shape)}")
    in_channels = int(xb0.shape[1])
    input_size = int(xb0.shape[2])

    build = build_autoencoder_for_images(config, in_channels=in_channels, input_size=input_size)
    model = build.autoencoder.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.AE_LR)
    criterion = nn.MSELoss()

    history: Dict[str, Any] = {
        # Backwards-compatible key (legacy)
        "epoch_losses": [],
        "epoch_times": [],
        # New keys for JSON logging (one value per epoch)
        "train_loss": [],
        "val_loss": [],
    }

    pbar = tqdm(range(1, config.AE_EPOCHS + 1), desc="AE training", leave=True)
    for _epoch in pbar:
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n = 0
        for xb in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad(set_to_none=True)
            x_hat = model(xb)
            loss = criterion(x_hat, xb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            total_loss += float(loss.item()) * bs
            n += bs

        avg_loss = total_loss / max(1, n)
        dt = time.time() - t0
        history["epoch_losses"].append(avg_loss)
        history["epoch_times"].append(dt)
        history["train_loss"].append(float(avg_loss))

        # quick val loss (optional but useful for sanity)
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            vn = 0
            for xb in val_loader:
                xb = xb.to(device)
                x_hat = model(xb)
                loss = criterion(x_hat, xb)
                bs = xb.size(0)
                vloss += float(loss.item()) * bs
                vn += bs
            vloss = vloss / max(1, vn)
            history["val_loss"].append(float(vloss))

        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "val": f"{vloss:.4f}", "t": f"{dt:.2f}s"})

    return model, history


@torch.no_grad()
def extract_ae_features(
    encoder: nn.Module,
    images_numpy: np.ndarray,
    config: Config,
    batch_size: int = 256,
) -> np.ndarray:
    """Extract AE latent features from images (N,C,H,W) -> (N, latent_dim)."""

    device = config.DEVICE
    encoder.eval()
    encoder.to(device)
    x = np.asarray(images_numpy, dtype=np.float32)
    if x.ndim == 3:
        x = x[:, None, :, :]
    if x.ndim != 4:
        raise ValueError(f"Expected images with shape (N,C,H,W), got {x.shape}")
    feats: list[np.ndarray] = []
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i : i + batch_size]).to(device)
        z = encoder(xb).detach().cpu().numpy().astype(np.float32)
        feats.append(z)
    return np.concatenate(feats, axis=0)


def train_and_extract_ae_representations(
    train_loader: DataLoader,
    val_loader: DataLoader,
    x_train_img: np.ndarray,
    x_val_img: np.ndarray,
    x_test_img: np.ndarray,
    config: Config,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], nn.Module]:
    """Train AE, then extract encoder features for all splits."""

    t_train0 = time.time()
    model, history = train_autoencoder(train_loader, val_loader, config)
    history["train_seconds"] = float(time.time() - t_train0)
    encoder = get_encoder(model)

    extract_seconds: Dict[str, float] = {}

    t0 = time.time()
    ftr = extract_ae_features(encoder, x_train_img, config)
    extract_seconds["train"] = float(time.time() - t0)

    t1 = time.time()
    fva = extract_ae_features(encoder, x_val_img, config)
    extract_seconds["val"] = float(time.time() - t1)

    t2 = time.time()
    fte = extract_ae_features(encoder, x_test_img, config)
    extract_seconds["test"] = float(time.time() - t2)

    history["extract_seconds"] = extract_seconds

    features = {"train": ftr, "val": fva, "test": fte}
    return features, history, model


if __name__ == "__main__":
    from torch.utils.data import TensorDataset

    cfg = Config(DRY_RUN=True, AE_EPOCHS=2)
    set_global_seed(cfg.SEED)
    x = torch.rand(200, 3, 224, 224)
    ds = TensorDataset(x)
    tl = DataLoader(ds, batch_size=32, shuffle=True)
    vl = DataLoader(ds, batch_size=32, shuffle=False)
    model, hist = train_autoencoder(tl, vl, cfg)
    enc = get_encoder(model)
    feats = extract_ae_features(enc, x.numpy(), cfg)
    print("AE feats:", feats.shape, "hist keys:", hist.keys())

