"""Convolutional Autoencoder for 28x28 grayscale images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from configs.config import Config, set_global_seed


class ConvEncoder(nn.Module):
    """Encoder half of the convolutional autoencoder."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
        )
        self.fc = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        return z


class ConvDecoder(nn.Module):
    """Decoder half of the convolutional autoencoder."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 14x14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),  # 28x28
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(h.size(0), 64, 7, 7)
        x_hat = self.deconv(h)
        return x_hat


class ConvAutoencoder(nn.Module):
    """Full convolutional autoencoder (encoder + decoder)."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


def get_encoder(model: nn.Module) -> nn.Module:
    """Return the encoder half of an autoencoder-like module."""

    if hasattr(model, "encoder"):
        return getattr(model, "encoder")
    raise AttributeError("Model does not have an 'encoder' attribute.")


@dataclass(frozen=True)
class AEBuild:
    """Convenience bundle containing encoder and full AE."""

    encoder: nn.Module
    autoencoder: nn.Module


def build_autoencoder(config: Config) -> AEBuild:
    """Build an autoencoder according to config."""

    set_global_seed(config.SEED)
    ae = ConvAutoencoder(latent_dim=config.AE_LATENT_DIM)
    return AEBuild(encoder=ae.encoder, autoencoder=ae)


if __name__ == "__main__":
    cfg = Config(DRY_RUN=True)
    build = build_autoencoder(cfg)
    x = torch.from_numpy(np.random.rand(4, 1, 28, 28).astype(np.float32))
    y = build.autoencoder(x)
    z = build.encoder(x)
    print("AE out:", y.shape, "latent:", z.shape)

