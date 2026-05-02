"""Convolutional Autoencoder for MedMNIST images.

Supports variable input size via max-pooling down to 7x7 and a configurable
number of channels (1 for grayscale, 3 for RGB-like datasets).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from configs.config import Config, set_global_seed


class ConvEncoder(nn.Module):
    """Encoder half of the convolutional autoencoder."""

    def __init__(self, in_channels: int, input_size: int, latent_dim: int):
        super().__init__()
        if input_size % 7 != 0:
            raise ValueError(f"input_size must be divisible by 7, got {input_size}")
        pools = int(np.log2(input_size // 7))
        if 2**pools * 7 != input_size:
            raise ValueError(
                f"input_size must be 7 * 2^k (e.g., 28 or 224), got {input_size}"
            )

        # Channel schedule: grows with depth; last stage feeds FC at 7x7
        channels = [32, 64, 128, 256, 512]
        depth = min(pools, len(channels))
        chosen = channels[:depth]

        layers: list[nn.Module] = []
        prev_c = in_channels
        for c in chosen:
            layers.append(nn.Conv2d(prev_c, c, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
            prev_c = c

        self.conv = nn.Sequential(*layers)
        self.out_channels = prev_c
        self.fc = nn.Linear(self.out_channels * 7 * 7, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        return z


class ConvDecoder(nn.Module):
    """Decoder half of the convolutional autoencoder."""

    def __init__(self, bottleneck_channels: int, input_size: int, latent_dim: int, out_channels: int):
        super().__init__()
        if input_size % 7 != 0:
            raise ValueError(f"input_size must be divisible by 7, got {input_size}")
        pools = int(np.log2(input_size // 7))
        if 2**pools * 7 != input_size:
            raise ValueError(
                f"input_size must be 7 * 2^k (e.g., 28 or 224), got {input_size}"
            )

        self.fc = nn.Linear(latent_dim, bottleneck_channels * 7 * 7)

        # Mirror of encoder channel schedule (reverse conv blocks)
        channels = [32, 64, 128, 256, 512]
        depth = min(pools, len(channels))
        chosen = channels[:depth]
        # chosen[-1] should match out_channels
        if bottleneck_channels != chosen[-1]:
            # fallback: allow custom out_channels without enforcing schedule
            chosen = chosen[:-1] + [bottleneck_channels]

        deconvs: list[nn.Module] = []
        prev_c = bottleneck_channels
        for next_c in reversed(chosen[:-1]):
            deconvs.append(nn.ConvTranspose2d(prev_c, next_c, kernel_size=2, stride=2))
            deconvs.append(nn.ReLU(inplace=True))
            prev_c = next_c

        # Final upsample steps until reaching input_size, then project to 1 or 3 channels
        # If depth==1, chosen[:-1] is empty and we still need pools-1 stages; handle by repeating.
        while len([m for m in deconvs if isinstance(m, nn.ConvTranspose2d)]) < pools:
            deconvs.append(nn.ConvTranspose2d(prev_c, prev_c, kernel_size=2, stride=2))
            deconvs.append(nn.ReLU(inplace=True))

        self.deconv = nn.Sequential(
            *deconvs,
            nn.Conv2d(prev_c, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self._final_out_channels = int(out_channels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(h.size(0), 64, 7, 7)
        x_hat = self.deconv(h)
        return x_hat


class ConvAutoencoder(nn.Module):
    """Full convolutional autoencoder (encoder + decoder)."""

    def __init__(self, latent_dim: int, in_channels: int = 1, input_size: int = 28):
        super().__init__()
        self.encoder = ConvEncoder(in_channels=in_channels, input_size=input_size, latent_dim=latent_dim)
        self.decoder = ConvDecoder(
            bottleneck_channels=self.encoder.out_channels,
            input_size=input_size,
            latent_dim=latent_dim,
            out_channels=in_channels,
        )
        self.in_channels = int(in_channels)
        self.input_size = int(input_size)

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
    ae = ConvAutoencoder(latent_dim=config.AE_LATENT_DIM, in_channels=1, input_size=28)
    return AEBuild(encoder=ae.encoder, autoencoder=ae)


def build_autoencoder_for_images(config: Config, in_channels: int, input_size: int) -> AEBuild:
    """Build an autoencoder for a given image shape."""

    set_global_seed(config.SEED)
    ae = ConvAutoencoder(latent_dim=config.AE_LATENT_DIM, in_channels=in_channels, input_size=input_size)
    return AEBuild(encoder=ae.encoder, autoencoder=ae)


if __name__ == "__main__":
    cfg = Config(DRY_RUN=True)
    build = build_autoencoder_for_images(cfg, in_channels=3, input_size=224)
    x = torch.from_numpy(np.random.rand(4, 3, 224, 224).astype(np.float32))
    y = build.autoencoder(x)
    z = build.encoder(x)
    print("AE out:", y.shape, "latent:", z.shape)

