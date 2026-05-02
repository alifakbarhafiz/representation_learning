"""Tests for the convolutional autoencoder module."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from models.autoencoder import ConvAutoencoder, get_encoder


@pytest.mark.parametrize("latent_dim", [64, 128, 256])
def test_different_latent_dims(latent_dim: int, sample_batch: torch.Tensor) -> None:
    """Encoder width should follow the requested latent dimension."""
    model = ConvAutoencoder(latent_dim=latent_dim)
    z = model.encoder(sample_batch)
    assert z.shape == (4, latent_dim)


def test_encoder_output_shape(autoencoder: ConvAutoencoder, sample_batch: torch.Tensor) -> None:
    """Default encoder maps a 4x1x28x28 batch to 128-dim latents."""
    z = autoencoder.encoder(sample_batch)
    assert z.shape == (4, 128)


def test_decoder_output_shape(autoencoder: ConvAutoencoder) -> None:
    """Decoder maps latents back to image-sized tensors."""
    z = torch.randn(4, 128)
    x_hat = autoencoder.decoder(z)
    assert x_hat.shape == (4, 1, 28, 28)


def test_full_forward_shape(autoencoder: ConvAutoencoder, sample_batch: torch.Tensor) -> None:
    """Full AE forward pass preserves spatial layout."""
    x_hat = autoencoder(sample_batch)
    assert x_hat.shape == (4, 1, 28, 28)


def test_reconstruction_range(autoencoder: ConvAutoencoder, sample_batch: torch.Tensor) -> None:
    """Sigmoid outputs should stay within [0, 1]."""
    x_hat = autoencoder(sample_batch)
    assert float(x_hat.detach().min()) >= 0.0
    assert float(x_hat.detach().max()) <= 1.0


def test_get_encoder_returns_module(autoencoder: ConvAutoencoder) -> None:
    """get_encoder should return an nn.Module."""
    enc = get_encoder(autoencoder)
    assert isinstance(enc, nn.Module)


def test_get_encoder_output_shape(autoencoder: ConvAutoencoder, sample_batch: torch.Tensor) -> None:
    """Encoder helper should match manual encoder forward."""
    enc = get_encoder(autoencoder)
    z = enc(sample_batch)
    assert z.shape == (4, 128)


def test_encoder_no_grad_needed(autoencoder: ConvAutoencoder, sample_batch: torch.Tensor) -> None:
    """Encoder forward must be safe under torch.no_grad()."""
    with torch.no_grad():
        z = autoencoder.encoder(sample_batch)
    assert z.shape == (4, 128)


def test_loss_decreases_after_one_step(sample_batch: torch.Tensor) -> None:
    """Several optimization steps should monotonically improve MSE on a fixed batch."""
    torch.manual_seed(0)
    model = ConvAutoencoder(latent_dim=128)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    crit = nn.MSELoss()

    x = sample_batch.detach()
    with torch.no_grad():
        loss_before = float(crit(model(x), x).item())

    loss_after = loss_before
    for _ in range(25):
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), x)
        loss.backward()
        opt.step()
        with torch.no_grad():
            loss_after = float(crit(model(x), x).item())

    assert loss_after < loss_before

