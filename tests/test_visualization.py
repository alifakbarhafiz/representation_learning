"""Visualization smoke tests writing to ephemeral tmp dirs."""

from __future__ import annotations

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.visualization import (
    plot_epoch_times,
    plot_final_comparison,
    plot_reconstruction_samples,
    plot_training_curves,
    plot_tsne,
)


@pytest.fixture(autouse=True)
def _force_agg_backend() -> None:
    """Use a headless backend for deterministic figure creation."""
    matplotlib.use("Agg", force=True)


def test_plot_training_curves_returns_figure(dummy_history: dict, tmp_path) -> None:
    """plot_training_curves must return a matplotlib Figure."""
    path = tmp_path / "curves.png"
    fig = plot_training_curves(dummy_history, title="MLP", save_path=str(path))
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_training_curves_saves_file(dummy_history: dict, tmp_path) -> None:
    """Training curve helper should persist PNG output."""
    path = tmp_path / "curves.png"
    plot_training_curves(dummy_history, title="MLP", save_path=str(path))
    assert path.exists()


def test_plot_training_curves_has_two_subplots(dummy_history: dict, tmp_path) -> None:
    """Loss/accuracy plots need two subplot axes."""
    path = tmp_path / "curves.png"
    fig = plot_training_curves(dummy_history, title="MLP", save_path=str(path))
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_epoch_times_returns_figure(tmp_path) -> None:
    """Epoch timing plots should yield a Figure."""
    path = tmp_path / "times.png"
    fig = plot_epoch_times([0.1, 0.2, 0.15], title="epochs", save_path=str(path))
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_epoch_times_saves_file(tmp_path) -> None:
    """Epoch timing helper should persist PNG output."""
    path = tmp_path / "times.png"
    plot_epoch_times([0.1, 0.2, 0.15], title="epochs", save_path=str(path))
    assert path.exists()


def test_plot_tsne_returns_figure(tmp_path) -> None:
    """t-SNE helper returns a matplotlib Figure instance."""
    path = tmp_path / "tsne.png"
    labels = np.repeat(np.arange(3, dtype=np.int64), repeats=40)
    feats = {
        "PCA": np.random.randn(120, 8).astype(np.float32),
        "AE": np.random.randn(120, 8).astype(np.float32),
        "MAE": np.random.randn(120, 8).astype(np.float32),
    }
    fig = plot_tsne(feats, labels=labels, save_path=str(path))
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_tsne_correct_subplot_count(tmp_path) -> None:
    """Subplot axes should scale with number of embedding methods."""
    path = tmp_path / "tsne.png"
    labels = np.repeat(np.arange(3, dtype=np.int64), repeats=20)
    feats = {
        "PCA": np.random.randn(60, 6).astype(np.float32),
        "AE": np.random.randn(60, 6).astype(np.float32),
        "MAE": np.random.randn(60, 6).astype(np.float32),
    }
    fig = plot_tsne(feats, labels=labels, save_path=str(path), perplexity=5)
    assert len(fig.axes) in (len(feats), len(feats) + 1)
    plt.close(fig)


def test_plot_final_comparison_returns_figure(tmp_path) -> None:
    """Comparison bar plots should emit a Figure handle."""
    path = tmp_path / "compare.png"
    res = {"PCA": {"accuracy": 0.6, "macro_f1": 0.58}, "AE": {"accuracy": 0.55, "macro_f1": 0.53}}
    fig = plot_final_comparison(res, save_path=str(path))
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


class _IdentityAE(torch.nn.Module):
    """Toy autoencoder returning the original batch unchanged."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_plot_reconstruction_samples_returns_figure(tmp_path) -> None:
    """Reconstruction plotter should tolerate standard image batches."""
    path = tmp_path / "recon.png"
    x = torch.rand(16, 1, 28, 28)
    ds = TensorDataset(x, x)
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    model = _IdentityAE()
    fig = plot_reconstruction_samples(model, dl, device="cpu", n=4, save_path=str(path))
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)
