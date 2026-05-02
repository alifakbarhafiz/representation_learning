"""Visualization utilities.

All functions save plots to disk and return the matplotlib Figure object.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE


def _ensure_dir(save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)


def plot_training_curves(history: dict[str, Any], title: str, save_path: str) -> plt.Figure:
    """Plot train/val loss and train/val accuracy side by side."""

    _ensure_dir(save_path)
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])

    epochs = np.arange(1, len(train_loss) + 1)
    best_epoch: Optional[int] = None
    if len(val_acc) > 0:
        best_epoch = int(np.argmax(val_acc)) + 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="Train Loss")
    axes[0].plot(epochs, val_loss, label="Val Loss")
    axes[0].set_title(f"{title} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="Train Acc")
    axes[1].plot(epochs, val_acc, label="Val Acc")
    axes[1].set_title(f"{title} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    if best_epoch is not None:
        for ax in axes:
            ax.axvline(best_epoch, linestyle="--", color="gray", alpha=0.7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_epoch_times(epoch_times: list[float], title: str, save_path: str) -> plt.Figure:
    """Bar chart of per-epoch wall-clock times with mean line."""

    _ensure_dir(save_path)
    times = np.asarray(epoch_times, dtype=float)
    epochs = np.arange(1, len(times) + 1)
    mean_t = float(times.mean()) if len(times) else 0.0

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(epochs, times, color=sns.color_palette("deep")[0])
    ax.axhline(mean_t, linestyle="--", color="gray", alpha=0.8, label=f"Mean={mean_t:.2f}s")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


@torch.no_grad()
def plot_reconstruction_samples(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    n: int = 8,
    save_path: str = "outputs/reconstructions.png",
) -> plt.Figure:
    """Show n originals and AE reconstructions side by side."""

    _ensure_dir(save_path)
    model.eval()
    batch = next(iter(dataloader))
    # Support both DataLoader styles:
    # - images only: Tensor[B,C,H,W]
    # - (images, labels): Tuple[Tensor[B,C,H,W], Tensor[B]]
    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    x = x.to(device)[:n]
    if x.ndim == 3:  # (C,H,W) -> (1,C,H,W)
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"Expected images with shape (B,C,H,W), got {tuple(x.shape)}")
    recon = model(x).detach().cpu().numpy()
    x_np = x.detach().cpu().numpy()

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        axes[0, i].imshow(x_np[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
    axes[0, 0].set_title("Original", loc="left")
    axes[1, 0].set_title("Reconstruction", loc="left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_tsne(
    features_dict: dict[str, np.ndarray],
    labels: np.ndarray,
    save_path: str,
    perplexity: int = 30,
    random_state: int = 42,
) -> plt.Figure:
    """Plot t-SNE embeddings for each method in subplots."""

    _ensure_dir(save_path)
    labels = np.asarray(labels).reshape(-1)
    methods = list(features_dict.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4), squeeze=False)
    for idx, m in enumerate(methods):
        feats = np.asarray(features_dict[m], dtype=np.float32)
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, max(5, (len(feats) - 1) // 3)),
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        )
        emb = tsne.fit_transform(feats)
        ax = axes[0, idx]
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab10", s=10, alpha=0.8)
        ax.set_title(f"t-SNE — {m}")
        ax.set_xticks([])
        ax.set_yticks([])
        if idx == n_methods - 1:
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_final_comparison(all_results: dict[str, dict[str, float]], save_path: str) -> plt.Figure:
    """Grouped bar chart comparing accuracy and macro F1 across methods."""

    _ensure_dir(save_path)
    methods = list(all_results.keys())
    acc = [float(all_results[m]["accuracy"]) for m in methods]
    f1 = [float(all_results[m]["macro_f1"]) for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, acc, width, label="Accuracy")
    ax.bar(x + width / 2, f1, width, label="Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Final Comparison (Test Set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    # Minimal sanity check (no file output errors)
    dummy = {"PCA": np.random.randn(200, 8), "AE": np.random.randn(200, 8)}
    y = np.random.randint(0, 5, size=(200,))
    plot_tsne(dummy, y, "project/outputs/_tsne_sanity.png")
    plot_final_comparison({"PCA": {"accuracy": 0.5, "macro_f1": 0.4}}, "project/outputs/_final_sanity.png")
    print("Visualization sanity checks saved to project/outputs/.")

