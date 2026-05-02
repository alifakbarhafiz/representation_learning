"""Shared pytest fixtures (offline-first, fast training settings)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from configs.config import Config, set_global_seed
from models.autoencoder import ConvAutoencoder
from representations.ae_repr import train_autoencoder
from tests.compat import run_ae, run_pca

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch.manual_seed(42)
np.random.seed(42)


class _ImagesOnlyDataset(Dataset):
    """AE training dataset that returns only image tensors."""

    def __init__(self, x: torch.Tensor) -> None:
        self.x = x

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


def _make_synthetic_bundle(config: Config) -> tuple[dict, dict]:
    """Create RetinaMNIST-like splits without network access."""
    rng = np.random.default_rng(42)
    x_train = rng.random((200, 1, 28, 28), dtype=np.float32)
    x_val = rng.random((50, 1, 28, 28), dtype=np.float32)
    x_test = rng.random((50, 1, 28, 28), dtype=np.float32)

    y_train = rng.integers(0, 5, size=(200,), dtype=np.int64)
    y_val = rng.integers(0, 5, size=(50,), dtype=np.int64)
    y_test = rng.integers(0, 5, size=(50,), dtype=np.int64)

    xt = torch.from_numpy(x_train)
    xv = torch.from_numpy(x_val)
    xs = torch.from_numpy(x_test)

    loaders = {
        "train": DataLoader(
            _ImagesOnlyDataset(xt),
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        ),
        "val": DataLoader(
            _ImagesOnlyDataset(xv),
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        ),
        "test": DataLoader(
            _ImagesOnlyDataset(xs),
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        ),
    }

    arrays = {
        "train": (x_train.reshape(200, 784).astype(np.float32), y_train.astype(np.int64)),
        "val": (x_val.reshape(50, 784).astype(np.float32), y_val.astype(np.int64)),
        "test": (x_test.reshape(50, 784).astype(np.float32), y_test.astype(np.int64)),
    }
    return loaders, arrays


class _SilentTqdm:
    """Tiny tqdm stub that preserves iteration and common callback methods."""

    def __init__(self, iterable=None, **_kwargs):  # noqa: ANN001
        self._iterable = iterable if iterable is not None else iter(())

    def __iter__(self):
        return iter(self._iterable)

    def set_postfix(self, *_args, **_kwargs) -> None:
        """No-op: training loops call this for logging."""

        return None

    def set_description(self, *_args, **_kwargs) -> None:
        """No-op: AE training adjusts descriptions occasionally."""

        return None


@pytest.fixture(autouse=True)
def _silence_tqdm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence tqdm in project modules without breaking third-party tqdm imports."""

    monkeypatch.setattr("representations.ae_repr.tqdm", _SilentTqdm, raising=True)
    monkeypatch.setattr("models.classifier.tqdm", _SilentTqdm, raising=True)


@pytest.fixture(scope="session")
def config() -> Config:
    """Fast training configuration for the full test session."""
    from dataclasses import replace

    base = Config()
    return replace(
        base,
        DRY_RUN=True,
        SEED=42,
        AE_EPOCHS=2,
        MLP_EPOCHS=3,
        DEVICE="cpu",
        BATCH_SIZE=32,
    )


@pytest.fixture(scope="session")
def dataset(config: Config) -> tuple[dict, dict]:
    """Synthetic RetinaMNIST-like dry-run dataset (no downloads)."""
    set_global_seed(config.SEED)
    return _make_synthetic_bundle(config)


@pytest.fixture(scope="session")
def autoencoder() -> ConvAutoencoder:
    """Untrained convolutional autoencoder."""
    return ConvAutoencoder(latent_dim=128)


@pytest.fixture(scope="session")
def trained_autoencoder(config: Config, dataset: tuple[dict, dict]) -> tuple[torch.nn.Module, dict]:
    """AE trained for a couple epochs using synthetic loaders."""
    set_global_seed(config.SEED)
    loaders, _arrays = dataset
    model, history = train_autoencoder(loaders["train"], loaders["val"], config)
    return model, history


@pytest.fixture(scope="session")
def pca_features(config: Config, dataset: tuple[dict, dict]) -> dict:
    """PCA decomposition on synthetic flattened splits."""
    set_global_seed(config.SEED)
    _loaders, arrays = dataset
    xtr, _ytr = arrays["train"]
    xva, _yva = arrays["val"]
    xte, _yte = arrays["test"]
    return run_pca(xtr, xva, xte, config.PCA_N_COMPONENTS, config)


@pytest.fixture(scope="session")
def ae_features(config: Config, dataset: tuple[dict, dict]) -> tuple[dict, dict]:
    """AE features extracted after short training."""
    set_global_seed(config.SEED)
    loaders, arrays = dataset
    return run_ae(loaders, arrays, config)


@pytest.fixture
def sample_batch() -> torch.Tensor:
    """Mini batch tensor for AE shape checks."""
    torch.manual_seed(42)
    return torch.rand(4, 1, 28, 28)


@pytest.fixture
def sample_features_128() -> np.ndarray:
    """Random 128-dimensional features."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((32, 128), dtype=np.float32)


@pytest.fixture
def sample_features_768() -> np.ndarray:
    """Random 768-dimensional features."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((32, 768), dtype=np.float32)


@pytest.fixture
def sample_labels() -> np.ndarray:
    """Random 5-way labels."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 5, size=(32,), dtype=np.int64)


@pytest.fixture
def dummy_history() -> dict:
    """Fake classifier history for plotting tests."""
    return {
        "train_loss": [1.61, 1.53, 1.41, 1.39, 1.37],
        "val_loss": [1.62, 1.55, 1.43, 1.41, 1.39],
        "train_acc": [0.41, 0.44, 0.47, 0.48, 0.49],
        "val_acc": [0.40, 0.43, 0.46, 0.475, 0.485],
        "epoch_times": [0.21, 0.19, 0.23, 0.20, 0.22],
    }
