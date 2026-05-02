"""MedMNIST dataset loading and splitting.

Provides:
  - PyTorch DataLoaders for training the autoencoder (images only)
  - Numpy arrays for PCA / feature extraction / MLP (images + labels)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from configs.config import Config, set_global_seed


@dataclass(frozen=True)
class DatasetBundle:
    """Container for splits in both torch and numpy formats."""

    loaders: Dict[str, DataLoader]
    arrays: Dict[str, Tuple[np.ndarray, np.ndarray]]  # split -> (X, y)


class _ImagesOnlyDataset(Dataset):
    """Dataset wrapper that returns only images (for AE training)."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:  # noqa: D401
        return len(self.base)

    def __getitem__(self, idx: int):
        x, _y = self.base[idx]
        return x


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _get_medmnist_class(dataset_name: str):
    name = dataset_name.lower().strip()
    try:
        import medmnist  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Failed to import medmnist. Install with: pip install medmnist") from e

    mapping = {
        "dermamnist": getattr(medmnist, "DermaMNIST"),
        "pathmnist": getattr(medmnist, "PathMNIST"),
        "retinamnist": getattr(medmnist, "RetinaMNIST"),
        "pneumoniamnist": getattr(medmnist, "PneumoniaMNIST"),
    }
    if name not in mapping:
        raise ValueError(
            f"Unknown dataset_name='{dataset_name}'. Expected one of: {sorted(mapping.keys())}"
        )
    return mapping[name]


def load_medmnist(
    config: Config,
    dataset_name: str | None = None,
    dry_run: bool | None = None,
    num_workers: int = 2,
) -> DatasetBundle:
    """Load a MedMNIST dataset at the configured resolution.

    Args:
        config: experiment configuration
        dataset_name: one of {'dermamnist','pathmnist','retinamnist','pneumoniamnist'}.
            If None, uses config.DATASET_NAME.
        dry_run: if True, subsample to 200/50/50 (train/val/test)
        num_workers: DataLoader workers

    Returns:
        DatasetBundle with:
          - loaders: {'train','val','test'} images-only DataLoaders (Tensor[B,C,H,W])
          - arrays: {'train','val','test'} -> (X_img, y) with X_img float32 [0,1] shaped (N,C,H,W)
    """

    set_global_seed(config.SEED)
    use_dry = config.DRY_RUN if dry_run is None else bool(dry_run)
    ds_name = config.DATASET_NAME if dataset_name is None else str(dataset_name)
    ds_cls = _get_medmnist_class(ds_name)

    root = os.path.join(_project_root(), "data_cache")
    os.makedirs(root, exist_ok=True)

    def _to_float_tensor(img: np.ndarray) -> torch.Tensor:
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        # HWC -> CHW
        arr = arr.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    class _TransformWrapper(Dataset):
        def __init__(self, base_ds: Dataset):
            self.base_ds = base_ds

        def __len__(self) -> int:
            return len(self.base_ds)

        def __getitem__(self, idx: int):
            img, label = self.base_ds[idx]
            x = _to_float_tensor(img)
            y = int(np.asarray(label).reshape(-1)[0])
            return x, y

    train_ds = _TransformWrapper(ds_cls(split="train", root=root, download=True, size=config.DATASET_SIZE))
    val_ds = _TransformWrapper(ds_cls(split="val", root=root, download=True, size=config.DATASET_SIZE))
    test_ds = _TransformWrapper(ds_cls(split="test", root=root, download=True, size=config.DATASET_SIZE))

    if use_dry:
        train_ds = torch.utils.data.Subset(train_ds, list(range(min(200, len(train_ds)))))
        val_ds = torch.utils.data.Subset(val_ds, list(range(min(50, len(val_ds)))))
        test_ds = torch.utils.data.Subset(test_ds, list(range(min(50, len(test_ds)))))

    # AE loaders (images only)
    loaders = {
        "train": DataLoader(
            _ImagesOnlyDataset(train_ds),
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(config.DEVICE == "cuda"),
        ),
        "val": DataLoader(
            _ImagesOnlyDataset(val_ds),
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(config.DEVICE == "cuda"),
        ),
        "test": DataLoader(
            _ImagesOnlyDataset(test_ds),
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(config.DEVICE == "cuda"),
        ),
    }

    def _to_numpy_xy(ds: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
        xs, ys = [], []
        for xb, yb in loader:
            xs.append(xb.numpy())
            ys.append(yb.numpy())
        x = np.concatenate(xs, axis=0).astype(np.float32)
        y = np.concatenate(ys, axis=0).astype(int).reshape(-1)
        return x, y

    x_train, y_train = _to_numpy_xy(train_ds)
    x_val, y_val = _to_numpy_xy(val_ds)
    x_test, y_test = _to_numpy_xy(test_ds)

    arrays = {"train": (x_train, y_train), "val": (x_val, y_val), "test": (x_test, y_test)}
    return DatasetBundle(loaders=loaders, arrays=arrays)


def load_retinamnist(
    config: Config,
    dry_run: bool | None = None,
    num_workers: int = 2,
) -> DatasetBundle:
    """Backwards-compatible RetinaMNIST loader.

    Prefer using `load_medmnist(..., dataset_name='retinamnist')` going forward.
    """
    return load_medmnist(config, dataset_name="retinamnist", dry_run=dry_run, num_workers=num_workers)


def make_mlp_loader(
    features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool = True
) -> DataLoader:
    """Create DataLoader for MLP training from numpy features."""

    x = torch.from_numpy(np.asarray(features, dtype=np.float32))
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    cfg = Config(DRY_RUN=True)
    bundle = load_medmnist(cfg, dataset_name=cfg.DATASET_NAME, dry_run=True, num_workers=0)
    for split in ["train", "val", "test"]:
        x, y = bundle.arrays[split]
        print(split, x.shape, y.shape, "min/max", x.min(), x.max(), "classes", np.unique(y))
    xb = next(iter(bundle.loaders["train"]))
    print("AE batch:", xb.shape, xb.min().item(), xb.max().item())

