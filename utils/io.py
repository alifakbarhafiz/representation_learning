"""I/O helpers for experiment outputs."""

from __future__ import annotations

import json
import os
import sys
from typing import Any


def save_json(path: str, payload: dict[str, Any]) -> None:
    """Save a JSON file with pretty formatting."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def collect_environment() -> dict[str, Any]:
    """Collect runtime environment info for experiment JSON artifacts."""

    py_ver = str(sys.version).splitlines()[0]
    try:
        import torch

        torch_ver = str(torch.__version__)
        cuda_ver = str(torch.version.cuda) if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover
        torch_ver = "unknown"
        cuda_ver = "unknown"

    try:
        import timm  # type: ignore

        timm_ver: str | None = str(getattr(timm, "__version__", None))
    except Exception:
        timm_ver = None

    try:
        import medmnist  # type: ignore

        medmnist_ver: str | None = str(getattr(medmnist, "__version__", None))
    except Exception:
        medmnist_ver = None

    return {
        "python_version": py_ver,
        "torch_version": torch_ver,
        "timm_version": timm_ver,
        "medmnist_version": medmnist_ver,
        "cuda_version": cuda_ver,
    }


def _str_key_dict(d: dict[Any, Any]) -> dict[str, Any]:
    return {str(k): v for k, v in d.items()}


def _str_key_nested(d: dict[str, Any]) -> dict[str, Any]:
    """Ensure nested dicts that represent class-indexed maps use string keys."""

    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[str(k)] = _str_key_dict(v)
        else:
            out[str(k)] = v
    return out


def build_results_payload(
    *,
    dataset: str,
    dataset_size: int,
    dry_run: bool,
    seed: int,
    config: Any,
    dataset_info: dict[str, Any] | None,
    class_weights: dict[str, float] | None,
    metrics: dict[str, Any],
    training_curves: dict[str, Any],
    times: dict[str, Any],
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the comprehensive experiment JSON payload (target schema)."""

    cfg = config.to_dict() if hasattr(config, "to_dict") else dict(config)

    payload = {
        "dataset": str(dataset),
        "dataset_size": int(dataset_size),
        "dry_run": bool(dry_run),
        "seed": int(seed),
        "config": {
            "pca_n_components": int(cfg["PCA_N_COMPONENTS"]),
            "ae_latent_dim": int(cfg["AE_LATENT_DIM"]),
            "ae_epochs": int(cfg["AE_EPOCHS"]),
            "ae_lr": float(cfg["AE_LR"]),
            "mlp_hidden_dims": list(cfg["MLP_HIDDEN_DIMS"]),
            "mlp_epochs": int(cfg["MLP_EPOCHS"]),
            "mlp_lr": float(cfg["MLP_LR"]),
            "mlp_dropout": float(cfg.get("MLP_DROPOUT", 0.3)),
            "batch_size": int(cfg["BATCH_SIZE"]),
            "mae_model_names": list(cfg["MAE_MODEL_NAMES"]),
            "device": str(cfg["DEVICE"]),
        },
        "dataset_info": dataset_info,
        "class_weights": class_weights,
        "metrics": metrics,
        "training_curves": training_curves,
        "times": times,
        "environment": collect_environment() if environment is None else environment,
    }

    # Normalize class-indexed dict keys to strings for consistency
    if isinstance(payload.get("class_weights"), dict):
        payload["class_weights"] = _str_key_dict(payload["class_weights"])

    # dataset_info distributions
    di = payload.get("dataset_info")
    if isinstance(di, dict):
        cd = di.get("class_distribution")
        if isinstance(cd, dict):
            fixed = {}
            for split, dist in cd.items():
                if isinstance(dist, dict):
                    fixed[str(split)] = _str_key_dict(dist)
                else:
                    fixed[str(split)] = dist
            di["class_distribution"] = fixed

    # metrics: method_key -> split -> per_class_acc
    m = payload.get("metrics")
    if isinstance(m, dict):
        for method_key, per_method in m.items():
            if not isinstance(per_method, dict):
                continue
            for split_key, split_metrics in per_method.items():
                if not isinstance(split_metrics, dict):
                    continue
                pca = split_metrics.get("per_class_acc")
                if isinstance(pca, dict):
                    split_metrics["per_class_acc"] = _str_key_dict(pca)

    return payload


def save_results(path: str, payload: dict[str, Any]) -> None:
    """Save comprehensive experiment JSON to disk."""

    save_json(path, payload)


if __name__ == "__main__":
    save_json("results/_io_sanity.json", {"ok": True})
    print("Wrote results/_io_sanity.json")

