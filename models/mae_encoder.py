"""Pretrained ViT MAE encoder feature extractor via timm."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch
from torchvision import transforms

from configs.config import Config, set_global_seed


def build_mae_feature_extractor(config: Config) -> Callable[[np.ndarray], np.ndarray]:
    """Create a frozen MAE/ViT encoder feature extractor.

    Args:
        config: Experiment config containing MAE_MODEL_NAME and DEVICE.

    Returns:
        extract_mae_features(images_numpy) -> numpy array (N, 768) for ViT-Base.

    Raises:
        RuntimeError: if timm model or pretrained weights fail to load.
    """

    set_global_seed(config.SEED)
    try:
        import timm
    except Exception as e:  # pragma: no cover
        print(
            "[MAE] Error: failed to import timm. Install with: pip install timm\n"
            f"Original error: {e}"
        )
        raise

    try:
        model = timm.create_model(config.MAE_MODEL_NAME, pretrained=True, num_classes=0)
    except Exception as e:  # pragma: no cover
        print(
            f"[MAE] Error: failed to load pretrained model '{config.MAE_MODEL_NAME}'.\n"
            "Make sure timm is installed and the model supports pretrained weights.\n"
            f"Original error: {e}"
        )
        raise RuntimeError("Failed to load pretrained MAE encoder.") from e

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(config.DEVICE)

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
        ]
    )

    @torch.no_grad()
    def extract_mae_features(images_numpy: np.ndarray) -> np.ndarray:
        """Extract flattened CLS-token features from images.

        Accepts images shaped:
          - (N, 1, 28, 28) float32 in [0,1]
          - (N, 28, 28) float32 in [0,1]
        Returns:
          - (N, D) float32, where D is typically 768 for ViT-Base
        """

        x = np.asarray(images_numpy, dtype=np.float32)
        if x.ndim == 3:
            x = x[:, None, :, :]
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("Expected images of shape (N,1,H,W) or (N,H,W).")

        xt = torch.from_numpy(x)
        # repeat grayscale -> 3ch
        xt = xt.repeat(1, 3, 1, 1)

        # Resize to 224x224
        # torchvision transforms expect PIL or torch with CHW; for batch we apply interpolate
        xt = torch.nn.functional.interpolate(xt, size=(224, 224), mode="bilinear", align_corners=False)

        xt = xt.to(config.DEVICE)

        # timm ViT models expose forward_features; returns tokens (B, num_tokens, D)
        if hasattr(model, "forward_features"):
            feats = model.forward_features(xt)
        else:  # pragma: no cover
            raise RuntimeError("Loaded model does not expose forward_features().")

        # feats could be (B, D) or (B, T, D)
        if feats.ndim == 3:
            cls = feats[:, 0, :]
        elif feats.ndim == 2:
            cls = feats
        else:  # pragma: no cover
            raise RuntimeError(f"Unexpected features shape from model: {feats.shape}")

        return cls.detach().cpu().numpy().astype(np.float32)

    # Quick smoke check to ensure feature dim is consistent
    _x = np.zeros((2, 1, 28, 28), dtype=np.float32)
    _f = extract_mae_features(_x)
    if _f.ndim != 2:
        raise RuntimeError(f"MAE features should be 2D, got shape {_f.shape}")
    return extract_mae_features


if __name__ == "__main__":
    cfg = Config(DRY_RUN=True)
    t0 = time.time()
    extractor = build_mae_feature_extractor(cfg)
    feats = extractor(np.random.rand(4, 1, 28, 28).astype(np.float32))
    print("Features:", feats.shape, "time:", time.time() - t0)

