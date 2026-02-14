"""Model factory using segmentation_models.pytorch."""

from __future__ import annotations

from typing import Any

import segmentation_models_pytorch as smp
import torch.nn as nn


def create_model(cfg: dict[str, Any]) -> nn.Module:
    """Create a U-Net model from config.

    Args:
        cfg: model section of train.yaml / infer.yaml.

    Returns:
        U-Net model instance.
    """
    model = smp.Unet(
        encoder_name=cfg.get("encoder", "resnet34"),
        encoder_weights=cfg.get("encoder_weights", "imagenet"),
        in_channels=cfg.get("in_channels", 3),
        classes=cfg.get("classes", 1),
    )
    return model
