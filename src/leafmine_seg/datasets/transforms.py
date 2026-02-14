"""Albumentations-based augmentation pipelines."""

from __future__ import annotations

from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_train_transform(cfg: dict[str, Any]) -> A.Compose:
    """Build training augmentation pipeline from config.

    Args:
        cfg: augmentation section of train.yaml.
    """
    transforms: list[A.BasicTransform] = []

    if cfg.get("horizontal_flip", True):
        transforms.append(A.HorizontalFlip(p=0.5))

    if cfg.get("vertical_flip", True):
        transforms.append(A.VerticalFlip(p=0.5))

    if cfg.get("random_rotate_90", True):
        transforms.append(A.RandomRotate90(p=0.5))

    bc = cfg.get("brightness_contrast")
    if bc:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=bc.get("brightness_limit", 0.2),
                contrast_limit=bc.get("contrast_limit", 0.2),
                p=0.5,
            )
        )

    gn = cfg.get("gauss_noise")
    if gn:
        transforms.append(
            A.GaussNoise(
                var_limit=tuple(gn.get("var_limit", [10.0, 50.0])),
                p=0.3,
            )
        )

    # ImageNet normalization (matches pretrained encoder expectations)
    transforms.append(A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=255.0,
    ))
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def build_val_transform() -> A.Compose:
    """Build validation/inference transform (normalize + to tensor only)."""
    return A.Compose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
