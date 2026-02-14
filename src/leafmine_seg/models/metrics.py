"""Evaluation metrics for binary segmentation."""

from __future__ import annotations

import torch


def iou_score(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6
) -> torch.Tensor:
    """Compute Intersection over Union (IoU / Jaccard index).

    Args:
        logits: Raw model output (B, 1, H, W).
        targets: Binary targets (B, 1, H, W) in [0, 1].
        threshold: Threshold to binarize predictions.
        smooth: Smoothing factor to avoid division by zero.
    """
    preds = (torch.sigmoid(logits) > threshold).float()
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum() - intersection

    return (intersection + smooth) / (union + smooth)


def dice_score(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6
) -> torch.Tensor:
    """Compute Dice coefficient.

    Args:
        logits: Raw model output (B, 1, H, W).
        targets: Binary targets (B, 1, H, W) in [0, 1].
        threshold: Threshold to binarize predictions.
        smooth: Smoothing factor to avoid division by zero.
    """
    preds = (torch.sigmoid(logits) > threshold).float()
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum()

    return (2.0 * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)
