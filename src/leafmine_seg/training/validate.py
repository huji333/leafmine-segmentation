"""Validation loop."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from leafmine_seg.models.metrics import dice_score, iou_score


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool = True,
) -> dict[str, float]:
    """Run one validation epoch.

    Returns:
        Dictionary with val_loss, val_dice, val_iou.
    """
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, masks)

        total_loss += loss.item()
        total_dice += dice_score(logits, masks).item()
        total_iou += iou_score(logits, masks).item()
        n_batches += 1

    n_batches = max(n_batches, 1)
    return {
        "val_loss": total_loss / n_batches,
        "val_dice": total_dice / n_batches,
        "val_iou": total_iou / n_batches,
    }
