"""Stitch overlapping tile predictions back into a full-size mask."""

from __future__ import annotations

import numpy as np

from leafmine_seg.datasets.tiling import TileInfo


def stitch_tiles(
    tile_preds: list[np.ndarray],
    tile_infos: list[TileInfo],
    image_h: int,
    image_w: int,
    threshold: float = 0.5,
) -> np.ndarray:
    """Stitch tile predictions into a full-size binary mask.

    Overlapping regions are averaged before thresholding.

    Args:
        tile_preds: List of probability maps (H, W) float in [0, 1].
        tile_infos: Corresponding tile positions.
        image_h: Height of the original (padded) image.
        image_w: Width of the original (padded) image.
        threshold: Threshold for binarization after averaging.

    Returns:
        Binary mask (image_h, image_w) uint8 with values 0 or 255.
    """
    accum = np.zeros((image_h, image_w), dtype=np.float64)
    count = np.zeros((image_h, image_w), dtype=np.float64)

    for pred, info in zip(tile_preds, tile_infos):
        accum[info.y : info.y + info.h, info.x : info.x + info.w] += pred
        count[info.y : info.y + info.h, info.x : info.x + info.w] += 1.0

    # Average overlapping regions
    avg = np.divide(accum, count, out=np.zeros_like(accum), where=count > 0)

    # Binarize
    mask = (avg > threshold).astype(np.uint8) * 255
    return mask
