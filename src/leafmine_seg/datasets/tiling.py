"""Tile extraction for training (random crop) and inference (grid with overlap)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TileInfo:
    """Metadata for a single tile within a larger image."""
    x: int
    y: int
    w: int
    h: int


def pad_to_min_size(image: np.ndarray, min_h: int, min_w: int, pad_value: int = 0) -> np.ndarray:
    """Pad image to at least (min_h, min_w) with constant value."""
    h, w = image.shape[:2]
    pad_h = max(0, min_h - h)
    pad_w = max(0, min_w - w)
    if pad_h == 0 and pad_w == 0:
        return image

    if image.ndim == 3:
        padding = ((0, pad_h), (0, pad_w), (0, 0))
    else:
        padding = ((0, pad_h), (0, pad_w))

    return np.pad(image, padding, mode="constant", constant_values=pad_value)


def random_crop(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a random patch_size x patch_size crop from image and mask.

    If the image is smaller than patch_size, it will be padded first.
    """
    if rng is None:
        rng = np.random.default_rng()

    image = pad_to_min_size(image, patch_size, patch_size)
    mask = pad_to_min_size(mask, patch_size, patch_size)

    h, w = image.shape[:2]
    y = rng.integers(0, h - patch_size + 1)
    x = rng.integers(0, w - patch_size + 1)

    img_crop = image[y : y + patch_size, x : x + patch_size]
    msk_crop = mask[y : y + patch_size, x : x + patch_size]
    return img_crop, msk_crop


def compute_grid_tiles(
    image_h: int,
    image_w: int,
    patch_size: int,
    overlap_ratio: float = 0.25,
) -> list[TileInfo]:
    """Compute grid tile positions with overlap for inference.

    Args:
        image_h: Height of the (padded) image.
        image_w: Width of the (padded) image.
        patch_size: Size of each square tile.
        overlap_ratio: Fraction of overlap between adjacent tiles.

    Returns:
        List of TileInfo describing each tile's position.
    """
    stride = int(patch_size * (1 - overlap_ratio))
    stride = max(stride, 1)

    tiles: list[TileInfo] = []
    y = 0
    while y < image_h:
        x = 0
        while x < image_w:
            # Clamp tile to image boundary
            actual_y = min(y, max(0, image_h - patch_size))
            actual_x = min(x, max(0, image_w - patch_size))
            tiles.append(TileInfo(x=actual_x, y=actual_y, w=patch_size, h=patch_size))
            x += stride
        y += stride

    return tiles


def extract_tile(image: np.ndarray, tile: TileInfo) -> np.ndarray:
    """Extract a tile from an image."""
    return image[tile.y : tile.y + tile.h, tile.x : tile.x + tile.w].copy()
