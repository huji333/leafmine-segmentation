"""Extract leaf regions from scanned images by removing white background."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int


def extract_leaf_bboxes(
    image: np.ndarray,
    bg_threshold: int = 230,
    min_area: int = 10000,
) -> list[BBox]:
    """Find bounding boxes of leaf regions in a grayscale scanner image.

    Args:
        image: Grayscale image (H, W) uint8.
        bg_threshold: Pixels above this value are treated as background.
        min_area: Minimum connected-component area to keep.

    Returns:
        List of BBox for each detected leaf region.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Threshold: foreground = pixels darker than bg_threshold
    _, binary = cv2.threshold(gray, bg_threshold, 255, cv2.THRESH_BINARY_INV)

    # Connected-component analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    bboxes: list[BBox] = []
    for i in range(1, num_labels):  # skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        bboxes.append(BBox(x=x, y=y, w=w, h=h))

    return bboxes


def crop_region(image: np.ndarray, bbox: BBox) -> np.ndarray:
    """Crop a region from an image using a BBox."""
    return image[bbox.y : bbox.y + bbox.h, bbox.x : bbox.x + bbox.w].copy()


def process_image_pair(
    image_path: Path,
    mask_path: Path,
    output_image_dir: Path,
    output_mask_dir: Path,
    bg_threshold: int = 230,
    min_area: int = 10000,
) -> list[Path]:
    """Process a single image-mask pair: extract leaf regions and save crops.

    Returns:
        List of output image paths that were saved.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")

    bboxes = extract_leaf_bboxes(image, bg_threshold=bg_threshold, min_area=min_area)

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    stem = image_path.stem

    for idx, bbox in enumerate(bboxes):
        suffix = f"_{idx}" if len(bboxes) > 1 else ""
        out_name = f"{stem}{suffix}.png"

        img_crop = crop_region(image, bbox)
        msk_crop = crop_region(mask, bbox)

        out_img = output_image_dir / out_name
        out_msk = output_mask_dir / out_name

        cv2.imwrite(str(out_img), img_crop)
        cv2.imwrite(str(out_msk), msk_crop)
        saved.append(out_img)

    return saved
