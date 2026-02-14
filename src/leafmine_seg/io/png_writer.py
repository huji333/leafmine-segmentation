"""Save masks as PNG files."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    """Save a binary mask as a PNG file.

    Args:
        mask: Binary mask (H, W) uint8 with values 0 or 255.
        output_path: Path to save the PNG file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mask)
