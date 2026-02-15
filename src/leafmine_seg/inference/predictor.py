"""Tile-based inference predictor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

from leafmine_seg.datasets.tiling import (
    TileInfo,
    compute_grid_tiles,
    extract_tile,
    pad_to_min_size,
)
from leafmine_seg.inference.stitch import stitch_tiles
from leafmine_seg.models.factory import create_model
from leafmine_seg.preprocessing.leaf_extractor import BBox, extract_leaf_bboxes, crop_region


class Predictor:
    """Run tile-based inference on full-size images."""

    def __init__(self, cfg: dict[str, Any], device: torch.device | None = None) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        infer_cfg = cfg.get("inference", {})
        self.patch_size = infer_cfg.get("patch_size", 1024)
        self.overlap_ratio = infer_cfg.get("overlap_ratio", 0.25)
        self.batch_size = infer_cfg.get("batch_size", 4)
        self.threshold = infer_cfg.get("threshold", 0.5)
        self.amp = infer_cfg.get("amp", True)

        preprocess_cfg = cfg.get("preprocess", {})
        self.bg_threshold = preprocess_cfg.get("bg_threshold", 230)
        self.min_area = preprocess_cfg.get("min_area", 10000)
        self.max_aspect_ratio = preprocess_cfg.get("max_aspect_ratio", 5.0)
        self.max_area_ratio = preprocess_cfg.get("max_area_ratio", 0.8)

        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        model = create_model(self.cfg.get("model", {}))
        ckpt_path = self.cfg.get("checkpoint_path", "checkpoints/best_model.pth")
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def predict_image(self, image_path: Path) -> np.ndarray:
        """Predict mine mask for a full scanner image.

        Returns:
            Binary mask (H, W) uint8 at original image size.
        """
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        orig_h, orig_w = image.shape[:2]

        # Extract leaf regions
        bboxes = extract_leaf_bboxes(
            image,
            bg_threshold=self.bg_threshold,
            min_area=self.min_area,
            max_aspect_ratio=self.max_aspect_ratio,
            max_area_ratio=self.max_area_ratio,
        )

        # Full-size output mask
        full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        for bbox in bboxes:
            leaf_crop = crop_region(image, bbox)
            leaf_mask = self._predict_crop(leaf_crop)
            # Place back into full mask
            full_mask[bbox.y : bbox.y + bbox.h, bbox.x : bbox.x + bbox.w] = leaf_mask[
                : bbox.h, : bbox.w
            ]

        return full_mask

    def _predict_crop(self, image: np.ndarray) -> np.ndarray:
        """Predict mask for a single cropped leaf region."""
        h, w = image.shape[:2]
        padded = pad_to_min_size(image, self.patch_size, self.patch_size)
        pad_h, pad_w = padded.shape[:2]

        tiles = compute_grid_tiles(pad_h, pad_w, self.patch_size, self.overlap_ratio)
        tile_preds: list[np.ndarray] = []
        tile_infos: list[TileInfo] = []

        # Batch inference
        batch_tiles: list[np.ndarray] = []
        batch_infos: list[TileInfo] = []

        for tile_info in tiles:
            tile = extract_tile(padded, tile_info)
            # Grayscale -> 3ch -> ImageNet normalize -> (C, H, W)
            tile_3ch = np.stack([tile] * 3, axis=-1).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            tile_3ch = (tile_3ch - mean) / std
            tile_3ch = tile_3ch.transpose(2, 0, 1)  # (3, H, W)
            batch_tiles.append(tile_3ch)
            batch_infos.append(tile_info)

            if len(batch_tiles) == self.batch_size:
                preds = self._run_batch(batch_tiles)
                tile_preds.extend(preds)
                tile_infos.extend(batch_infos)
                batch_tiles = []
                batch_infos = []

        # Remaining tiles
        if batch_tiles:
            preds = self._run_batch(batch_tiles)
            tile_preds.extend(preds)
            tile_infos.extend(batch_infos)

        # Stitch
        stitched = stitch_tiles(tile_preds, tile_infos, pad_h, pad_w, self.threshold)

        # Crop back to original size
        return stitched[:h, :w]

    @torch.no_grad()
    def _run_batch(self, tiles: list[np.ndarray]) -> list[np.ndarray]:
        """Run model inference on a batch of tiles."""
        batch = torch.from_numpy(np.stack(tiles)).to(self.device)

        with torch.autocast(device_type=self.device.type, enabled=self.amp):
            logits = self.model(batch)

        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()  # (B, H, W)
        return [probs[i] for i in range(probs.shape[0])]
