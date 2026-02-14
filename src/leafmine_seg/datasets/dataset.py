"""PyTorch Dataset for leafmine segmentation."""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from leafmine_seg.datasets.tiling import random_crop


class LeafmineDataset(Dataset):
    """Dataset that loads processed image-mask pairs and returns random crops.

    Images are grayscale and converted to 3-channel for the encoder.
    Masks are binary (0 or 255) and returned as float [0, 1].
    """

    def __init__(
        self,
        image_paths: list[Path],
        mask_dir: Path,
        patch_size: int = 1024,
        transform: A.Compose | None = None,
        crops_per_image: int = 4,
    ) -> None:
        self.image_paths = image_paths
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.transform = transform
        self.crops_per_image = crops_per_image
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.image_paths) * self.crops_per_image

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_idx = idx // self.crops_per_image
        img_path = self.image_paths[img_idx]
        mask_path = self.mask_dir / img_path.name

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")

        # Random crop
        image, mask = random_crop(image, mask, self.patch_size, rng=self.rng)

        # Grayscale -> 3-channel
        image = np.stack([image] * 3, axis=-1)  # (H, W, 3)

        # Binarize mask
        mask = (mask > 127).astype(np.uint8) * 255

        # Apply augmentation
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image_t = augmented["image"]  # (3, H, W) float
            mask_t = augmented["mask"]  # (H, W) float
        else:
            image_t = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255.0)
            mask_t = torch.from_numpy(mask.astype(np.float32) / 255.0)

        # Ensure mask is (1, H, W)
        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0)

        return {"image": image_t, "mask": mask_t}
