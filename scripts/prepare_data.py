"""CLI script: preprocess raw images and generate train/val splits."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split

from leafmine_seg.preprocessing.leaf_extractor import process_image_pair

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess leaf images and create splits.")
    parser.add_argument(
        "--config", type=str, default="configs/preprocess.yaml", help="Path to preprocess config.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    raw_image_dir = Path(cfg["raw_image_dir"])
    raw_mask_dir = Path(cfg["raw_mask_dir"])
    output_image_dir = Path(cfg["output_image_dir"])
    output_mask_dir = Path(cfg["output_mask_dir"])
    split_dir = Path(cfg["split_dir"])

    bg_threshold = cfg.get("bg_threshold", 230)
    min_area = cfg.get("min_area", 10000)
    val_ratio = cfg.get("val_ratio", 0.2)
    random_seed = cfg.get("random_seed", 42)

    # Collect image files
    image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    image_files = sorted(
        p for p in raw_image_dir.iterdir() if p.suffix.lower() in image_extensions
    )

    if not image_files:
        logger.error(f"No images found in {raw_image_dir}")
        return

    logger.info(f"Found {len(image_files)} images in {raw_image_dir}")

    # Process each image-mask pair
    all_outputs: list[str] = []
    for img_path in image_files:
        # Find masks: look for {stem}_1.ext, {stem}_2.ext, ... pattern
        mask_paths: list[Path] = []
        for suffix in image_extensions:
            mask_paths.extend(
                sorted(raw_mask_dir.glob(f"{img_path.stem}_*{suffix}"))
            )

        # Also check for exact name match (single mask case)
        if not mask_paths:
            for suffix in image_extensions:
                candidate = raw_mask_dir / (img_path.stem + suffix)
                if candidate.exists():
                    mask_paths = [candidate]
                    break

        if not mask_paths:
            logger.warning(f"No mask found for {img_path.name}, skipping.")
            continue

        logger.info(f"Found {len(mask_paths)} mask(s) for {img_path.name}")

        try:
            saved = process_image_pair(
                img_path, mask_paths, output_image_dir, output_mask_dir,
                bg_threshold=bg_threshold, min_area=min_area,
            )
            all_outputs.extend(p.name for p in saved)
            logger.info(f"Processed {img_path.name} -> {len(saved)} region(s)")
        except Exception as e:
            logger.error(f"Failed to process {img_path.name}: {e}")

    if not all_outputs:
        logger.error("No processed images generated.")
        return

    # Train/val split
    train_names, val_names = train_test_split(
        all_outputs, test_size=val_ratio, random_state=random_seed,
    )

    split_dir.mkdir(parents=True, exist_ok=True)
    with open(split_dir / "train.json", "w") as f:
        json.dump(train_names, f, indent=2)
    with open(split_dir / "val.json", "w") as f:
        json.dump(val_names, f, indent=2)

    logger.info(f"Split: {len(train_names)} train, {len(val_names)} val")
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
