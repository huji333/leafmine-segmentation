"""CLI script: run inference on images."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from leafmine_seg.inference.predictor import Predictor
from leafmine_seg.io.png_writer import save_mask

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leafmine segmentation inference.")
    parser.add_argument(
        "--config", type=str, default="configs/infer.yaml", help="Path to inference config.",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Override input directory or single image path.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    input_path = Path(args.input) if args.input else Path(cfg["input_dir"])
    output_dir = Path(cfg["output_dir"])

    predictor = Predictor(cfg)

    # Collect images
    image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    if input_path.is_file():
        image_files = [input_path]
    else:
        image_files = sorted(
            p for p in input_path.iterdir() if p.suffix.lower() in image_extensions
        )

    if not image_files:
        logger.error(f"No images found at {input_path}")
        return

    logger.info(f"Processing {len(image_files)} image(s)")

    for img_path in image_files:
        logger.info(f"Predicting: {img_path.name}")
        mask = predictor.predict_image(img_path)
        out_path = output_dir / f"{img_path.stem}_mask.png"
        save_mask(mask, out_path)
        logger.info(f"Saved: {out_path}")

    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
