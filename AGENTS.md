# Leafmine Segmentation

## Goal
- Build a U-Net based segmentation pipeline to infer `mine-only` masks from scanned leaf images.

## Scope (v0)
- Input images are grayscale scanner images with white background.
- Annotation format is PNG mask. Multiple masks per image are supported (`{stem}_1.png`, `{stem}_2.png`, ...) and merged via logical OR.
- Segmentation target is mine area only (binary mask).
- Train/validation split is `8:2`.
- Training patch size is `1024`.
- Preprocessing is mandatory for both training and inference.
- Inference output is PNG mask restored to the original image size.

## Directory Structure
```text
leafmine-segmentation/
  configs/
    preprocess.yaml       # Background threshold, area/aspect filters, split ratio
    train.yaml            # Batch size, epochs, lr, augmentation, etc.
    infer.yaml            # Checkpoint, overlap ratio, threshold
  data/
    raw/images/           # Original scanner images
    raw/masks/            # Annotation masks ({stem}_1.png, {stem}_2.png, ...)
    processed/images/     # Cropped leaf images
    processed/masks/      # Cropped leaf masks
    splits/               # train.json / val.json
  src/leafmine_seg/
    preprocessing/
      leaf_extractor.py   # White-bg removal, connected-component bbox crop + artifact filters
    datasets/
      dataset.py          # PyTorch Dataset (random crop, gray->3ch)
      tiling.py           # Random crop (train) / grid tiling (infer)
      transforms.py       # Albumentations augmentation pipelines
    models/
      factory.py          # smp.Unet(resnet34, imagenet)
      losses.py           # DiceBCELoss
      metrics.py          # IoU, Dice
    training/
      trainer.py          # Training loop (AMP, checkpoint, TensorBoard/CSV)
      validate.py         # Validation loop
    inference/
      predictor.py        # Tile-based batch inference
      stitch.py           # Overlap averaging -> binary mask
    io/
      png_writer.png      # Save mask as PNG
  scripts/
    prepare_data.py       # CLI: preprocess + split generation
    train.py              # CLI: model training
    predict.py            # CLI: inference
```

## Preprocessing Policy
- Use a fixed threshold (`bg_threshold`) to remove white background. Default is `200`.
- Extract leaf regions via connected-component analysis + bounding box crop.
- Filter out scanner artifacts with two additional guards:
  - `max_aspect_ratio` (default 5.0): rejects elongated edge strips.
  - `max_area_ratio` (default 0.8): rejects components whose bbox spans most of the image (scanner border frames).
- If multiple valid regions are found, each is saved as a separate crop (`{stem}_0.png`, `{stem}_1.png`, ...).
- Use the same preprocessing parameters for training and inference (both read from config).
- Keep thresholds configurable in `configs/preprocess.yaml` — edge cases that slip through are handled manually.

## Modeling Policy
- Use `segmentation_models.pytorch` for U-Net construction.
- Encoder: ResNet34 (ImageNet pretrained).
- Convert grayscale input to 3-channel before model input (`gray -> repeat(3)`).
- Normalize with ImageNet statistics (mean/std).
- Loss: DiceLoss + BCEWithLogitsLoss (weighted sum).
- Use binary mask target (`0/255`) consistently.

## Training Policy
- Random 1024x1024 crop per image (padded if smaller).
- Augmentation: HorizontalFlip, VerticalFlip, RandomRotate90, RandomBrightnessContrast, GaussNoise.
- AMP (mixed precision) enabled by default.
- Best model checkpoint saved by validation Dice score.

## Inference Policy
- Grid tiling with configurable overlap (default 25%).
- Overlap regions averaged before thresholding.
- Reconstruct final mask to original image size after stitching.
- Save inference mask as PNG.

## Docker Policy
- Base: `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` + Python 3.9 (deadsnakes).
- Package management via `uv`.
- Run with `gpus: all` via Docker Compose.
- Mount `data/`, `checkpoints/`, `outputs/`, `runs/` as volumes.
- Makefile targets: `build`, `preprocess`, `remove_processed`, `reprocess`, `train`, `predict`, `shell`.

## Current Status
- Full pipeline implemented: preprocess -> train -> predict.
- Docker / Makefile environment ready.
- Multiple masks per image supported (merged via OR).
- Scanner artifact filtering implemented (aspect ratio + image coverage ratio).
