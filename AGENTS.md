# Leafmine Segmentation: Initial Architecture Plan

## Goal
- Build a U-Net based segmentation pipeline to infer `mine-only` masks from scanned leaf images.
- Keep this phase limited to architecture, directory structure, and dependency setup.

## Scope (v0)
- Input images are grayscale scanner images with white background.
- Annotation format is PNG mask.
- Segmentation target is mine area only (binary mask).
- Train/validation split is `8:2`.
- Training patch size is `1024`.
- Preprocessing is mandatory for both training and inference.
- Inference output is PNG mask restored to the original image size.

## Proposed Directory Structure
```text
leafmine-segmentation/
  configs/
    preprocess.yaml
    train.yaml
    infer.yaml
  data/
    raw/images
    raw/masks
    processed/images
    processed/masks
    splits/
  src/leafmine_seg/
    preprocessing/
      leaf_extractor.py
    datasets/
      dataset.py
      tiling.py
      transforms.py
    models/
      factory.py
      losses.py
      metrics.py
    training/
      trainer.py
      validate.py
    inference/
      predictor.py
      stitch.py
    io/
      png_writer.py
  scripts/
    prepare_data.py
    train.py
    predict.py
```

## Preprocessing Policy
- Use a fixed threshold to remove white background.
- Extract leaf region with connected-component + bounding box flow.
- Use the same preprocessing path for training and inference.
- Keep threshold configurable so it can be revised if mine-like white regions are over-removed.

## Modeling Policy
- Use `segmentation_models.pytorch` for U-Net construction.
- Convert grayscale input to 3-channel before model input (`gray -> repeat(3)`).
- Use binary mask target (`0/255`) consistently.

## Output Policy
- Save inference mask as PNG.
- Reconstruct final mask to original image size after tile prediction and stitching.

## Docker Policy (Planned)
- GPU-enabled container (CUDA runtime base image).
- Run with `gpus: all` via Docker Compose.
- Mount `data/` as volume and keep datasets outside image layers.

## Current Status
- Architecture and dependency baseline are defined.
- Training/inference implementation has not started yet.
