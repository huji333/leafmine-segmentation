# Leafmine Segmentation

U-Net based binary segmentation pipeline for detecting mine regions in scanned leaf images.

## Setup

### Requirements
- Docker with NVIDIA GPU support, or
- Python 3.9+ with CUDA-capable GPU

### Docker (recommended)

```bash
make build
```

### Local

```bash
uv sync
```

## Usage

### 1. Prepare data

Place raw scanner images in `data/raw/images/` and corresponding PNG masks in `data/raw/masks/`.

Multiple masks per image are supported: name them `{stem}_1.png`, `{stem}_2.png`, etc.
They will be merged via logical OR before processing.

```bash
# Docker
make preprocess

# Local
uv run python scripts/prepare_data.py --config configs/preprocess.yaml
```

This will:
- Extract leaf regions from white background using connected-component analysis
- Filter out scanner artifacts (edge strips, border frames) automatically
- Save cropped images/masks to `data/processed/`
- Generate train/val splits in `data/splits/`

If multiple leaf regions are found in one image, they are saved as `{stem}_0.png`, `{stem}_1.png`, etc.

**Re-running preprocessing** (clears processed data and splits, then re-runs):

```bash
make reprocess
```

To only clear processed data without re-running:

```bash
make remove_processed
```

### 2. Train

```bash
# Docker
make train

# Local
uv run python scripts/train.py --config configs/train.yaml
```

Checkpoints are saved to `checkpoints/`. Training logs (TensorBoard + CSV) are saved to `runs/`.

### 3. Predict

```bash
# Docker
make predict

# Local
uv run python scripts/predict.py --config configs/infer.yaml
```

Output masks are saved to `outputs/`.

## Architecture

- **Encoder**: ResNet34 (ImageNet pretrained)
- **Decoder**: U-Net (via segmentation-models-pytorch)
- **Loss**: Dice + BCE
- **Input**: Grayscale scanner images (converted to 3-channel internally)
- **Output**: Binary PNG mask (mine regions = 255)

## Configuration

All settings are in YAML files under `configs/`:

| File | Purpose |
|------|---------|
| `preprocess.yaml` | Background threshold, area filters, aspect ratio filter, split ratio |
| `train.yaml` | Batch size, epochs, LR, augmentation, encoder |
| `infer.yaml` | Checkpoint path, overlap ratio, threshold |

Key preprocessing parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bg_threshold` | `200` | Pixels above this are treated as background |
| `min_area` | `10000` | Minimum component area to keep |
| `max_aspect_ratio` | `5.0` | Rejects scanner-edge strips |
| `max_area_ratio` | `0.8` | Rejects components spanning the whole image (scanner border frames) |

## Project Structure

```
src/leafmine_seg/
  preprocessing/    # Leaf region extraction
  datasets/         # Dataset, tiling, augmentation
  models/           # U-Net factory, losses, metrics
  training/         # Training & validation loops
  inference/        # Tile-based prediction & stitching
  io/               # PNG output
scripts/            # CLI entry points
configs/            # YAML configuration
```
