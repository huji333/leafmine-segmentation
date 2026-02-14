"""CLI script: train the segmentation model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from leafmine_seg.datasets.dataset import LeafmineDataset
from leafmine_seg.datasets.transforms import build_train_transform, build_val_transform
from leafmine_seg.models.factory import create_model
from leafmine_seg.models.losses import DiceBCELoss
from leafmine_seg.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train leafmine segmentation model.")
    parser.add_argument(
        "--config", type=str, default="configs/train.yaml", help="Path to train config.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    data_cfg = cfg.get("data", {})
    image_dir = Path(data_cfg["image_dir"])
    mask_dir = Path(data_cfg["mask_dir"])
    split_dir = Path(data_cfg["split_dir"])
    patch_size = data_cfg.get("patch_size", 1024)
    num_workers = data_cfg.get("num_workers", 4)

    # Load splits
    with open(split_dir / "train.json") as f:
        train_names = json.load(f)
    with open(split_dir / "val.json") as f:
        val_names = json.load(f)

    train_paths = [image_dir / name for name in train_names]
    val_paths = [image_dir / name for name in val_names]

    logger.info(f"Train: {len(train_paths)} images, Val: {len(val_paths)} images")

    # Transforms
    aug_cfg = cfg.get("augmentation", {})
    train_transform = build_train_transform(aug_cfg)
    val_transform = build_val_transform()

    # Datasets
    train_dataset = LeafmineDataset(
        image_paths=train_paths, mask_dir=mask_dir,
        patch_size=patch_size, transform=train_transform,
    )
    val_dataset = LeafmineDataset(
        image_paths=val_paths, mask_dir=mask_dir,
        patch_size=patch_size, transform=val_transform,
    )

    train_cfg = cfg.get("training", {})
    batch_size = train_cfg.get("batch_size", 4)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # Model
    model = create_model(cfg.get("model", {}))
    logger.info(f"Model: U-Net with {cfg.get('model', {}).get('encoder', 'resnet34')} encoder")

    # Loss
    loss_cfg = cfg.get("loss", {})
    criterion = DiceBCELoss(
        dice_weight=loss_cfg.get("dice_weight", 1.0),
        bce_weight=loss_cfg.get("bce_weight", 1.0),
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    # Train
    trainer = Trainer(
        model=model, criterion=criterion, optimizer=optimizer,
        train_loader=train_loader, val_loader=val_loader,
        cfg=cfg, device=device,
    )
    trainer.train()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
