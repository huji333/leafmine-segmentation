"""Training loop with AMP, checkpointing, and logging."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from leafmine_seg.models.metrics import dice_score, iou_score
from leafmine_seg.training.validate import validate

logger = logging.getLogger(__name__)


class Trainer:
    """Standard PyTorch training loop with AMP support."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        train_cfg = cfg.get("training", {})
        self.epochs = train_cfg.get("epochs", 100)
        self.amp = train_cfg.get("amp", True)

        self.checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(cfg.get("log_dir", "runs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = torch.amp.GradScaler(enabled=self.amp)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.best_dice = 0.0

        # CSV log
        self.csv_path = self.log_dir / "metrics.csv"
        self._init_csv()

    def _init_csv(self) -> None:
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "train_dice", "train_iou",
                "val_loss", "val_dice", "val_iou",
            ])

    def train(self) -> None:
        """Run the full training loop."""
        for epoch in range(1, self.epochs + 1):
            train_metrics = self._train_one_epoch(epoch)
            val_metrics = validate(
                self.model, self.val_loader, self.criterion, self.device, amp=self.amp,
            )

            # Logging
            self._log_metrics(epoch, train_metrics, val_metrics)

            # Checkpoint
            if val_metrics["val_dice"] > self.best_dice:
                self.best_dice = val_metrics["val_dice"]
                self._save_checkpoint(epoch, is_best=True)
                logger.info(
                    f"Epoch {epoch}: New best dice={self.best_dice:.4f} — saved best_model.pth"
                )

            logger.info(
                f"Epoch {epoch}/{self.epochs} — "
                f"train_loss={train_metrics['train_loss']:.4f} "
                f"val_dice={val_metrics['val_dice']:.4f} "
                f"val_iou={val_metrics['val_iou']:.4f}"
            )

        self.writer.close()

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        n_batches = 0

        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            self.optimizer.zero_grad()
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            with torch.no_grad():
                total_dice += dice_score(logits, masks).item()
                total_iou += iou_score(logits, masks).item()
            n_batches += 1

        n_batches = max(n_batches, 1)
        return {
            "train_loss": total_loss / n_batches,
            "train_dice": total_dice / n_batches,
            "train_iou": total_iou / n_batches,
        }

    def _log_metrics(
        self, epoch: int, train: dict[str, float], val: dict[str, float],
    ) -> None:
        for key, value in {**train, **val}.items():
            self.writer.add_scalar(key, value, epoch)

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train['train_loss']:.6f}",
                f"{train['train_dice']:.6f}",
                f"{train['train_iou']:.6f}",
                f"{val['val_loss']:.6f}",
                f"{val['val_dice']:.6f}",
                f"{val['val_iou']:.6f}",
            ])

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_dice": self.best_dice,
        }
        if is_best:
            torch.save(state, self.checkpoint_dir / "best_model.pth")
