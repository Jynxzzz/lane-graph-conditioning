import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from models import build_model
from training.metrics import compute_ade, compute_fde, compute_metrics_at_horizons


class TrajPredModule(pl.LightningModule):
    """PyTorch Lightning module for trajectory prediction training."""

    def __init__(
        self,
        model_name="lstm_baseline",
        model_kwargs=None,
        lr=1e-3,
        weight_decay=1e-4,
        max_epochs=100,
        lambda_structure=0.1,
        warmup_epochs=0,
    ):
        super().__init__()
        self.save_hyperparameters()

        model_kwargs = model_kwargs or {}
        self.model = build_model(model_name, **model_kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.lambda_structure = lambda_structure
        self.warmup_epochs = warmup_epochs

    def forward(self, batch):
        return self.model(batch)

    def _compute_loss(self, output, batch):
        pred = output["pred_future"]   # (B, T, 2)
        gt = batch["sdc_future"]       # (B, T, 2)
        traj_loss = F.smooth_l1_loss(pred, gt)

        # Dual supervision: structure prediction loss (Phase 3)
        if "lane_logits" in output and "lane_id_sequence" in batch:
            lane_logits = output["lane_logits"]  # (B, T, max_lanes)
            lane_ids = batch["lane_id_sequence"]  # (B, T) long
            lane_mask = batch["lane_id_mask"]     # (B, T) float

            # Clamp lane_ids to valid range (replace -1 with 0, mask handles it)
            lane_ids_clamped = lane_ids.clamp(min=0)

            B, T, C = lane_logits.shape
            struct_loss = F.cross_entropy(
                lane_logits.reshape(B * T, C),
                lane_ids_clamped.reshape(B * T),
                reduction="none",
            )
            # Mask invalid assignments
            struct_loss = (struct_loss * lane_mask.reshape(B * T)).sum()
            n_valid = lane_mask.sum().clamp(min=1)
            struct_loss = struct_loss / n_valid

            self.log("train/struct_loss", struct_loss, prog_bar=False)
            return traj_loss + self.lambda_structure * struct_loss

        return traj_loss

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None

        output = self(batch)
        loss = self._compute_loss(output, batch)

        with torch.no_grad():
            pred = output["pred_future"]
            gt = batch["sdc_future"]
            ade = compute_ade(pred, gt)
            fde = compute_fde(pred, gt)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/ade", ade, prog_bar=True)
        self.log("train/fde", fde, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return None

        output = self(batch)
        loss = self._compute_loss(output, batch)

        pred = output["pred_future"]
        gt = batch["sdc_future"]
        ade = compute_ade(pred, gt)
        fde = compute_fde(pred, gt)

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/ade", ade, prog_bar=True, sync_dist=True)
        self.log("val/fde", fde, prog_bar=True, sync_dist=True)

        # Log at multiple horizons
        horizon_metrics = compute_metrics_at_horizons(pred, gt)
        for k, v in horizon_metrics.items():
            self.log(f"val/{k}", v, sync_dist=True)

        return loss

    def configure_optimizers(self):
        import math

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        if self.warmup_epochs > 0:
            warmup = self.warmup_epochs
            total = self.max_epochs

            def lr_lambda(epoch):
                if epoch < warmup:
                    return (epoch + 1) / warmup
                progress = (epoch - warmup) / max(total - warmup, 1)
                return 0.5 * (1 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_epochs
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
