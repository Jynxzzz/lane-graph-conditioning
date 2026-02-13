"""PyTorch Lightning module for multi-modal trajectory prediction."""

import torch
import pytorch_lightning as pl
from models import build_model
from .metrics import compute_multimodal_metrics


class MultiModalTrajectoryPredictor(pl.LightningModule):
    """Lightning module for multi-modal trajectory prediction with WTA loss."""

    def __init__(
        self,
        model_name="multimodal_lstm_baseline",
        model_kwargs=None,
        lr=1e-3,
        weight_decay=1e-4,
        max_epochs=100,
        num_modes=6,
        diversity_weight=0.1,
        warmup_epochs=0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.num_modes = num_modes
        self.diversity_weight = diversity_weight
        self.warmup_epochs = warmup_epochs

        # Build model
        model_kwargs = model_kwargs or {}
        model_kwargs["num_modes"] = num_modes
        self.model = build_model(model_name, **model_kwargs)

    def forward(self, batch):
        return self.model(batch)

    def compute_wta_loss(self, predictions, gt, confidences=None):
        """Winner-Takes-All loss: only backprop through best prediction.

        Args:
            predictions: (B, K, T, 2)
            gt: (B, T, 2)
            confidences: (B, K) optional mode confidences

        Returns:
            loss: scalar
        """
        B, K, T, _ = predictions.shape
        gt_expanded = gt.unsqueeze(1).expand(B, K, T, 2)

        # Compute L2 error for each mode
        errors = torch.norm(predictions - gt_expanded, dim=-1)  # (B, K, T)
        ade_per_mode = errors.mean(dim=2)  # (B, K)

        # Winner-takes-all: only use best mode for each sample
        best_mode_idx = ade_per_mode.argmin(dim=1)  # (B,)
        best_errors = ade_per_mode[torch.arange(B), best_mode_idx]  # (B,)
        wta_loss = best_errors.mean()

        # Optional: confidence loss (NLL)
        if confidences is not None:
            log_probs = torch.log(confidences + 1e-8)
            best_log_probs = log_probs[torch.arange(B), best_mode_idx]
            confidence_loss = -best_log_probs.mean()
            total_loss = wta_loss + 0.1 * confidence_loss
        else:
            total_loss = wta_loss

        # Diversity loss: penalize similar predictions
        # NOTE: Disabled for now - WTA loss already encourages diversity implicitly
        # If needed, add a margin-based diversity loss that penalizes mode collapse

        return total_loss

    def training_step(self, batch, batch_idx):
        output = self(batch)
        predictions = output["pred_future"]  # (B, K, T, 2)
        confidences = output.get("confidences")  # (B, K)
        gt = batch["sdc_future"]  # (B, T, 2)

        # WTA loss
        loss = self.compute_wta_loss(predictions, gt, confidences)

        # Metrics (minADE, minFDE)
        with torch.no_grad():
            minADE, minFDE = compute_multimodal_metrics(predictions, gt)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/minADE", minADE, prog_bar=True)
        self.log("train/minFDE", minFDE)

        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        predictions = output["pred_future"]
        confidences = output.get("confidences")
        gt = batch["sdc_future"]

        # WTA loss
        loss = self.compute_wta_loss(predictions, gt, confidences)

        # Metrics
        minADE, minFDE = compute_multimodal_metrics(predictions, gt)

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/minADE", minADE, prog_bar=True, sync_dist=True)
        self.log("val/minFDE", minFDE, sync_dist=True)

    def configure_optimizers(self):
        import math

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Warmup + Cosine annealing LR schedule
        if self.warmup_epochs > 0:
            def lr_lambda(epoch):
                if epoch < self.warmup_epochs:
                    return (epoch + 1) / self.warmup_epochs
                progress = (epoch - self.warmup_epochs) / max(
                    self.max_epochs - self.warmup_epochs, 1
                )
                return 0.5 * (1 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            # Standard cosine annealing
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_epochs
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
