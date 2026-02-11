"""PyTorch Lightning module for LaneFlowNet flow matching training.

Training procedure:
    1. Sample t ~ U(0, 1) for each batch element
    2. Sample x_0 ~ N(0, sigma^2 * I)
    3. Compute x_t = (1-t)*x_0 + t*x_1 where x_1 = GT future trajectory
    4. Compute target velocity u_t = x_1 - x_0
    5. Predict v_theta = model(x_t, t, conditions)
    6. Loss = ||v_theta - u_t||^2

Validation procedure:
    1. Generate K=6 samples via ODE integration
    2. Compute minADE, minFDE, MR (miss rate)
"""

import math

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from models.flow_matching import LaneFlowNet
from training.metrics import compute_multimodal_metrics


class FlowMatchingModule(pl.LightningModule):
    """Training module for LaneFlowNet flow matching trajectory prediction."""

    def __init__(
        self,
        model_config=None,
        lr=1e-4,
        weight_decay=1e-4,
        max_epochs=100,
        warmup_epochs=5,
        n_eval_samples=6,
        n_eval_steps=10,
        miss_threshold=2.0,
    ):
        """
        Args:
            model_config: dict of LaneFlowNet config parameters
            lr: learning rate
            weight_decay: AdamW weight decay
            max_epochs: total training epochs (for LR schedule)
            warmup_epochs: linear warmup epochs
            n_eval_samples: number of trajectory samples during validation
            n_eval_steps: number of ODE integration steps during validation
            miss_threshold: FDE threshold (meters) for miss rate computation
        """
        super().__init__()
        self.save_hyperparameters()

        model_config = model_config or {}
        self.model = LaneFlowNet(model_config)

        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.n_eval_samples = n_eval_samples
        self.n_eval_steps = n_eval_steps
        self.miss_threshold = miss_threshold

    def forward(self, batch, **kwargs):
        return self.model(batch, **kwargs)

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None

        output = self.model(batch)
        pred_v = output["pred_velocity"]      # (B, T, 2)
        target_v = output["target_velocity"]  # (B, T, 2)

        # Flow matching loss: MSE between predicted and target velocity
        loss = F.mse_loss(pred_v, target_v)

        # Also log L1 for interpretability
        with torch.no_grad():
            l1_error = (pred_v - target_v).abs().mean()

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/velocity_l1", l1_error, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return None

        # Generate samples via ODE integration
        trajectories = self.model.sample(
            batch, n_samples=self.n_eval_samples, n_steps=self.n_eval_steps
        )  # (B, K, T, 2)

        gt = batch["sdc_future"]  # (B, T, 2)

        # Compute multi-modal metrics
        minADE, minFDE = compute_multimodal_metrics(trajectories, gt)

        # Miss rate: fraction of samples where best FDE > threshold
        B, K, T, _ = trajectories.shape
        gt_final = gt[:, -1, :].unsqueeze(1).expand(B, K, 2)  # (B, K, 2)
        pred_final = trajectories[:, :, -1, :]  # (B, K, 2)
        fde_per_mode = torch.norm(pred_final - gt_final, dim=-1)  # (B, K)
        best_fde = fde_per_mode.min(dim=1)[0]  # (B,)
        miss_rate = (best_fde > self.miss_threshold).float().mean()

        # Also compute flow matching loss for monitoring
        output = self.model(batch)
        val_loss = F.mse_loss(output["pred_velocity"], output["target_velocity"])

        self.log("val/loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val/minADE", minADE, prog_bar=True, sync_dist=True)
        self.log("val/minFDE", minFDE, prog_bar=True, sync_dist=True)
        self.log("val/miss_rate", miss_rate, sync_dist=True)

        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        warmup = self.warmup_epochs
        total = self.max_epochs

        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(total - warmup, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
