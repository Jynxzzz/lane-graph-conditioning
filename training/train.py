"""Training entry point for trajectory prediction models."""

import os
import sys

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.trajectory.traj_dataset import TrajectoryPredictionDataset, collate_fn
from training.lightning_module import TrajPredModule
from training.multimodal_lightning_module import MultiModalTrajectoryPredictor
from training.flow_matching_module import FlowMatchingModule


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Seed everything
    pl.seed_everything(cfg.seed, workers=True)

    # Determine if lanes are needed based on model
    LANE_MODELS = {"lane_conditioned", "dual_supervised", "tf_lane_cond", "tf_dual_sup", "multimodal_lane_cond", "flow_matching"}
    DUAL_MODELS = {"dual_supervised", "tf_dual_sup"}
    MULTIMODAL_MODELS = {"multimodal_lstm_baseline", "multimodal_lane_cond"}
    FLOW_MODELS = {"flow_matching"}
    include_lanes = cfg.model.name in LANE_MODELS
    include_lane_ids = cfg.model.name in DUAL_MODELS

    # Build datasets
    train_dataset = TrajectoryPredictionDataset(
        scene_list_path=cfg.data.scene_list,
        split="train",
        val_ratio=cfg.data.val_ratio,
        history_len=cfg.data.history_len,
        future_len=cfg.data.future_len,
        max_neighbors=cfg.data.max_neighbors,
        neighbor_distance=cfg.data.neighbor_distance,
        anchor_frames=list(cfg.data.anchor_frames),
        augment=True,
        seed=cfg.seed,
        include_lanes=include_lanes,
        max_lanes=cfg.data.get("max_lanes", 16),
        lane_points=cfg.data.get("lane_points", 10),
        include_lane_ids=include_lane_ids,
        expand_anchors=cfg.data.get("expand_anchors", False),
        augment_rotation=cfg.data.get("augment_rotation", False),
    )

    val_dataset = TrajectoryPredictionDataset(
        scene_list_path=cfg.data.scene_list,
        split="val",
        val_ratio=cfg.data.val_ratio,
        history_len=cfg.data.history_len,
        future_len=cfg.data.future_len,
        max_neighbors=cfg.data.max_neighbors,
        neighbor_distance=cfg.data.neighbor_distance,
        anchor_frames=list(cfg.data.anchor_frames),
        augment=False,
        seed=cfg.seed,
        include_lanes=include_lanes,
        max_lanes=cfg.data.get("max_lanes", 16),
        lane_points=cfg.data.get("lane_points", 10),
        include_lane_ids=include_lane_ids,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Build model
    is_transformer = cfg.model.name.startswith("tf_")

    if is_transformer:
        model_kwargs = {
            "input_dim": cfg.model.input_dim,
            "d_model": cfg.model.get("d_model", 128),
            "nhead": cfg.model.get("nhead", 4),
            "num_layers": cfg.model.get("num_layers", 2),
            "dim_feedforward": cfg.model.get("dim_feedforward", 256),
            "dropout": cfg.model.get("dropout", 0.1),
            "future_len": cfg.model.future_len,
            "use_neighbors": cfg.model.use_neighbors,
            "neighbor_hidden_dim": cfg.model.neighbor_hidden_dim,
        }
    else:
        model_kwargs = {
            "input_dim": cfg.model.input_dim,
            "embed_dim": cfg.model.embed_dim,
            "hidden_dim": cfg.model.hidden_dim,
            "num_layers": cfg.model.num_layers,
            "future_len": cfg.model.future_len,
            "use_neighbors": cfg.model.use_neighbors,
            "neighbor_hidden_dim": cfg.model.neighbor_hidden_dim,
            "decoder_type": cfg.model.get("decoder_type", "mlp"),
        }

    # Add lane-specific params
    if cfg.model.name in LANE_MODELS:
        lane_points = cfg.data.get("lane_points", 10)
        lane_feat_dim = lane_points * 2 + 2 + 1 + 3  # centerline + dir + len + flags
        model_kwargs["lane_feat_dim"] = lane_feat_dim
        model_kwargs["lane_hidden_dim"] = cfg.model.get("lane_hidden_dim", 64)
        model_kwargs["n_mp_layers"] = cfg.model.get("n_mp_layers", 0)
    if cfg.model.name in DUAL_MODELS:
        model_kwargs["max_lanes"] = cfg.data.get("max_lanes", 16)

    # Add multi-modal specific params
    if cfg.model.name in MULTIMODAL_MODELS:
        model_kwargs["num_modes"] = cfg.model.get("num_modes", 6)

    # Instantiate appropriate module
    if cfg.model.name in FLOW_MODELS:
        # Flow matching uses its own config dict and Lightning module
        flow_config = {
            "input_dim": cfg.model.input_dim,
            "embed_dim": cfg.model.get("embed_dim", 64),
            "hidden_dim": cfg.model.get("hidden_dim", 128),
            "num_layers": cfg.model.get("num_layers", 2),
            "future_len": cfg.model.future_len,
            "use_neighbors": cfg.model.use_neighbors,
            "neighbor_hidden_dim": cfg.model.neighbor_hidden_dim,
            "denoiser_hidden_dim": cfg.model.get("denoiser_hidden_dim", 256),
            "denoiser_n_heads": cfg.model.get("denoiser_n_heads", 4),
            "denoiser_n_layers": cfg.model.get("denoiser_n_layers", 3),
            "lane_field_dim": cfg.model.get("lane_field_dim", 64),
            "lane_field_resolution": cfg.model.get("lane_field_resolution", 128),
            "bev_range": list(cfg.model.get("bev_range", [-30, 30, -10, 50])),
            "sigma": cfg.model.get("sigma", 1.0),
            "lane_field_scale": cfg.model.get("lane_field_scale", 50.0),
            "use_lane_field": cfg.model.get("use_lane_field", True),
        }
        module = FlowMatchingModule(
            model_config=flow_config,
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            max_epochs=cfg.training.max_epochs,
            warmup_epochs=cfg.training.get("warmup_epochs", 5),
            n_eval_samples=cfg.model.get("n_eval_samples", 6),
            n_eval_steps=cfg.model.get("n_eval_steps", 10),
            miss_threshold=cfg.training.get("miss_threshold", 2.0),
        )
    elif cfg.model.name in MULTIMODAL_MODELS:
        module = MultiModalTrajectoryPredictor(
            model_name=cfg.model.name,
            model_kwargs=model_kwargs,
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            max_epochs=cfg.training.max_epochs,
            num_modes=cfg.model.get("num_modes", 6),
            diversity_weight=cfg.training.get("diversity_weight", 0.1),
            warmup_epochs=cfg.training.get("warmup_epochs", 0),
        )
    else:
        module = TrajPredModule(
            model_name=cfg.model.name,
            model_kwargs=model_kwargs,
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            max_epochs=cfg.training.max_epochs,
            lambda_structure=cfg.training.get("lambda_structure", 0.1),
            warmup_epochs=cfg.training.get("warmup_epochs", 0),
        )

    # Print model summary
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # Output directory
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Callbacks (use minADE for multi-modal / flow matching models, ADE for single-mode)
    monitor_metric = "val/minADE" if cfg.model.name in (MULTIMODAL_MODELS | FLOW_MODELS) else "val/ade"
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename=f"best-{{epoch:02d}}-{{{monitor_metric}:.4f}}",
            monitor=monitor_metric,
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        pl.callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=20,
            mode="min",
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]

    # Logger
    logger = pl.loggers.CSVLogger(
        save_dir=output_dir,
        name="csv_logs",
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=32,
        gradient_clip_val=cfg.training.gradient_clip_val,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        log_every_n_steps=cfg.training.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        deterministic=False,
    )

    # Train
    trainer.fit(module, train_loader, val_loader)

    print(f"Training complete. Best model saved to {output_dir}/checkpoints/")
    print(f"Logs saved to {output_dir}/csv_logs/")


if __name__ == "__main__":
    main()
