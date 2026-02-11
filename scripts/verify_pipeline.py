"""Quick verification that the data pipeline and model work end-to-end."""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.trajectory.traj_dataset import TrajectoryPredictionDataset, collate_fn
from models.lstm_baseline import LSTMBaseline
from training.metrics import compute_ade, compute_fde


def main():
    scene_list = "/home/xingnan/scenario-dreamer/intersection_list.txt"

    print("=" * 60)
    print("1. Testing dataset loading...")
    print("=" * 60)

    dataset = TrajectoryPredictionDataset(
        scene_list_path=scene_list,
        split="train",
        val_ratio=0.15,
        history_len=11,
        future_len=30,
        max_neighbors=10,
        augment=False,
    )
    print(f"   Dataset size: {len(dataset)} scenes")

    sample = dataset[0]
    print(f"   Sample keys: {list(sample.keys())}")
    for k, v in sample.items():
        print(f"   {k}: shape={v.shape}, dtype={v.dtype}")

    # Verify shapes
    assert sample["sdc_history"].shape == (11, 2), f"Bad history shape: {sample['sdc_history'].shape}"
    assert sample["sdc_future"].shape == (30, 2), f"Bad future shape: {sample['sdc_future'].shape}"
    assert sample["neighbor_history"].shape == (10, 11, 2), f"Bad neighbor shape: {sample['neighbor_history'].shape}"
    assert sample["neighbor_mask"].shape == (10,), f"Bad mask shape: {sample['neighbor_mask'].shape}"
    print("   Shape checks PASSED")

    # Check values are reasonable (BEV coords should be within ~100m of origin)
    hist_max = sample["sdc_history"].abs().max().item()
    fut_max = sample["sdc_future"].abs().max().item()
    print(f"   History max abs value: {hist_max:.2f}m")
    print(f"   Future max abs value: {fut_max:.2f}m")
    assert hist_max < 200, f"History values too large: {hist_max}"
    assert fut_max < 200, f"Future values too large: {fut_max}"
    print("   Value range checks PASSED")

    n_valid_neighbors = int(sample["neighbor_mask"].sum().item())
    print(f"   Valid neighbors: {n_valid_neighbors}")

    print()
    print("=" * 60)
    print("2. Testing DataLoader with collation...")
    print("=" * 60)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    batch = next(iter(loader))
    print(f"   Batch keys: {list(batch.keys())}")
    for k, v in batch.items():
        print(f"   {k}: shape={v.shape}")

    print()
    print("=" * 60)
    print("3. Testing model forward pass...")
    print("=" * 60)

    model = LSTMBaseline(
        input_dim=2,
        embed_dim=64,
        hidden_dim=128,
        num_layers=2,
        future_len=30,
        use_neighbors=True,
        neighbor_hidden_dim=64,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model params: {total_params:,}")

    with torch.no_grad():
        output = model(batch)

    pred = output["pred_future"]
    print(f"   Output pred_future shape: {pred.shape}")
    assert pred.shape == (4, 30, 2), f"Bad output shape: {pred.shape}"
    print("   Forward pass PASSED")

    print()
    print("=" * 60)
    print("4. Testing metrics...")
    print("=" * 60)

    ade = compute_ade(pred, batch["sdc_future"])
    fde = compute_fde(pred, batch["sdc_future"])
    print(f"   ADE: {ade.item():.4f}m")
    print(f"   FDE: {fde.item():.4f}m")

    print()
    print("=" * 60)
    print("5. Testing backward pass (gradient flow)...")
    print("=" * 60)

    output = model(batch)
    loss = torch.nn.functional.smooth_l1_loss(output["pred_future"], batch["sdc_future"])
    loss.backward()
    print(f"   Loss: {loss.item():.4f}")

    # Check gradients exist
    grad_ok = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
        if p.requires_grad
    )
    print(f"   Gradients flowing: {grad_ok}")
    assert grad_ok, "Some parameters have no gradient!"
    print("   Backward pass PASSED")

    print()
    print("=" * 60)
    print("ALL CHECKS PASSED! Pipeline is ready for training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
