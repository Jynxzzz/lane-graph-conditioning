"""Verify lane-conditioned data pipeline and model work end-to-end."""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.trajectory.traj_dataset import TrajectoryPredictionDataset, collate_fn
from models.lane_conditioned_lstm import LaneConditionedLSTM
from training.metrics import compute_ade, compute_fde


def main():
    scene_list = "/home/xingnan/scenario-dreamer/intersection_list.txt"

    print("=" * 60)
    print("1. Testing dataset with lane features...")
    print("=" * 60)

    dataset = TrajectoryPredictionDataset(
        scene_list_path=scene_list,
        split="train",
        include_lanes=True,
        max_lanes=16,
        lane_points=10,
        include_lane_ids=True,
        augment=False,
    )
    print(f"   Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"   Sample keys: {sorted(sample.keys())}")
    for k, v in sorted(sample.items()):
        if isinstance(v, torch.Tensor):
            print(f"   {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"   {k}: {v}")

    # Check lane shapes
    assert sample["lane_features"].shape == (16, 26), f"Bad lane shape: {sample['lane_features'].shape}"
    assert sample["lane_adj"].shape == (16, 16)
    assert sample["lane_mask"].shape == (16,)
    n_valid = sample["lane_mask"].sum().item()
    print(f"\n   Valid lanes: {int(n_valid)}/16")
    print(f"   Ego lane idx: {sample['ego_lane_idx'].item()}")

    # Check lane ID assignments (Phase 3)
    if "lane_id_sequence" in sample:
        n_assigned = (sample["lane_id_mask"] > 0).sum().item()
        print(f"   GT lane assignments: {int(n_assigned)}/30 future frames")

    print()
    print("=" * 60)
    print("2. Testing DataLoader batching...")
    print("=" * 60)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    batch = next(iter(loader))
    for k, v in sorted(batch.items()):
        if isinstance(v, torch.Tensor):
            print(f"   {k}: shape={v.shape}")

    print()
    print("=" * 60)
    print("3. Testing LaneConditionedLSTM forward pass...")
    print("=" * 60)

    model = LaneConditionedLSTM(
        input_dim=2,
        embed_dim=64,
        hidden_dim=128,
        num_layers=2,
        future_len=30,
        use_neighbors=True,
        neighbor_hidden_dim=64,
        lane_feat_dim=26,
        lane_hidden_dim=64,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model params: {total_params:,}")

    with torch.no_grad():
        output = model(batch)
    pred = output["pred_future"]
    print(f"   Output shape: {pred.shape}")
    assert pred.shape == (4, 30, 2)

    ade = compute_ade(pred, batch["sdc_future"])
    fde = compute_fde(pred, batch["sdc_future"])
    print(f"   ADE: {ade.item():.4f}m, FDE: {fde.item():.4f}m")

    print()
    print("=" * 60)
    print("4. Testing backward pass...")
    print("=" * 60)

    output = model(batch)
    loss = torch.nn.functional.smooth_l1_loss(output["pred_future"], batch["sdc_future"])
    loss.backward()
    grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Gradients flowing: {grad_ok}")

    print()
    print("=" * 60)
    print("ALL CHECKS PASSED! Lane pipeline ready for training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
