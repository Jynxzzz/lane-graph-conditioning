"""Error decomposition analysis for multi-modal trajectory prediction.

Decomposes prediction errors into:
1. Lateral (perpendicular to heading) vs Longitudinal (along heading)
2. Miss rate at standard thresholds
3. Per-timestep error growth curves
4. Mode diversity analysis

Supports both single-modal (TrajPredModule) and multi-modal (MultiModalTrajectoryPredictor).

Usage:
    python scripts/error_decomposition.py \
        --checkpoint /path/to/best.ckpt \
        --label "BL (K=6)" \
        --output-dir /path/to/output

    # Compare two models:
    python scripts/error_decomposition.py \
        --checkpoint /path/to/bl.ckpt /path/to/lc.ckpt \
        --label "LSTM BL (K=6)" "LSTM+Lane (K=6)" \
        --output-dir /path/to/output
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.trajectory.traj_dataset import TrajectoryPredictionDataset, collate_fn


def load_model(checkpoint_path, device):
    """Load model from checkpoint, auto-detecting module type."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    hparams = ckpt.get("hyper_parameters", {})
    model_name = hparams.get("model_name", "")

    # Detect multi-modal vs single-modal
    is_multimodal = "multimodal" in model_name or "num_modes" in hparams

    if is_multimodal:
        from training.multimodal_lightning_module import MultiModalTrajectoryPredictor
        module = MultiModalTrajectoryPredictor.load_from_checkpoint(
            checkpoint_path, map_location=device
        )
    else:
        from training.lightning_module import TrajPredModule
        module = TrajPredModule.load_from_checkpoint(
            checkpoint_path, map_location=device
        )

    module.eval()
    module.to(device)
    include_lanes = "lane" in model_name
    return module, is_multimodal, include_lanes, model_name


def compute_heading(history):
    """Compute heading and lateral direction from trajectory history.

    Args:
        history: (B, T, 2) trajectory history

    Returns:
        heading: (B, 2) unit heading vector (along travel direction)
        lateral: (B, 2) unit lateral vector (perpendicular, pointing left)
    """
    heading = history[:, -1, :] - history[:, -2, :]
    norms = torch.norm(heading, dim=-1, keepdim=True).clamp(min=1e-6)
    heading = heading / norms
    lateral = torch.stack([-heading[:, 1], heading[:, 0]], dim=-1)
    return heading, lateral


@torch.no_grad()
def run_evaluation(module, val_loader, device, is_multimodal):
    """Run evaluation and collect per-sample decomposed errors."""
    all_results = []

    for batch_idx, batch in enumerate(val_loader):
        if batch is None:
            continue

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        output = module(batch)
        gt = batch["sdc_future"]  # (B, T, 2)
        history = batch["sdc_history"]

        heading, lateral = compute_heading(history)

        if is_multimodal:
            predictions = output["pred_future"]  # (B, K, T, 2)
            B, K, T, _ = predictions.shape

            gt_exp = gt.unsqueeze(1).expand(B, K, T, 2)
            error_vec = predictions - gt_exp  # (B, K, T, 2)
            l2_error = torch.norm(error_vec, dim=-1)  # (B, K, T)

            # Project onto heading/lateral
            h_exp = heading[:, None, None, :]
            l_exp = lateral[:, None, None, :]
            lon_error = (error_vec * h_exp).sum(dim=-1)  # signed
            lat_error = (error_vec * l_exp).sum(dim=-1)  # signed

            # Best mode per sample
            ade_per_mode = l2_error.mean(dim=2)  # (B, K)
            best_idx = ade_per_mode.argmin(dim=1)  # (B,)
            bi = torch.arange(B, device=device)

            all_results.append({
                "best_l2": l2_error[bi, best_idx].cpu(),  # (B, T)
                "best_lon": lon_error[bi, best_idx].cpu(),
                "best_lat": lat_error[bi, best_idx].cpu(),
                "ade_per_mode": ade_per_mode.cpu(),
                "endpoint_l2_all": l2_error[:, :, -1].cpu(),  # (B, K)
            })
        else:
            pred = output["pred_future"]  # (B, T, 2)
            error_vec = pred - gt
            l2_error = torch.norm(error_vec, dim=-1)

            lon_error = (error_vec * heading.unsqueeze(1)).sum(dim=-1)
            lat_error = (error_vec * lateral.unsqueeze(1)).sum(dim=-1)

            all_results.append({
                "best_l2": l2_error.cpu(),
                "best_lon": lon_error.cpu(),
                "best_lat": lat_error.cpu(),
                "ade_per_mode": l2_error.mean(dim=1, keepdim=True).cpu(),
                "endpoint_l2_all": l2_error[:, -1:].cpu(),
            })

        if (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx+1}...")

    # Concatenate
    best_l2 = torch.cat([r["best_l2"] for r in all_results], dim=0)
    best_lon = torch.cat([r["best_lon"] for r in all_results], dim=0)
    best_lat = torch.cat([r["best_lat"] for r in all_results], dim=0)
    ade_per_mode = torch.cat([r["ade_per_mode"] for r in all_results], dim=0)
    endpoint_l2 = torch.cat([r["endpoint_l2_all"] for r in all_results], dim=0)

    return best_l2, best_lon, best_lat, ade_per_mode, endpoint_l2


def analyze(best_l2, best_lon, best_lat, ade_per_mode, endpoint_l2,
            label="Model", dt=0.1):
    """Compute and print all analysis metrics."""
    N, T = best_l2.shape
    K = ade_per_mode.shape[1]
    abs_lon = best_lon.abs()
    abs_lat = best_lat.abs()

    print(f"\n{'='*70}")
    print(f"  {label}  |  N={N}, K={K}, horizon={T*dt:.1f}s")
    print(f"{'='*70}")

    # 1. Overall
    minADE = ade_per_mode.min(dim=1)[0].mean().item()
    minFDE = endpoint_l2.min(dim=1)[0].mean().item()
    print(f"\n  minADE: {minADE:.4f} m    minFDE: {minFDE:.4f} m")

    # 2. Lat vs Lon
    avg_lon = abs_lon.mean().item()
    avg_lat = abs_lat.mean().item()
    end_lon = abs_lon[:, -1].mean().item()
    end_lat = abs_lat[:, -1].mean().item()

    print(f"\n  --- Lateral vs Longitudinal (best mode) ---")
    print(f"  {'':20s} {'Longitudinal':>12} {'Lateral':>12} {'Lon/Lat':>10}")
    print(f"  {'Average error':<20s} {avg_lon:>12.4f} {avg_lat:>12.4f} {avg_lon/max(avg_lat,1e-6):>10.2f}x")
    print(f"  {'Endpoint error':<20s} {end_lon:>12.4f} {end_lat:>12.4f} {end_lon/max(end_lat,1e-6):>10.2f}x")

    # 3. Miss rate
    best_endpoint = endpoint_l2.min(dim=1)[0]
    print(f"\n  --- Miss Rate ---")
    miss_rates = {}
    for thr in [1.0, 2.0, 3.0, 5.0]:
        mr = (best_endpoint > thr).float().mean().item()
        miss_rates[thr] = mr
        print(f"  MR@{thr:.0f}m: {mr*100:.1f}%")

    # 4. Error growth
    print(f"\n  --- Error Growth (best mode, per second) ---")
    print(f"  {'Time':>6s} {'L2':>8s} {'Lon':>8s} {'Lat':>8s} {'Lat%':>8s}")
    per_sec = {}
    for t_sec in [1, 2, 3, 4, 5, 6, 7, 8]:
        t_idx = int(t_sec / dt) - 1
        if t_idx >= T:
            break
        l2_t = best_l2[:, t_idx].mean().item()
        lon_t = abs_lon[:, t_idx].mean().item()
        lat_t = abs_lat[:, t_idx].mean().item()
        lat_pct = lat_t / (lon_t + lat_t) * 100 if (lon_t + lat_t) > 0 else 0
        print(f"  {t_sec:>5.0f}s {l2_t:>8.3f} {lon_t:>8.3f} {lat_t:>8.3f} {lat_pct:>7.1f}%")
        per_sec[t_sec] = {"l2": l2_t, "lon": lon_t, "lat": lat_t}

    # 5. Mode diversity (multi-modal only)
    if K > 1:
        print(f"\n  --- Mode Diversity ---")
        spread = ade_per_mode.std(dim=1).mean().item()
        gap = (ade_per_mode.max(dim=1)[0] - ade_per_mode.min(dim=1)[0]).mean().item()
        print(f"  Mode ADE std: {spread:.4f}, best-worst gap: {gap:.4f}")

    return {
        "label": label, "N": N, "K": K,
        "minADE": minADE, "minFDE": minFDE,
        "avg_lon": avg_lon, "avg_lat": avg_lat,
        "end_lon": end_lon, "end_lat": end_lat,
        "miss_rates": miss_rates,
        "per_sec": per_sec,
        "per_timestep_l2": best_l2.mean(dim=0).numpy(),
        "per_timestep_lon": abs_lon.mean(dim=0).numpy(),
        "per_timestep_lat": abs_lat.mean(dim=0).numpy(),
    }


def generate_figures(all_results, output_dir, dt=0.1):
    """Generate comparison figures for all models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
    T = len(all_results[0]["per_timestep_l2"])
    time_axis = np.arange(1, T + 1) * dt

    # === Figure 1: Error growth curves (3 panels: L2, Lon, Lat) ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (key, title) in zip(axes, [
        ("per_timestep_l2", "Total Error (L2)"),
        ("per_timestep_lon", "Longitudinal Error"),
        ("per_timestep_lat", "Lateral Error"),
    ]):
        for i, r in enumerate(all_results):
            ax.plot(time_axis, r[key], color=colors[i % len(colors)],
                    linewidth=2, label=r["label"])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Error (m)")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "error_growth_curves.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # === Figure 2: Lat vs Lon bar chart at key horizons ===
    if len(all_results) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, (comp, comp_label) in zip(axes, [
            ("per_timestep_lon", "Longitudinal Error"),
            ("per_timestep_lat", "Lateral Error"),
        ]):
            horizons = []
            for t_sec in [2, 4, 6, 8]:
                t_idx = int(t_sec / dt) - 1
                if t_idx < T:
                    horizons.append((t_idx, f"{t_sec}s"))

            x = np.arange(len(horizons))
            width = 0.8 / len(all_results)

            for i, r in enumerate(all_results):
                vals = [r[comp][t_idx] for t_idx, _ in horizons]
                offset = (i - len(all_results) / 2 + 0.5) * width
                bars = ax.bar(x + offset, vals, width, label=r["label"],
                              color=colors[i % len(colors)], edgecolor="black",
                              linewidth=0.5)

            ax.set_xticks(x)
            ax.set_xticklabels([lbl for _, lbl in horizons])
            ax.set_xlabel("Prediction Horizon")
            ax.set_ylabel("Error (m)")
            ax.set_title(comp_label, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        path = os.path.join(output_dir, "lat_lon_comparison.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

    # === Figure 3: Improvement breakdown ===
    if len(all_results) >= 2:
        bl = all_results[0]
        lc = all_results[1]

        fig, ax = plt.subplots(figsize=(10, 5))
        horizons = []
        for t_sec in [1, 2, 3, 4, 5, 6, 7, 8]:
            t_idx = int(t_sec / dt) - 1
            if t_idx < T:
                horizons.append((t_idx, f"{t_sec}s"))

        x = np.arange(len(horizons))
        width = 0.25

        for ci, (comp, comp_label, color) in enumerate([
            ("per_timestep_lon", "Longitudinal", "#2196F3"),
            ("per_timestep_lat", "Lateral", "#F44336"),
            ("per_timestep_l2", "Total L2", "#4CAF50"),
        ]):
            imps = []
            for t_idx, _ in horizons:
                bl_val = bl[comp][t_idx]
                lc_val = lc[comp][t_idx]
                imp = (bl_val - lc_val) / bl_val * 100 if bl_val > 0 else 0
                imps.append(imp)

            offset = (ci - 1) * width
            bars = ax.bar(x + offset, imps, width, label=comp_label,
                          color=color, edgecolor="black", linewidth=0.5)

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in horizons])
        ax.set_xlabel("Prediction Horizon")
        ax.set_ylabel("Improvement over Baseline (%)")
        ax.set_title(f"Lane Conditioning Improvement: {lc['label']} vs {bl['label']}",
                      fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        path = os.path.join(output_dir, "improvement_breakdown.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

    # === Print comparison table ===
    if len(all_results) >= 2:
        bl = all_results[0]
        lc = all_results[1]
        print(f"\n{'='*70}")
        print(f"  Comparison: {bl['label']}  vs  {lc['label']}")
        print(f"{'='*70}")
        print(f"  {'Metric':<20s} {'BL':>10s} {'LC':>10s} {'Improve':>10s}")
        for name, bl_v, lc_v in [
            ("minADE", bl["minADE"], lc["minADE"]),
            ("minFDE", bl["minFDE"], lc["minFDE"]),
            ("Avg Lon err", bl["avg_lon"], lc["avg_lon"]),
            ("Avg Lat err", bl["avg_lat"], lc["avg_lat"]),
            ("End Lon err", bl["end_lon"], lc["end_lon"]),
            ("End Lat err", bl["end_lat"], lc["end_lat"]),
            ("MR@2m", bl["miss_rates"][2.0], lc["miss_rates"][2.0]),
            ("MR@5m", bl["miss_rates"][5.0], lc["miss_rates"][5.0]),
        ]:
            if "MR" in name:
                imp = (bl_v - lc_v) / max(bl_v, 1e-6) * 100
                print(f"  {name:<20s} {bl_v*100:>9.1f}% {lc_v*100:>9.1f}% {imp:>+9.1f}%")
            else:
                imp = (bl_v - lc_v) / max(bl_v, 1e-6) * 100
                print(f"  {name:<20s} {bl_v:>10.4f} {lc_v:>10.4f} {imp:>+9.1f}%")


def find_best_ckpt(exp_dir):
    """Find the best checkpoint in an experiment directory."""
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    best_path = None
    best_val = float("inf")

    for root, dirs, files in os.walk(ckpt_dir):
        for f in files:
            if f.endswith(".ckpt") and f != "last.ckpt":
                try:
                    val = float(f.split("=")[-1].replace(".ckpt", ""))
                    if val < best_val:
                        best_val = val
                        best_path = os.path.join(root, f)
                except (ValueError, IndexError):
                    pass

    if best_path is None:
        last = os.path.join(ckpt_dir, "last.ckpt")
        if os.path.exists(last):
            return last
    return best_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--label", nargs="+", default=[])
    parser.add_argument("--scene-list",
                        default="/home/xingnan/scenario-dreamer/scene_list_123k_signal_ssd.txt")
    parser.add_argument("--future-len", type=int, default=80)
    parser.add_argument("--history-len", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output-dir", default="/mnt/hdd12t/outputs/scenario_dreamer_figures")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_analysis = []

    for i, ckpt_path in enumerate(args.checkpoint):
        # If directory, find best checkpoint
        if os.path.isdir(ckpt_path):
            ckpt_path = find_best_ckpt(ckpt_path)
            if ckpt_path is None:
                print(f"No checkpoint found in {args.checkpoint[i]}")
                continue

        label = args.label[i] if i < len(args.label) else f"Model {i}"
        print(f"\n{'#'*70}")
        print(f"  Loading: {ckpt_path}")
        print(f"  Label: {label}")

        module, is_multimodal, include_lanes, model_name = load_model(ckpt_path, device)
        print(f"  Model: {model_name}, multimodal={is_multimodal}, lanes={include_lanes}")

        # Build val dataset
        val_dataset = TrajectoryPredictionDataset(
            scene_list_path=args.scene_list,
            split="val",
            val_ratio=0.15,
            history_len=args.history_len,
            future_len=args.future_len,
            max_neighbors=10,
            neighbor_distance=30.0,
            anchor_frames=[10],
            augment=False,
            seed=42,
            include_lanes=include_lanes,
            max_lanes=16,
            lane_points=10,
        )
        print(f"  Val samples: {len(val_dataset)}")

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        print("  Running evaluation...")
        best_l2, best_lon, best_lat, ade_per_mode, endpoint_l2 = run_evaluation(
            module, val_loader, device, is_multimodal
        )

        result = analyze(best_l2, best_lon, best_lat, ade_per_mode, endpoint_l2, label=label)
        all_analysis.append(result)

        # Free GPU memory
        del module
        torch.cuda.empty_cache()

    # Generate comparison figures
    if all_analysis:
        print("\nGenerating figures...")
        generate_figures(all_analysis, args.output_dir)


if __name__ == "__main__":
    main()
