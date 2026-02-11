"""Evaluate additional baselines for fair comparison.

Adds non-learned baselines that are standard in trajectory prediction:
1. Constant Velocity (CV): assume last observed velocity continues
2. Constant Acceleration (CA): assume last observed acceleration continues
3. Linear Extrapolation: fit line to history, extrapolate

These are important for calibrating our learned models against
trivial baselines — a common reviewer question.
"""

import math
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.trajectory.traj_dataset import (
    _extract_ego_info,
    _extract_full_trajectory,
    _world_to_ego,
)
from datasets.trajectory.lane_feature_utils import extract_lane_features
from training.metrics import compute_ade, compute_fde, compute_metrics_at_horizons


def constant_velocity_predict(history, future_len=30):
    """Predict future by assuming constant velocity from last 2 observations.

    Args:
        history: (T_hist, 2) numpy array of observed positions
        future_len: number of future steps to predict

    Returns:
        pred: (future_len, 2) predicted positions
    """
    velocity = history[-1] - history[-2]  # (2,) last velocity
    last_pos = history[-1]
    pred = np.zeros((future_len, 2))
    for t in range(future_len):
        pred[t] = last_pos + velocity * (t + 1)
    return pred


def constant_acceleration_predict(history, future_len=30):
    """Predict with constant acceleration from last 3 observations."""
    if len(history) < 3:
        return constant_velocity_predict(history, future_len)

    v1 = history[-2] - history[-3]
    v2 = history[-1] - history[-2]
    accel = v2 - v1
    velocity = v2

    last_pos = history[-1]
    pred = np.zeros((future_len, 2))
    current_v = velocity.copy()
    current_pos = last_pos.copy()
    for t in range(future_len):
        current_v = current_v + accel
        current_pos = current_pos + current_v
        pred[t] = current_pos
    return pred


def linear_extrapolation_predict(history, future_len=30):
    """Fit a line to history and extrapolate."""
    T = len(history)
    t_hist = np.arange(T).astype(float)

    # Fit line for x and y separately
    pred = np.zeros((future_len, 2))
    for dim in range(2):
        coeffs = np.polyfit(t_hist, history[:, dim], deg=1)
        t_future = np.arange(T, T + future_len).astype(float)
        pred[:, dim] = np.polyval(coeffs, t_future)
    return pred


def evaluate_baselines_on_scenes(scene_paths, anchor_frame=30, history_len=11, future_len=30):
    """Run CV/CA/Linear baselines on scenes and compute metrics."""
    methods = {
        "Constant Velocity": constant_velocity_predict,
        "Constant Acceleration": constant_acceleration_predict,
        "Linear Extrapolation": linear_extrapolation_predict,
    }

    all_results = {name: {"ade": [], "fde": [], "per_step": []} for name in methods}

    for i, scene_path in enumerate(scene_paths):
        if i % 50 == 0:
            print(f"  Processing scene {i+1}/{len(scene_paths)}...")

        try:
            with open(scene_path, "rb") as f:
                scene = pickle.load(f)
            ego_pos, ego_heading = _extract_ego_info(scene, anchor_frame)
            ego_traj, ego_valid = _extract_full_trajectory(scene["objects"][scene["av_idx"]])
            ego_bev = _world_to_ego(ego_traj, ego_pos, ego_heading)
        except Exception:
            continue

        h_start = anchor_frame - (history_len - 1)
        h_end = anchor_frame + 1
        f_start = anchor_frame + 1
        f_end = anchor_frame + 1 + future_len

        history = ego_bev[h_start:h_end]
        gt_future = ego_bev[f_start:f_end]

        if len(history) < history_len or len(gt_future) < future_len:
            continue

        for name, predict_fn in methods.items():
            pred = predict_fn(history, future_len)
            ade = np.mean(np.linalg.norm(pred - gt_future, axis=1))
            fde = np.linalg.norm(pred[-1] - gt_future[-1])
            per_step = np.linalg.norm(pred - gt_future, axis=1)

            all_results[name]["ade"].append(ade)
            all_results[name]["fde"].append(fde)
            all_results[name]["per_step"].append(per_step)

    return all_results


def main():
    with open("/home/xingnan/scenario-dreamer/intersection_list.txt") as f:
        scenes = [l.strip() for l in f if l.strip()]

    # Use all scenes from the val split (same as training val split)
    # Val scenes = last 15% after shuffle with seed=42
    np.random.seed(42)
    indices = np.random.permutation(len(scenes))
    n_val = int(len(scenes) * 0.15)
    val_indices = indices[-n_val:]
    val_scenes = [scenes[i] for i in val_indices]
    print(f"Evaluating on {len(val_scenes)} validation scenes "
          f"(x5 anchor frames = {len(val_scenes)*5} samples)")

    # Evaluate with multiple anchor frames (same as training)
    all_results = None
    for anchor in [10, 20, 30, 40, 50]:
        print(f"\n--- Anchor frame {anchor} ---")
        results = evaluate_baselines_on_scenes(val_scenes, anchor_frame=anchor)

        if all_results is None:
            all_results = results
        else:
            for name in all_results:
                for key in ["ade", "fde", "per_step"]:
                    all_results[name][key].extend(results[name][key])

    # Print results
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 80)

    # Also compute per-horizon metrics
    print(f"\n{'Method':<25} {'ADE@1s':>10} {'FDE@1s':>10} {'ADE@2s':>10} "
          f"{'FDE@2s':>10} {'ADE@3s':>10} {'FDE@3s':>10}")
    print("-" * 95)

    for name in all_results:
        per_step = np.array(all_results[name]["per_step"])  # (N, 30)
        n = len(per_step)

        ade_1s = per_step[:, :10].mean()
        fde_1s = per_step[:, 9].mean()
        ade_2s = per_step[:, :20].mean()
        fde_2s = per_step[:, 19].mean()
        ade_3s = per_step[:, :30].mean()
        fde_3s = per_step[:, 29].mean()

        print(f"{name:<25} {ade_1s:>9.3f}m {fde_1s:>9.3f}m "
              f"{ade_2s:>9.3f}m {fde_2s:>9.3f}m "
              f"{ade_3s:>9.3f}m {fde_3s:>9.3f}m")

    # Add our learned models for comparison (V3 MLP decoder results)
    print(f"\n--- Our Learned Models (V3 MLP decoder, mean over 5 seeds) ---")
    print(f"{'LSTM Baseline':<25} {'0.178':>10} {'0.403':>10} "
          f"{'0.537':>10} {'1.390':>10} {'1.096':>10} {'2.970':>10}")
    print(f"{'Lane-Conditioned':<25} {'0.173':>10} {'0.390':>10} "
          f"{'0.529':>10} {'1.378':>10} {'1.086':>10} {'2.949':>10}")
    print(f"{'Dual Supervised':<25} {'0.178':>10} {'0.397':>10} "
          f"{'0.535':>10} {'1.380':>10} {'1.089':>10} {'2.943':>10}")

    print("=" * 80)
    n_samples = len(all_results["Constant Velocity"]["ade"])
    print(f"\nTotal samples evaluated: {n_samples}")


if __name__ == "__main__":
    main()
