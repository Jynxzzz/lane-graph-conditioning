"""Generate publication-quality qualitative trajectory visualization figures.

Produces side-by-side comparison figures: LSTM Baseline vs LSTM + Lane Conditioning.
Each figure shows K=6 multi-modal predictions, ground truth, history, and lane structure.

Usage:
    CUDA_VISIBLE_DEVICES="" python scripts/generate_paper_figures.py
"""

import math
import os
import pickle
import random
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.trajectory.traj_dataset import (
    TrajectoryPredictionDataset,
    _extract_ego_info,
    _extract_full_trajectory,
    _world_to_ego,
)
from datasets.trajectory.lane_feature_utils import extract_lane_features
from tools.lane_graph.lane_explorer import build_waterflow_graph, find_ego_lane_id
from training.multimodal_lightning_module import MultiModalTrajectoryPredictor


# ============================================================================
# Configuration
# ============================================================================
BL_CKPT = "/mnt/hdd12t/outputs/scenario_dreamer_big_v1/multimodal_bl_8s_seed42/checkpoints/last.ckpt"
LC_CKPT = "/mnt/hdd12t/outputs/scenario_dreamer_big_v1/multimodal_lc_8s_seed42/checkpoints/last.ckpt"
SCENE_LIST = "/home/xingnan/scenario-dreamer/scene_list_123k_signal_ssd.txt"
OUTPUT_DIR = "/home/xingnan/projects/sustainability-paper/figures/qualitative"
PAPER_DIR = "/home/xingnan/projects/sustainability-paper/figures"

HISTORY_LEN = 11
FUTURE_LEN = 80  # 8 seconds at 10Hz
ANCHOR_FRAME = 10  # matches training config
NUM_MODES = 6
NUM_SAMPLES = 20
SEED = 12345
BEV_RADIUS = 70  # meters, wider for 8s prediction

# ============================================================================
# Color scheme (publication quality, colorblind-friendly)
# ============================================================================
COLOR_HISTORY = "#1565C0"          # blue
COLOR_GT = "#212121"               # near-black
COLOR_LANE_BG = "#B0BEC5"         # light gray
COLOR_LANE_EGO = "#1565C0"        # blue
COLOR_LANE_CONNECTED = "#78909C"  # blue-gray
COLOR_NEIGHBOR = "#90A4AE"        # light blue-gray
COLOR_SDC = "#0D47A1"             # dark blue

# Mode colors: ordered from warm to cool for visual distinction
MODE_COLORS = [
    "#E53935",  # red
    "#FB8C00",  # orange
    "#FDD835",  # yellow
    "#43A047",  # green
    "#1E88E5",  # blue
    "#8E24AA",  # purple
]


def load_model(ckpt_path, device="cpu"):
    """Load a MultiModalTrajectoryPredictor from checkpoint."""
    print(f"  Loading checkpoint: {os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))}")
    module = MultiModalTrajectoryPredictor.load_from_checkpoint(
        ckpt_path, map_location=device
    )
    module.eval()
    return module.model


def load_scene(scene_path):
    """Load and preprocess a scene pickle file."""
    with open(scene_path, "rb") as f:
        scene = pickle.load(f)
    return scene


def build_sample(scene, anchor_frame=10, history_len=11, future_len=80,
                 max_neighbors=10, neighbor_distance=30.0,
                 include_lanes=True, max_lanes=16, lane_points=10):
    """Build a model-ready sample dict from a raw scene.

    Returns:
        sample: dict of tensors
        ego_pos: world position
        ego_heading: heading in degrees
        ego_traj_bev: full trajectory in BEV
        lane_data: dict with lane centerlines etc (or None)
    """
    ego_pos, ego_heading = _extract_ego_info(scene, anchor_frame)
    av_idx = scene["av_idx"]
    ego_obj = scene["objects"][av_idx]
    ego_traj_world, ego_valid = _extract_full_trajectory(ego_obj)
    ego_traj_bev = _world_to_ego(ego_traj_world, ego_pos, ego_heading)

    n_frames = len(ego_obj["position"])

    h_start = anchor_frame - (history_len - 1)
    h_end = anchor_frame + 1
    f_start = anchor_frame + 1
    f_end = min(anchor_frame + 1 + future_len, n_frames)

    if h_start < 0 or f_end > n_frames:
        return None, None, None, None, None

    sdc_history = ego_traj_bev[h_start:h_end].copy()
    sdc_future = ego_traj_bev[f_start:f_end].copy()

    # Pad future if shorter than expected
    if sdc_future.shape[0] < future_len:
        pad = np.zeros((future_len - sdc_future.shape[0], 2), dtype=np.float32)
        sdc_future = np.concatenate([sdc_future, pad], axis=0)

    sdc_history_valid = ego_valid[h_start:h_end]
    sdc_future_valid = ego_valid[f_start:f_end]
    if not sdc_history_valid.all() or not sdc_future_valid[:future_len].all():
        return None, None, None, None, None

    # Neighbors
    ego_world_pos = ego_traj_world[anchor_frame]
    neighbor_history = np.zeros((max_neighbors, history_len, 2), dtype=np.float32)
    neighbor_mask = np.zeros(max_neighbors, dtype=np.float32)
    n_count = 0

    for i, obj in enumerate(scene["objects"]):
        if i == av_idx or n_count >= max_neighbors:
            break
        if not obj["valid"][anchor_frame]:
            continue
        pos = obj["position"][anchor_frame]
        dx = float(pos["x"]) - ego_world_pos[0]
        dy = float(pos["y"]) - ego_world_pos[1]
        if math.sqrt(dx * dx + dy * dy) > neighbor_distance:
            continue
        obj_traj, obj_valid = _extract_full_trajectory(obj)
        obj_bev = _world_to_ego(obj_traj, ego_pos, ego_heading)
        nh = obj_bev[h_start:h_end].copy()
        nv = obj_valid[h_start:h_end]
        nh[~nv] = 0.0
        neighbor_history[n_count] = nh
        neighbor_mask[n_count] = 1.0
        n_count += 1

    sample = {
        "sdc_history": torch.from_numpy(sdc_history),
        "sdc_future": torch.from_numpy(sdc_future),
        "neighbor_history": torch.from_numpy(neighbor_history),
        "neighbor_mask": torch.from_numpy(neighbor_mask),
    }

    # Lane features
    lane_data = None
    if include_lanes:
        lane_data = extract_lane_features(
            scene, ego_pos, ego_heading,
            max_lanes=max_lanes, lane_points=lane_points,
        )
        sample["lane_features"] = torch.from_numpy(lane_data["lane_features"])
        sample["lane_adj"] = torch.from_numpy(lane_data["lane_adj"])
        sample["lane_mask"] = torch.from_numpy(lane_data["lane_mask"])
        sample["ego_lane_idx"] = torch.tensor(lane_data["ego_lane_idx"], dtype=torch.long)

    return sample, ego_pos, ego_heading, ego_traj_bev, lane_data


def predict(model, sample, device="cpu"):
    """Run inference and return predictions + confidences as numpy.

    Returns:
        predictions: (K, T, 2) numpy
        confidences: (K,) numpy
    """
    with torch.no_grad():
        batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
        output = model(batch)
        predictions = output["pred_future"][0].cpu().numpy()  # (K, T, 2)
        confidences = output["confidences"][0].cpu().numpy()  # (K,)
    return predictions, confidences


def compute_metrics(predictions, gt_future, confidences=None):
    """Compute minADE, minFDE, and miss rate for one sample.

    Args:
        predictions: (K, T, 2)
        gt_future: (T, 2)
        confidences: (K,) optional

    Returns:
        dict with minADE, minFDE, miss_rate, best_mode_idx
    """
    K = predictions.shape[0]
    ade_per_mode = np.zeros(K)
    fde_per_mode = np.zeros(K)

    for k in range(K):
        errors = np.linalg.norm(predictions[k] - gt_future, axis=1)
        ade_per_mode[k] = errors.mean()
        fde_per_mode[k] = errors[-1]

    best_idx = ade_per_mode.argmin()
    min_ade = ade_per_mode[best_idx]
    min_fde = fde_per_mode[best_idx]
    miss_rate = float(fde_per_mode.min() > 5.0)

    return {
        "minADE": min_ade,
        "minFDE": min_fde,
        "miss_rate": miss_rate,
        "best_mode_idx": best_idx,
        "ade_per_mode": ade_per_mode,
        "fde_per_mode": fde_per_mode,
    }


def draw_lane_structure(ax, scene, ego_pos, ego_heading, radius=70,
                        draw_waterflow=True, alpha_bg=0.25, alpha_wf=0.6):
    """Draw lane centerlines on the axes.

    Draws all nearby lanes as thin gray lines, and waterflow lanes slightly bolder.
    """
    lane_graph = scene["lane_graph"]

    # BEV convention: col0=lateral (right+), col1=forward
    # Plot convention: X=lateral (col0), Y=forward (col1) → no swap needed
    # All plots use (bev[:, 0], bev[:, 1]) so forward points UP on screen

    # Draw all lanes as thin background
    for lane_id, pts in lane_graph["lanes"].items():
        if pts is None or len(pts) < 2:
            continue
        pts_bev = _world_to_ego(pts[:, :2].astype(np.float64), ego_pos, ego_heading)
        if np.abs(pts_bev).max() > radius * 1.3:
            continue
        ax.plot(pts_bev[:, 0], pts_bev[:, 1],
                color=COLOR_LANE_BG, linewidth=1.0, alpha=0.35, zorder=1)

    # Draw waterflow lanes
    if draw_waterflow:
        sdc_xy = np.array([ego_pos[0], ego_pos[1]])
        ego_lane_id = find_ego_lane_id(sdc_xy, lane_graph, threshold=5.0)
        if ego_lane_id is not None:
            G, stages = build_waterflow_graph(lane_graph, ego_lane_id)
            stage_colors = [COLOR_LANE_EGO, "#43A047", "#FF8F00", COLOR_LANE_CONNECTED]
            for stage_idx, stage_nodes in enumerate(stages):
                color = stage_colors[min(stage_idx, len(stage_colors) - 1)]
                for lane_id in stage_nodes:
                    pts = lane_graph["lanes"].get(lane_id)
                    if pts is None or len(pts) < 2:
                        continue
                    pts_bev = _world_to_ego(
                        pts[:, :2].astype(np.float64), ego_pos, ego_heading
                    )
                    lw = 3.0 if stage_idx == 0 else 1.8
                    ax.plot(pts_bev[:, 0], pts_bev[:, 1],
                            color=color, linewidth=lw, alpha=0.7, zorder=2)


def draw_neighbors(ax, scene, ego_pos, ego_heading, anchor_frame=10, max_dist=50):
    """Draw neighbor vehicles as small gray squares."""
    av_idx = scene["av_idx"]
    for i, obj in enumerate(scene["objects"]):
        if i == av_idx or not obj["valid"][anchor_frame]:
            continue
        pos = obj["position"][anchor_frame]
        world_pt = np.array([float(pos["x"]), float(pos["y"])])
        ego_world = np.array([
            float(scene["objects"][av_idx]["position"][anchor_frame]["x"]),
            float(scene["objects"][av_idx]["position"][anchor_frame]["y"]),
        ])
        if np.linalg.norm(world_pt - ego_world) > max_dist:
            continue
        bev_pt = _world_to_ego(world_pt.reshape(1, 2), ego_pos, ego_heading)[0]
        ax.plot(bev_pt[0], bev_pt[1], "s", color=COLOR_NEIGHBOR,
                markersize=5, alpha=0.6, zorder=3)


def draw_traffic_lights(ax, scene, ego_pos, ego_heading, frame_idx=10):
    """Draw traffic light states at their stop points."""
    tls = scene.get("traffic_lights", [])
    if frame_idx >= len(tls):
        return
    tl_colors = {0: "#D32F2F", 1: "#FBC02D", 2: "#388E3C"}
    for tl in tls[frame_idx]:
        if not isinstance(tl, dict):
            continue
        sp = tl.get("stop_point")
        if sp is None:
            continue
        state = tl.get("state", -1)
        color = tl_colors.get(state, "#9E9E9E")
        pt = _world_to_ego(
            np.array([[sp["x"], sp["y"]]], dtype=np.float64), ego_pos, ego_heading
        )[0]
        ax.plot(pt[0], pt[1], marker="s", markersize=6, color=color,
                markeredgecolor="black", markeredgewidth=0.4, zorder=10)


def compute_dynamic_limits(gt_future, predictions, history, base_radius=70, margin=0.15):
    """Compute axis limits that include all trajectories with margin.

    Returns (x_min, x_max, y_min, y_max) in plot coordinates (lateral, forward).
    """
    # Plot coords: X=col0=lateral, Y=col1=forward
    all_x = [history[:, 0], gt_future[:, 0]]  # lateral
    all_y = [history[:, 1], gt_future[:, 1]]  # forward
    for k in range(predictions.shape[0]):
        all_x.append(predictions[k, :, 0])
        all_y.append(predictions[k, :, 1])

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)

    data_xmin, data_xmax = all_x.min(), all_x.max()
    data_ymin, data_ymax = all_y.min(), all_y.max()

    # Use at least base_radius, but expand if data exceeds it
    x_range = max(data_xmax - data_xmin, base_radius * 2)
    y_range = max(data_ymax - data_ymin, base_radius * 2)
    side = max(x_range, y_range) * (1 + margin)  # square + margin

    cx = (data_xmin + data_xmax) / 2
    cy = (data_ymin + data_ymax) / 2

    return cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2


def draw_single_panel(ax, scene, ego_pos, ego_heading, ego_traj_bev,
                      predictions, confidences, gt_future, history,
                      title, metrics, radius=70, show_lane_legend=False,
                      axis_limits=None):
    """Draw one panel of the side-by-side figure.

    Args:
        ax: matplotlib axes
        scene: raw scene dict
        ego_pos, ego_heading: for coordinate transforms
        ego_traj_bev: full ego trajectory in BEV
        predictions: (K, T, 2) predicted trajectories
        confidences: (K,) mode probabilities
        gt_future: (T, 2) ground truth future
        history: (H, 2) history trajectory
        title: panel title string
        metrics: dict from compute_metrics()
        radius: BEV plot radius
        show_lane_legend: whether to show lane topology legend
        axis_limits: tuple (xmin, xmax, ymin, ymax) or None for auto
    """
    if axis_limits is not None:
        ax.set_xlim(axis_limits[0], axis_limits[1])
        ax.set_ylim(axis_limits[2], axis_limits[3])
    else:
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
    ax.set_aspect("equal")
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, alpha=0.1, linewidth=0.3, color="#BDBDBD")

    # 1. Lane structure
    draw_lane_structure(ax, scene, ego_pos, ego_heading, radius=radius)

    # 2. Traffic lights
    draw_traffic_lights(ax, scene, ego_pos, ego_heading, frame_idx=ANCHOR_FRAME)

    # 3. Neighbors
    draw_neighbors(ax, scene, ego_pos, ego_heading, anchor_frame=ANCHOR_FRAME)

    # 4. Ego vehicle at origin — car points UP (forward = +y on screen)
    # Plot convention: X=lateral (col0), Y=forward (col1)
    # Car body: ~4.5m long (vertical/forward), ~2.0m wide (horizontal/lateral)
    ego_rect = mpatches.FancyBboxPatch(
        (-1.0, -2.25), 2.0, 4.5,
        boxstyle="round,pad=0.15", facecolor=COLOR_SDC,
        edgecolor="white", linewidth=1.2, alpha=0.9, zorder=15
    )
    ax.add_patch(ego_rect)

    # Forward direction indicator — large white chevron/arrow pointing up (forward)
    arrow_tri = plt.Polygon(
        [(-0.8, 0.5), (0.8, 0.5), (0.0, 3.5)],
        closed=True, facecolor="white", edgecolor="#0D47A1",
        linewidth=1.0, alpha=0.95, zorder=16,
    )
    ax.add_patch(arrow_tri)
    # Additional thin arrow extending above the car to show heading clearly
    ax.annotate("", xy=(0, 6.0), xytext=(0, 3.0),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5,
                                mutation_scale=18),
                zorder=17)

    # 5. History trajectory (X=lateral col0, Y=forward col1)
    ax.plot(history[:, 0], history[:, 1], "-", color=COLOR_HISTORY,
            linewidth=2.5, alpha=0.9, zorder=8)
    ax.plot(history[-1, 0], history[-1, 1], "o", color=COLOR_HISTORY,
            markersize=5, zorder=9)

    # 6. K predicted modes (sorted by confidence, draw lowest first)
    sorted_modes = np.argsort(confidences)  # ascending confidence
    for rank, k in enumerate(sorted_modes):
        color = MODE_COLORS[k % len(MODE_COLORS)]
        conf = confidences[k]
        is_best = (k == metrics["best_mode_idx"])

        lw = 3.5 if is_best else 2.0
        alpha = 0.95 if is_best else 0.5 + 0.3 * conf
        zorder = 12 if is_best else 6

        pred_k = predictions[k]
        ax.plot(pred_k[:, 0], pred_k[:, 1], "-", color=color,
                linewidth=lw, alpha=alpha, zorder=zorder)
        # Endpoint marker
        ax.plot(pred_k[-1, 0], pred_k[-1, 1], "o", color=color,
                markersize=6 if is_best else 3.5, alpha=alpha, zorder=zorder)

    # 7. Ground truth future — BOLD, with red star endpoint
    ax.plot(gt_future[:, 0], gt_future[:, 1], "--", color=COLOR_GT,
            linewidth=3.0, alpha=0.95, zorder=13,
            dashes=(5, 3),
            path_effects=[pe.Stroke(linewidth=4.5, foreground="white"), pe.Normal()])
    # Red star endpoint — large, high zorder, white edge for contrast
    ax.plot(gt_future[-1, 0], gt_future[-1, 1], "*", color="#D32F2F",
            markersize=28, zorder=100, markeredgecolor="white", markeredgewidth=1.5,
            path_effects=[pe.Stroke(linewidth=2.0, foreground="black"), pe.Normal()])

    # 8. Title with metrics — larger fonts
    min_ade = metrics["minADE"]
    min_fde = metrics["minFDE"]
    ax.set_title(
        f"{title}\nminADE={min_ade:.2f}m   minFDE={min_fde:.2f}m",
        fontsize=14, fontweight="bold", pad=10,
    )

    ax.set_xlabel("Lateral (m)", fontsize=11)
    ax.set_ylabel("Forward (m)", fontsize=11)
    ax.tick_params(labelsize=9)


def generate_comparison_figure(
    scene_path, bl_model, lc_model,
    sample_idx=0, save_path=None, device="cpu",
):
    """Generate a single side-by-side comparison figure.

    Returns:
        dict with metrics for both models, or None if scene is invalid.
    """
    scene = load_scene(scene_path)
    result = build_sample(
        scene, anchor_frame=ANCHOR_FRAME,
        history_len=HISTORY_LEN, future_len=FUTURE_LEN,
        include_lanes=True,
    )
    sample, ego_pos, ego_heading, ego_traj_bev, lane_data = result

    if sample is None:
        return None

    # Extract history and GT future
    h_start = ANCHOR_FRAME - (HISTORY_LEN - 1)
    h_end = ANCHOR_FRAME + 1
    f_start = ANCHOR_FRAME + 1
    f_end = ANCHOR_FRAME + 1 + FUTURE_LEN

    history = ego_traj_bev[h_start:h_end]
    gt_future = ego_traj_bev[f_start:f_end]

    if len(gt_future) < FUTURE_LEN:
        return None

    # Filter: GT endpoint must be within a reasonable range for visualization
    gt_endpoint = gt_future[-1]  # (2,) in BEV: [forward, lateral]
    gt_endpoint_dist = np.linalg.norm(gt_endpoint)
    if gt_endpoint_dist > BEV_RADIUS * 0.85:
        return None  # GT endpoint would be clipped or marginal

    # Run inference
    bl_preds, bl_confs = predict(bl_model, sample, device)
    lc_preds, lc_confs = predict(lc_model, sample, device)

    bl_metrics = compute_metrics(bl_preds, gt_future, bl_confs)
    lc_metrics = compute_metrics(lc_preds, gt_future, lc_confs)

    # Compute shared dynamic axis limits across both panels
    all_preds = np.concatenate([bl_preds, lc_preds], axis=0)  # (2K, T, 2)
    axis_limits = compute_dynamic_limits(
        gt_future, all_preds, history, base_radius=BEV_RADIUS, margin=0.15
    )

    # Create figure
    fig, (ax_bl, ax_lc) = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.subplots_adjust(wspace=0.25)

    draw_single_panel(
        ax_bl, scene, ego_pos, ego_heading, ego_traj_bev,
        bl_preds, bl_confs, gt_future, history,
        title="LSTM Baseline",
        metrics=bl_metrics,
        radius=BEV_RADIUS,
        axis_limits=axis_limits,
    )

    draw_single_panel(
        ax_lc, scene, ego_pos, ego_heading, ego_traj_bev,
        lc_preds, lc_confs, gt_future, history,
        title="LSTM + Lane Conditioning",
        metrics=lc_metrics,
        radius=BEV_RADIUS,
        axis_limits=axis_limits,
    )

    # Shared legend at the bottom — larger, bolder
    legend_elements = [
        plt.Line2D([0], [0], color=COLOR_HISTORY, linewidth=2.5, label="History (1.1s)"),
        plt.Line2D([0], [0], color=COLOR_GT, linewidth=3, linestyle="--",
                    dashes=(5, 3), label="Ground Truth (8.0s)"),
        plt.Line2D([0], [0], marker="*", color="#D32F2F", linewidth=0,
                    markersize=14, markeredgecolor="white", markeredgewidth=1.0,
                    label="GT Endpoint"),
    ]
    for k in range(NUM_MODES):
        legend_elements.append(
            plt.Line2D([0], [0], color=MODE_COLORS[k], linewidth=2.0,
                        label=f"Mode {k+1}")
        )
    legend_elements.extend([
        plt.Line2D([0], [0], color=COLOR_LANE_EGO, linewidth=2.5,
                    alpha=0.7, label="Ego Lane"),
        plt.Line2D([0], [0], color="#43A047", linewidth=1.8,
                    alpha=0.7, label="Connected Lanes"),
        plt.Line2D([0], [0], color=COLOR_LANE_BG, linewidth=1.0,
                    alpha=0.5, label="Background Lanes"),
    ])

    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=6, fontsize=9.5, framealpha=0.95,
        bbox_to_anchor=(0.5, -0.02),
    )

    # ADE improvement annotation
    ade_improve = (bl_metrics["minADE"] - lc_metrics["minADE"]) / bl_metrics["minADE"] * 100
    fde_improve = (bl_metrics["minFDE"] - lc_metrics["minFDE"]) / bl_metrics["minFDE"] * 100

    improvement_text = (
        f"LC improvement: minADE {ade_improve:+.1f}%  |  minFDE {fde_improve:+.1f}%"
    )
    fig.suptitle(
        f"Qualitative Trajectory Comparison (8s prediction, K=6 modes)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.text(0.5, -0.04, improvement_text, ha="center", fontsize=11,
             fontweight="bold", style="italic", color="#333333")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Saved: {save_path}")
    plt.close(fig)

    return {
        "bl_metrics": bl_metrics,
        "lc_metrics": lc_metrics,
        "ade_improvement_pct": ade_improve,
        "fde_improvement_pct": fde_improve,
        "scene_path": scene_path,
    }


def main():
    print("=" * 70)
    print("Generating Qualitative Paper Figures")
    print("  BL vs LC  |  Multi-modal K=6  |  8s prediction")
    print("=" * 70)

    device = "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load models
    print("\n[1/4] Loading model checkpoints...")
    bl_model = load_model(BL_CKPT, device=device)
    lc_model = load_model(LC_CKPT, device=device)
    print("  Both models loaded successfully.\n")

    # 2. Get validation scene paths (same split as training)
    print("[2/4] Loading validation scene list...")
    with open(SCENE_LIST, "r") as f:
        all_scenes = [line.strip() for line in f if line.strip()]
    all_scenes = [p for p in all_scenes if os.path.exists(p)]

    # Reproduce the same val split used in training (seed=42, val_ratio=0.15)
    rng = random.Random(42)
    indices = list(range(len(all_scenes)))
    rng.shuffle(indices)
    n_val = max(1, int(len(all_scenes) * 0.15))
    val_indices = indices[:n_val]
    val_scenes = [all_scenes[i] for i in val_indices]
    print(f"  Validation set: {len(val_scenes)} scenes")

    # 3. Sample random scenes and generate figures
    print(f"\n[3/4] Generating {NUM_SAMPLES} comparison figures...")
    rng2 = random.Random(SEED)
    sampled_indices = rng2.sample(range(len(val_scenes)), min(NUM_SAMPLES * 3, len(val_scenes)))

    results = []
    fig_count = 0
    for scene_idx in sampled_indices:
        if fig_count >= NUM_SAMPLES:
            break

        scene_path = val_scenes[scene_idx]
        save_path = os.path.join(OUTPUT_DIR, f"qual_comparison_{fig_count:03d}.png")

        try:
            result = generate_comparison_figure(
                scene_path, bl_model, lc_model,
                sample_idx=fig_count, save_path=save_path, device=device,
            )
            if result is not None:
                results.append(result)
                fig_count += 1
                bl_m = result["bl_metrics"]
                lc_m = result["lc_metrics"]
                print(f"    [{fig_count:2d}/{NUM_SAMPLES}] BL: minADE={bl_m['minADE']:.2f}, "
                      f"LC: minADE={lc_m['minADE']:.2f}, "
                      f"improvement: {result['ade_improvement_pct']:+.1f}%")
        except Exception as e:
            print(f"    Skipped scene {scene_idx}: {e}")
            continue

    # 4. Select best 3 figures (where LC wins most over BL)
    print(f"\n[4/4] Selecting best 3 figures where LC outperforms BL the most...")

    # Sort by ADE improvement (largest positive = LC wins most)
    results_sorted = sorted(results, key=lambda r: r["ade_improvement_pct"], reverse=True)

    os.makedirs(PAPER_DIR, exist_ok=True)
    print("\n  Top 3 selected:")
    for rank, r in enumerate(results_sorted[:3]):
        src = os.path.join(OUTPUT_DIR, f"qual_comparison_{results.index(r):03d}.png")
        dst = os.path.join(PAPER_DIR, f"fig_qualitative_{rank + 1}.png")
        shutil.copy2(src, dst)
        bl_m = r["bl_metrics"]
        lc_m = r["lc_metrics"]
        print(f"    #{rank+1}: ADE improve={r['ade_improvement_pct']:+.1f}%, "
              f"FDE improve={r['fde_improvement_pct']:+.1f}% "
              f"(BL: {bl_m['minADE']:.2f}/{bl_m['minFDE']:.2f}, "
              f"LC: {lc_m['minADE']:.2f}/{lc_m['minFDE']:.2f})")
        print(f"         -> {dst}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary across all generated figures:")
    ade_improvements = [r["ade_improvement_pct"] for r in results]
    fde_improvements = [r["fde_improvement_pct"] for r in results]
    bl_ades = [r["bl_metrics"]["minADE"] for r in results]
    lc_ades = [r["lc_metrics"]["minADE"] for r in results]
    print(f"  BL avg minADE: {np.mean(bl_ades):.3f} +/- {np.std(bl_ades):.3f}")
    print(f"  LC avg minADE: {np.mean(lc_ades):.3f} +/- {np.std(lc_ades):.3f}")
    print(f"  ADE improvement: {np.mean(ade_improvements):+.1f}% +/- {np.std(ade_improvements):.1f}%")
    print(f"  FDE improvement: {np.mean(fde_improvements):+.1f}% +/- {np.std(fde_improvements):.1f}%")
    print(f"  LC wins: {sum(1 for x in ade_improvements if x > 0)}/{len(ade_improvements)} scenes")
    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print(f"Best 3 copied to: {PAPER_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
