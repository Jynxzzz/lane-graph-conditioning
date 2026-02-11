import math
import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset


def _world_to_ego(points, ego_pos, ego_heading_deg):
    """Transform world coordinates to ego-centric BEV frame.

    Args:
        points: (N, 2) numpy array in world coordinates
        ego_pos: (x, y) tuple of ego position in world
        ego_heading_deg: ego heading in degrees (Waymo convention: 0=North)

    Returns:
        (N, 2) numpy array in ego frame (X+ forward, Y+ left)
    """
    heading_rad = math.radians(ego_heading_deg)
    adjusted = heading_rad - np.pi / 2
    dxdy = points - np.array(ego_pos, dtype=np.float64)
    c, s = np.cos(-adjusted), np.sin(-adjusted)
    R = np.array([[c, -s], [s, c]])
    return (dxdy @ R.T).astype(np.float32)


def _extract_ego_info(scene, frame_idx):
    """Extract ego position and heading at a given frame."""
    ego = scene["objects"][scene["av_idx"]]
    pos = ego["position"][frame_idx]
    ego_pos = (float(pos["x"]), float(pos["y"]))

    heading_raw = ego["heading"][frame_idx]
    if isinstance(heading_raw, (tuple, list)):
        heading = float(heading_raw[0])
    else:
        heading = float(heading_raw)

    return ego_pos, heading


def _extract_full_trajectory(obj):
    """Extract full (91, 2) trajectory and validity mask from an object."""
    n_frames = len(obj["position"])
    traj = np.zeros((n_frames, 2), dtype=np.float32)
    valid = np.zeros(n_frames, dtype=bool)

    for i in range(n_frames):
        if obj["valid"][i]:
            traj[i, 0] = float(obj["position"][i]["x"])
            traj[i, 1] = float(obj["position"][i]["y"])
            valid[i] = True

    return traj, valid


class TrajectoryPredictionDataset(Dataset):
    """Waymo trajectory prediction dataset.

    Loads pkl scene files, extracts ego and neighbor trajectories in BEV frame.
    Supports multi-anchor augmentation (multiple time windows per scene).

    Args:
        scene_list_path: path to txt file with absolute paths to pkl files
        split: "train" or "val"
        val_ratio: fraction of scenes for validation
        history_len: number of history frames (including current)
        future_len: number of future frames to predict
        max_neighbors: max number of neighbor agents to include
        neighbor_distance: max distance (m) to consider a neighbor
        anchor_frames: list of candidate anchor frame indices
        augment: whether to apply data augmentation
        seed: random seed for reproducible train/val split
        include_lanes: whether to extract lane graph features (Phase 2+)
        max_lanes: max number of lanes in waterflow graph
        lane_points: number of points per lane centerline
        include_lane_ids: whether to compute GT lane_id assignments (Phase 3)
    """

    def __init__(
        self,
        scene_list_path,
        split="train",
        val_ratio=0.15,
        history_len=11,
        future_len=30,
        max_neighbors=10,
        neighbor_distance=30.0,
        anchor_frames=None,
        augment=False,
        seed=42,
        include_lanes=False,
        max_lanes=16,
        lane_points=10,
        include_lane_ids=False,
        expand_anchors=False,
        augment_rotation=False,
    ):
        # Load scene list (absolute paths)
        with open(scene_list_path, "r") as f:
            all_scenes = [line.strip() for line in f if line.strip()]

        # Filter to existing files
        all_scenes = [p for p in all_scenes if os.path.exists(p)]

        # Deterministic train/val split
        rng = random.Random(seed)
        indices = list(range(len(all_scenes)))
        rng.shuffle(indices)
        n_val = max(1, int(len(all_scenes) * val_ratio))

        if split == "val":
            selected = indices[:n_val]
        else:
            selected = indices[n_val:]

        self.scenes = [all_scenes[i] for i in selected]
        self.split = split
        self.history_len = history_len
        self.future_len = future_len
        self.max_neighbors = max_neighbors
        self.neighbor_distance = neighbor_distance
        self.anchor_frames = anchor_frames or [10, 20, 30, 40, 50]
        self.augment = augment and (split == "train")
        self.augment_rotation = augment_rotation
        self.include_lanes = include_lanes
        self.max_lanes = max_lanes
        self.lane_points = lane_points
        self.include_lane_ids = include_lane_ids

        # Expand anchors: create one sample per (scene, anchor) pair for training
        if expand_anchors and self.split == "train":
            self.samples = []
            for si in range(len(self.scenes)):
                for anchor in self.anchor_frames:
                    self.samples.append((si, anchor))
        else:
            # Original behavior: one sample per scene, random/middle anchor
            self.samples = [(i, None) for i in range(len(self.scenes))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene_idx, fixed_anchor = self.samples[idx]

        # Load scene
        with open(self.scenes[scene_idx], "rb") as f:
            scene = pickle.load(f)

        n_frames = len(scene["objects"][0]["position"])
        av_idx = scene["av_idx"]

        # Pick anchor frame
        valid_anchors = [
            a for a in self.anchor_frames
            if a >= self.history_len - 1 and a + self.future_len < n_frames
        ]
        if not valid_anchors:
            valid_anchors = [self.history_len - 1]

        if fixed_anchor is not None and fixed_anchor in valid_anchors:
            anchor = fixed_anchor
        elif self.split == "train":
            anchor = random.choice(valid_anchors)
        else:
            anchor = valid_anchors[len(valid_anchors) // 2]  # deterministic for val

        # Extract ego trajectory
        ego_obj = scene["objects"][av_idx]
        ego_traj_world, ego_valid = _extract_full_trajectory(ego_obj)

        # Get ego pose at anchor frame for BEV transform
        ego_pos, ego_heading = _extract_ego_info(scene, anchor)

        # Transform ego trajectory to BEV
        ego_traj_bev = _world_to_ego(ego_traj_world, ego_pos, ego_heading)

        # Split history and future
        h_start = anchor - (self.history_len - 1)
        h_end = anchor + 1
        f_start = anchor + 1
        f_end = anchor + 1 + self.future_len

        sdc_history = ego_traj_bev[h_start:h_end].copy()  # (11, 2)
        sdc_future = ego_traj_bev[f_start:f_end].copy()    # (30, 2)
        sdc_history_valid = ego_valid[h_start:h_end].copy()
        sdc_future_valid = ego_valid[f_start:f_end].copy()

        # Check we have enough valid data
        if not sdc_history_valid.all() or not sdc_future_valid.all():
            # Fill invalid with linear interpolation or zeros
            sdc_history[~sdc_history_valid] = 0.0
            sdc_future[~sdc_future_valid] = 0.0

        # Extract neighbor trajectories
        ego_world_pos = ego_traj_world[anchor]
        neighbor_history = np.zeros(
            (self.max_neighbors, self.history_len, 2), dtype=np.float32
        )
        neighbor_mask = np.zeros(self.max_neighbors, dtype=np.float32)

        neighbor_count = 0
        for i, obj in enumerate(scene["objects"]):
            if i == av_idx or neighbor_count >= self.max_neighbors:
                break
            if not obj["valid"][anchor]:
                continue

            obj_pos = obj["position"][anchor]
            dx = float(obj_pos["x"]) - ego_world_pos[0]
            dy = float(obj_pos["y"]) - ego_world_pos[1]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist > self.neighbor_distance:
                continue

            obj_traj_world, obj_valid = _extract_full_trajectory(obj)
            obj_traj_bev = _world_to_ego(obj_traj_world, ego_pos, ego_heading)

            n_hist = obj_traj_bev[h_start:h_end].copy()
            n_valid = obj_valid[h_start:h_end]
            n_hist[~n_valid] = 0.0

            neighbor_history[neighbor_count] = n_hist
            neighbor_mask[neighbor_count] = 1.0
            neighbor_count += 1

        # Extract lane features (Phase 2+)
        lane_data = None
        if self.include_lanes:
            from datasets.trajectory.lane_feature_utils import (
                assign_traj_to_lanes,
                extract_lane_features,
            )
            lane_data = extract_lane_features(
                scene, ego_pos, ego_heading,
                max_lanes=self.max_lanes,
                lane_points=self.lane_points,
            )

        # Data augmentation: random rotation (replaces left-right flip)
        if self.augment and self.augment_rotation:
            angle = random.uniform(-np.pi, np.pi)
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)

            sdc_history = sdc_history @ R.T
            sdc_future = sdc_future @ R.T
            neighbor_history = (
                neighbor_history.reshape(-1, 2) @ R.T
            ).reshape(neighbor_history.shape)

            if lane_data is not None:
                # Rotate centerline points in features (first lane_points*2 values)
                lp2 = self.lane_points * 2
                pts = lane_data["lane_features"][:, :lp2].reshape(-1, 2)
                lane_data["lane_features"][:, :lp2] = (pts @ R.T).reshape(-1, lp2)
                # Rotate direction vector (next 2 values)
                lane_data["lane_features"][:, lp2:lp2+2] = (
                    lane_data["lane_features"][:, lp2:lp2+2] @ R.T
                )
                # Rotate centerlines_bev
                cl = lane_data["lane_centerlines_bev"]
                lane_data["lane_centerlines_bev"] = (
                    cl.reshape(-1, 2) @ R.T
                ).reshape(cl.shape)
        elif self.augment and random.random() < 0.5:
            # Fallback: left-right flip if rotation not enabled
            sdc_history[:, 1] *= -1
            sdc_future[:, 1] *= -1
            neighbor_history[:, :, 1] *= -1
            if lane_data is not None:
                lane_data["lane_features"][:, 1::2] *= -1
                lane_data["lane_centerlines_bev"][:, :, 1] *= -1

        # Small noise on history (training only)
        if self.augment:
            sdc_history += np.random.randn(*sdc_history.shape).astype(np.float32) * 0.1

        sample = {
            "sdc_history": torch.from_numpy(sdc_history),       # (11, 2)
            "sdc_future": torch.from_numpy(sdc_future),         # (30, 2)
            "neighbor_history": torch.from_numpy(neighbor_history),  # (10, 11, 2)
            "neighbor_mask": torch.from_numpy(neighbor_mask),    # (10,)
        }

        # Add lane features to sample
        if lane_data is not None:
            sample["lane_features"] = torch.from_numpy(lane_data["lane_features"])
            sample["lane_adj"] = torch.from_numpy(lane_data["lane_adj"])
            sample["lane_mask"] = torch.from_numpy(lane_data["lane_mask"])
            sample["ego_lane_idx"] = torch.tensor(lane_data["ego_lane_idx"], dtype=torch.long)
            sample["lane_centerlines_bev"] = torch.from_numpy(
                lane_data["lane_centerlines_bev"]
            )  # (max_lanes, lane_points, 2)

            # Phase 3: GT lane ID assignment for dual supervision
            if self.include_lane_ids:
                lane_ids, lane_valid = assign_traj_to_lanes(
                    sdc_future,
                    lane_data["lane_centerlines_bev"],
                    lane_data["lane_mask"],
                )
                sample["lane_id_sequence"] = torch.from_numpy(lane_ids)
                sample["lane_id_mask"] = torch.from_numpy(lane_valid)

        return sample


def collate_fn(batch):
    """Filter None samples and stack."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)
