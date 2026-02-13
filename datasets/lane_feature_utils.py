"""Lane feature extraction for trajectory prediction.

Extracts local lane graph features from the waterflow graph
(3-hop BFS from ego lane) for conditioning trajectory prediction models.
"""

import math

import numpy as np

from tools.lane_graph.lane_explorer import build_waterflow_graph, find_ego_lane_id
from utils.data import resample_polyline


def _world_to_ego(points, ego_pos, ego_heading_deg):
    """Transform world coordinates to ego-centric BEV frame."""
    heading_rad = math.radians(ego_heading_deg)
    adjusted = heading_rad - np.pi / 2
    dxdy = points - np.array(ego_pos, dtype=np.float64)
    c, s = np.cos(-adjusted), np.sin(-adjusted)
    R = np.array([[c, -s], [s, c]])
    return (dxdy @ R.T).astype(np.float32)


def extract_lane_features(
    scene,
    ego_pos,
    ego_heading,
    max_lanes=16,
    lane_points=10,
):
    """Extract structured lane features from the waterflow graph.

    Builds a local lane graph centered on the ego vehicle using 3-hop BFS,
    then encodes each lane as a fixed-size feature vector.

    Args:
        scene: dict, raw scene data with lane_graph, traffic_lights, etc.
        ego_pos: (x, y) tuple, ego position in world coordinates
        ego_heading: float, ego heading in degrees
        max_lanes: int, maximum number of lanes to include
        lane_points: int, number of points to resample each centerline to

    Returns:
        dict with:
            lane_features: (max_lanes, feat_dim) float32
            lane_adj: (max_lanes, max_lanes) float32
            lane_mask: (max_lanes,) float32
            ego_lane_idx: int, index of ego lane in the array (-1 if not found)
            lane_centerlines_bev: (max_lanes, lane_points, 2) float32, for visualization
    """
    lane_graph = scene["lane_graph"]
    sdc_xy = np.array([ego_pos[0], ego_pos[1]])

    # Find ego lane
    ego_lane_id = find_ego_lane_id(sdc_xy, lane_graph, threshold=5.0)

    # Build waterflow graph (3-hop BFS from ego)
    if ego_lane_id is not None:
        G, stages = build_waterflow_graph(lane_graph, ego_lane_id)
        lane_ids = list(G.nodes)
    else:
        # Fallback: use nearest lanes by distance
        lane_ids = _get_nearest_lanes(sdc_xy, lane_graph, max_lanes)
        G = None

    # Collect traffic light controlled lanes
    traffic_light_lanes = set()
    for tl_frame in scene.get("traffic_lights", []):
        if isinstance(tl_frame, list):
            for tl in tl_frame:
                if isinstance(tl, dict) and "lane" in tl:
                    traffic_light_lanes.add(tl["lane"])

    # Collect stop sign lanes
    stop_sign_lanes = set()
    for ss in lane_graph.get("stop_signs", []):
        if isinstance(ss, dict) and "lane" in ss:
            stop_sign_lanes.add(ss["lane"])

    # Truncate to max_lanes
    lane_ids = lane_ids[:max_lanes]

    # Per-lane feature: centerline_flat(lane_points*2) + direction(2) + length(1) + flags(3)
    feat_dim = lane_points * 2 + 2 + 1 + 3
    lane_features = np.zeros((max_lanes, feat_dim), dtype=np.float32)
    lane_centerlines_bev = np.zeros((max_lanes, lane_points, 2), dtype=np.float32)
    lane_mask = np.zeros(max_lanes, dtype=np.float32)
    lane_adj = np.zeros((max_lanes, max_lanes), dtype=np.float32)
    ego_lane_idx = -1

    # Map lane_id -> array index
    id_to_idx = {}

    for i, lane_id in enumerate(lane_ids):
        centerline = lane_graph["lanes"].get(lane_id)
        if centerline is None or len(centerline) < 2:
            continue

        # Get centerline points (may be (N, 2) or (N, 3+))
        pts = centerline[:, :2].astype(np.float64)

        # Transform to BEV
        pts_bev = _world_to_ego(pts, ego_pos, ego_heading)

        # Resample to fixed number of points
        if len(pts_bev) >= 2:
            pts_resampled = resample_polyline(pts_bev, num_points=lane_points)
        else:
            pts_resampled = np.zeros((lane_points, 2), dtype=np.float32)

        lane_centerlines_bev[i] = pts_resampled

        # Compute direction vector (start to end, normalized)
        direction = pts_resampled[-1] - pts_resampled[0]
        norm = np.linalg.norm(direction) + 1e-6
        direction = direction / norm

        # Compute lane length
        segments = np.diff(pts_resampled, axis=0)
        length = np.sum(np.linalg.norm(segments, axis=1))

        # Boolean flags
        is_ego_lane = 1.0 if lane_id == ego_lane_id else 0.0
        has_traffic_light = 1.0 if lane_id in traffic_light_lanes else 0.0
        has_stop_sign = 1.0 if lane_id in stop_sign_lanes else 0.0

        # Build feature vector
        feat = np.concatenate([
            pts_resampled.flatten(),          # (lane_points * 2,)
            direction,                         # (2,)
            [length / 50.0],                   # (1,) normalized by 50m
            [is_ego_lane, has_traffic_light, has_stop_sign],  # (3,)
        ])
        lane_features[i] = feat
        lane_mask[i] = 1.0

        if lane_id == ego_lane_id:
            ego_lane_idx = i

        id_to_idx[lane_id] = i

    # Build adjacency matrix from waterflow graph edges
    if G is not None:
        for u, v in G.edges:
            if u in id_to_idx and v in id_to_idx:
                lane_adj[id_to_idx[u], id_to_idx[v]] = 1.0
                lane_adj[id_to_idx[v], id_to_idx[u]] = 1.0  # undirected for attention

    return {
        "lane_features": lane_features,              # (max_lanes, feat_dim)
        "lane_adj": lane_adj,                        # (max_lanes, max_lanes)
        "lane_mask": lane_mask,                      # (max_lanes,)
        "ego_lane_idx": ego_lane_idx,                # int
        "lane_centerlines_bev": lane_centerlines_bev,  # (max_lanes, lane_points, 2)
    }


def assign_traj_to_lanes(
    future_traj_bev,
    lane_centerlines_bev,
    lane_mask,
    threshold=5.0,
):
    """Assign each future trajectory point to the nearest lane.

    For dual supervision (Phase 3): maps GT trajectory to lane index sequence.

    Args:
        future_traj_bev: (T, 2) future trajectory positions in BEV
        lane_centerlines_bev: (max_lanes, lane_points, 2) lane centerlines in BEV
        lane_mask: (max_lanes,) which lanes are valid
        threshold: max distance (m) to assign a lane

    Returns:
        lane_ids: (T,) int array, lane index for each timestep (-1 if no match)
        valid_mask: (T,) float32, 1.0 where assignment is valid
    """
    T = future_traj_bev.shape[0]
    lane_ids = np.full(T, -1, dtype=np.int64)
    valid_mask = np.zeros(T, dtype=np.float32)

    n_lanes = int(lane_mask.sum())
    if n_lanes == 0:
        return lane_ids, valid_mask

    for t in range(T):
        pt = future_traj_bev[t]  # (2,)
        best_dist = threshold
        best_idx = -1

        for i in range(lane_centerlines_bev.shape[0]):
            if lane_mask[i] < 0.5:
                continue
            # Distance from point to nearest centerline point
            dists = np.linalg.norm(lane_centerlines_bev[i] - pt, axis=1)
            min_dist = dists.min()
            if min_dist < best_dist:
                best_dist = min_dist
                best_idx = i

        if best_idx >= 0:
            lane_ids[t] = best_idx
            valid_mask[t] = 1.0

    return lane_ids, valid_mask


def _get_nearest_lanes(sdc_xy, lane_graph, max_lanes):
    """Fallback: get nearest lanes by distance when ego lane not found."""
    distances = []
    for lane_id, pts in lane_graph["lanes"].items():
        if pts is None or len(pts) < 2:
            continue
        dists = np.linalg.norm(pts[:, :2] - sdc_xy, axis=1)
        distances.append((lane_id, dists.min()))

    distances.sort(key=lambda x: x[1])
    return [lid for lid, _ in distances[:max_lanes]]
