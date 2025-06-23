import logging
import math

import numpy as np
from jynxzzzdebug import debug_print, setup_logger

logging = setup_logger("ego_utils", "logs/ego_utils.log")


def build_local_transform(ego_pos, heading_deg):
    logging.info(
        f"[DEBUG] Building local transform for ego at {ego_pos} with heading {heading_deg}°"
    )

    def w2e(points):
        return world_to_ego(points, ego_pos, heading_deg)

    return w2e


def world_to_ego(points, ego_pos, ego_heading_deg):
    heading_rad = math.radians(ego_heading_deg)
    adjusted_heading = heading_rad - np.pi / 2  # Waymo heading: 0° = 北，转成 X+ 朝前

    dxdy = points - np.array(ego_pos)
    c, s = np.cos(-adjusted_heading), np.sin(-adjusted_heading)
    R = np.array([[c, -s], [s, c]])
    return dxdy @ R.T


def extract_ego_info(scene, frame_idx):
    debug_print("extract_ego_info", "start extracting ego info")

    ego = scene["objects"][scene["av_idx"]]

    # === 提取坐标 ===
    ego_pos = (
        float(ego["position"][frame_idx]["x"]),
        float(ego["position"][frame_idx]["y"]),
    )

    # === 提取 heading ===
    heading_raw = ego["heading"][frame_idx]
    if isinstance(heading_raw, tuple):
        print(
            f"[⚠️ WARNING] heading is tuple at frame {frame_idx}, value = {heading_raw}"
        )
        heading = float(heading_raw[0])
    elif isinstance(heading_raw, list):
        print(
            f"[⚠️ WARNING] heading is list at frame {frame_idx}, value = {heading_raw}"
        )
        heading = float(heading_raw[0])
    else:
        heading = float(heading_raw)

    print(f"[DEBUG] Frame {frame_idx} heading: {heading:.2f}, type: {type(heading)}")
    return ego, ego_pos, heading


def transform_all_trajectories(scene, w2e, frame_idx):
    """
    对所有车辆轨迹进行坐标变换（世界坐标 → ego 坐标）
    """
    debug_print("transform_all_trajectories", "start transforming trajectories")
    debug_print("transform_all_trajectories", f"Frame index: {type(frame_idx)}")

    # 注意：这里重新构造 w2e（已带 ego pose + heading）
    ego, ego_pos, ego_heading_deg = extract_ego_info(scene, frame_idx)
    w2e = build_local_transform(ego_pos, ego_heading_deg)

    for obj in scene["objects"]:
        new_traj = []
        for pos, valid in zip(obj["position"], obj["valid"]):
            if not valid:
                new_traj.append([0.0, 0.0])
                continue
            pt = [pos["x"], pos["y"]]
            pt_local = w2e([pt])[0]  # ✅ 正确处理单点
            new_traj.append(pt_local)
        obj["position_bev"] = new_traj  # 🚗 存到新的字段


def transform_lane_graph_to_bev(lane_graph, w2e):
    """
    将 lane_graph["lanes"] 中每一条中心线进行 BEV 坐标变换
    """
    new_lanes = {}

    for lane_id, pts in lane_graph.get("lanes", {}).items():
        if not isinstance(pts, list) or len(pts) == 0:
            continue

        # 提取坐标 [[x1, y1], [x2, y2], ...]
        xy = [[p["x"], p["y"]] for p in pts]
        bev_xy = w2e(xy)  # 一次性转化多个点
        new_pts = []

        for p, new_p in zip(pts, bev_xy):
            p_new = dict(p)
            p_new["x"], p_new["y"] = new_p[0], new_p[1]
            new_pts.append(p_new)

        new_lanes[lane_id] = new_pts

    return {
        **lane_graph,
        "lanes": new_lanes,
    }
