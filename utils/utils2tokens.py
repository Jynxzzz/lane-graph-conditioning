from math import pi

import numpy as np


def angle2token(vec, bins=16):
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return -1  # 表示无法判断方向
    angle = np.arctan2(vec[1], vec[0])
    return int(((angle + np.pi) / (2 * np.pi)) * bins) % bins


def compute_lane_heading(scenario, lane_id):
    lane_pts = scenario["lane_graph"]["lanes"].get(lane_id)
    if lane_pts is None or len(lane_pts) < 2:
        return np.array([0.0, 0.0])  # fallback：无法计算方向

    vec = lane_pts[-1, :2] - lane_pts[0, :2]
    return vec / np.linalg.norm(vec)  # 返回单位向量，便于角度计算
