# trajectory_diffusion_dataset.py
import os
import pickle
from copy import deepcopy

import numpy as np
import torch
from _dev.encoder_debug import encode_lanes_debug
from datasets.diffusion.components.extract_utils import extract_sdc_and_neighbors
from datasets.diffusion.components.lane_utils import (
    build_local_transform,
    extract_ego_info,
    transform_all_trajectories,
    transform_lane_graph_to_bev,
)
from jynxzzzdebug import debug_print, setup_logger
from tools.coordinate_utils import world_to_ego
from tools.lane_graph.lane_explorer import find_ego_lane_id, get_lane_traversal
from torch.utils.data import Dataset


def build_rotation_matrix(theta_rad):
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def build_local_transform(ego_pos, ego_heading):
    # 世界 → ego 坐标（BEV）
    T = -np.array(ego_pos)
    R = build_rotation_matrix(-ego_heading)
    return R, T


def apply_transform(traj, R, T):
    traj = traj - T  # 平移
    return traj @ R.T  # 旋转


# === Stage 1: 世界坐标 → BEV 局部坐标 ===
class ToBEV:
    def __init__(self, frame_idx=0):
        self.frame_idx = frame_idx

    def __call__(self, sample):
        debug_print(
            "transform",
            f"[DEBUG] frame_idx type: {type(self.frame_idx)} value: {self.frame_idx}",
        )

        # === 深拷贝 scene，避免污染原始数据 ===
        scene = deepcopy(sample["scenario"])

        # === 提取 SDC + 邻车轨迹（世界坐标系）===
        extract_result = extract_sdc_and_neighbors(scene, frame_idx=self.frame_idx)
        sdc_traj_world = np.array(extract_result["sdc_traj"])  # shape (T, 2)

        # === 提取 ego 的世界坐标 + 朝向 ===
        ego, ego_pos, ego_heading = extract_ego_info(scene, frame_idx=self.frame_idx)

        # === 构建变换器：world → ego ===
        w2e = build_local_transform(ego_pos, ego_heading)

        # === 变换所有车辆轨迹（会写入 objects[i]["position_bev"]）===
        transform_all_trajectories(scene, w2e, frame_idx=self.frame_idx)

        # === 将 lane_graph 转换为 BEV 坐标 ===
        lane_graph_bev = transform_lane_graph_to_bev(scene["lane_graph"], w2e)

        extract_result = extract_sdc_and_neighbors(scene, frame_idx=self.frame_idx)
        sdc_traj_bev = np.array(
            extract_result["sdc_traj"]
        )  # 已在 transform 中被变换成 BEV

        # === 取出 SDC 车辆的 BEV 轨迹（我们用于建模的输入）===
        av_idx = scene["av_idx"]
        sdc_traj_bev = np.array(
            scene["objects"][av_idx]["position_bev"]
        )  # shape (T, 2)
        neighbors_traj_bev = []

        for i, obj in enumerate(scene["objects"]):
            if i == av_idx:
                continue  # 跳过 SDC
            traj = obj.get("position_bev", [])
            valid = obj.get("valid", [True] * len(traj))
            traj_filtered = [pt for pt, v in zip(traj, valid) if v]
            if traj_filtered:
                neighbors_traj_bev.append(traj_filtered)

        # === 回写到 sample["scenario"] 中 ===
        scene["lane_graph_bev"] = lane_graph_bev  # 存储变换后的 lane graph
        scene["w2e"] = w2e  # 可选保存，用于可视化等
        scene["ego_pose"] = (ego_pos, ego_heading)  # 保存 ego 位姿
        scene["sdc_traj_bev"] = sdc_traj_bev.tolist()  # ✅ 最终模型输入轨迹
        scene["neighbors_traj_bev"] = neighbors_traj_bev

        sample["scenario"] = scene
        return sample


# === Stage 2: 提取给模型用的字段（如 sdc history & future） ===
class ExtractModelInput:
    def __init__(self, history_len=10, future_len=20):
        self.hist = history_len
        self.fut = future_len

    def __call__(self, sample: dict) -> dict:
        traj = sample["scenario"]["sdc_traj_bev"]  # BEV 坐标下的轨迹
        hist = traj[: self.hist]
        fut = traj[self.hist : self.hist + self.fut]

        # ✅ inplace 添加新的训练字段，不要 return 新 dict
        sample["sdc_history"] = hist
        sample["sdc_future"] = fut
        return sample


# === Stage 3: numpy → torch.tensor ===
class ToTensor:
    def __call__(self, sample):
        out = {}
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                out[k] = torch.tensor(v, dtype=torch.float32)
            else:
                out[k] = v
        return out
