# tools/encoder/simple_token.py

import json
import logging
import math
import os
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from _dev.encoder_debug import encode_lanes_debug
from _dev.render_frame import build_local_transform, extract_ego_info
from jynxzzzdebug import debug_break, debug_print, explore_dict, setup_logger
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D
from tools.debug_scene_structure import explore_scene, print_scene_structure
from tools.lane_graph.lane_explorer import build_waterflow_graph, find_ego_lane_id
from tools.scene_loader import (
    load_random_scene_from_list,
    load_scene_data,
    load_selected_scene_list,
)
from utils.utils2tokens import angle2token, compute_lane_heading

logging = setup_logger("simple_token", "logs/simple_token.log")

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from tools.encoder.base_encoder import BaseEncoder  # ← 引入我们刚刚写的抽象类
from tools.encoder.token_types import LaneToken, TrafficLightToken
from tools.encoder.traj_tokenizer import encode_traj_to_tokens
from utils.traj_processing import extract_sdc_and_neighbors


class SimpleEncoder(BaseEncoder):
    def __init__(self, max_len=128, vocab_size=1024, discretize_bins=16, radius=50.0):
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.discretize_bins = discretize_bins
        self.radius = radius

    def encode_lanes(self, scene):
        return self._encode_lanes_impl(scene)

    def encode_traffic_lights(self, scene, frame_idx):
        return self._encode_traffic_lights_impl(scene, frame_idx)

    def encode_agents(self, scene, frame_idx):
        traj_info = extract_sdc_and_neighbors(scene, frame_idx=frame_idx)
        sdc_traj = traj_info["sdc_traj"]
        neighbors = traj_info["neighbor_trajs"]
        return encode_traj_to_tokens(sdc_traj, neighbors)

    # ✅ 你接下来要写的
    def extract_gt_path_lanes(self, scene):
        """
        从 ground truth 轨迹中解析出 goal 点，并寻找最短路径经过的 lane id 列表。
        返回：goal_lane_id, lane_path_ids
        """
        return self._extract_gt_path_lanes_impl(scene)

    def _encode_lanes_impl(self, scene):
        tokens = []
        lane_token_map = {}

        # === 1. 提取 ego 信息与 BEV 变换
        ego, ego_pos, ego_heading = extract_ego_info(scene, frame_idx=0)
        w2e = build_local_transform(ego_pos, ego_heading)
        sdc_xy = np.array([ego_pos[0], ego_pos[1]])

        # === 2. 找到 ego 所在 lane
        ego_lane_id = find_ego_lane_id(sdc_xy, scene["lane_graph"])

        # === 3. 构建水流图
        G, _ = build_waterflow_graph(scene["lane_graph"], ego_lane_id)
        lane_graph = scene["lane_graph"].get("lanes", {})

        # === 4. 提取红绿灯和停牌 lane id
        traffic_light_lanes = set()
        for light in scene.get("traffic_lights", []):
            if "lane" in light:
                traffic_light_lanes.add(light["lane"])

        stop_sign_lanes = set()
        for ss in scene["lane_graph"].get("stop_signs", []):
            if "lane" in ss:
                stop_sign_lanes.add(ss["lane"])

        # === 5. 构建 suc/pred 图索引（保证方向连通）
        suc_map = scene["lane_graph"].get("suc_pairs", {})
        pred_map = scene["lane_graph"].get("pre_pairs", {})

        logging.info(f"🚦 traffic_light lanes: {traffic_light_lanes}")
        logging.info(f"🛑 stop_sign lanes: {stop_sign_lanes}")

        # === 5. 遍历 lane 节点构建 token
        token_id = 0
        for lane_id in G.nodes:
            centerline = lane_graph.get(lane_id)
            if centerline is None or centerline.shape[0] < 2:
                continue

            # === 计算 heading vector
            center_vec = centerline[-1, :2] - centerline[0, :2]
            center_vec = center_vec / (np.linalg.norm(center_vec) + 1e-6)
            heading_token = angle2token(center_vec, bins=16)

            # === 获取邻居
            left_id = scene["lane_graph"]["left_pairs"].get(lane_id, [None])[0]
            right_id = scene["lane_graph"]["right_pairs"].get(lane_id, [None])[0]

            # === 后继/前驱信息（已是 lane_id）
            suc_ids = suc_map.get(lane_id, [])
            pred_ids = pred_map.get(lane_id, [])

            # === 状态标注
            is_start = lane_id == ego_lane_id
            has_light = lane_id in traffic_light_lanes
            has_stop = lane_id in stop_sign_lanes

            # === 构造 token 对象
            token = LaneToken(
                id=token_id,
                lane_id=lane_id,
                centerline=centerline,
                heading_token=heading_token,
                succ_id=suc_ids,
                pred_id=pred_ids,
                left_id=left_id,
                right_id=right_id,
                is_start=is_start,
                has_traffic_light=has_light,
                has_stop_sign=has_stop,
                ego_xy=sdc_xy,
                w2e=w2e,
            )
            # logging.info(f"🚧 Lane Token: {token}")

            tokens.append(token)
            lane_token_map[lane_id] = token_id
            token_id += 1

        logging.info(f"🚧 共生成 {len(tokens)} 个 LaneToken")
        return tokens, lane_token_map

    def _encode_traffic_lights_impl(self, scene, frame_idx):
        traffic_lights = scene.get("traffic_lights", [])

        if frame_idx >= len(traffic_lights):
            return [], {}

        tokens = []
        token_map = {}
        lane_graph = scene.get("lane_graph", {}).get("lanes", {})

        for i, tls in enumerate(traffic_lights[frame_idx]):
            stop_point = tls.get("stop_point")
            if stop_point is None:
                continue
            controlled_lane = tls.get("lane", None)

            # 计算 dx, dy 用于绘图偏移
            if controlled_lane is not None and controlled_lane in lane_graph:
                lane = lane_graph[controlled_lane]
                if len(lane) >= 2:
                    vec = lane[-1, :2] - lane[0, :2]
                    norm = np.linalg.norm(vec) + 1e-6
                    dx, dy = vec[0] / norm, vec[1] / norm

            token = TrafficLightToken(
                id=i,
                frame_idx=frame_idx,
                x=stop_point["x"],
                y=stop_point["y"],
                state=tls["state"],
                controlled_lane=controlled_lane,
                dx=dx,
                dy=dy,
            )
            # logging.info(f"🚦 Traffic Light Token: {token}")
            tokens.append(token)
            token_map[i] = len(tokens) - 1

        return tokens, token_map

    def _extract_gt_path_lanes_impl(self, scene):
        # 1. 找到 ego 车 ID（一般是 0）
        sdc_id = scene["sdc_id"]
        sdc_track = scene["tracks"][sdc_id]

        # 2. 提取其轨迹
        traj = sdc_track["trajectory"]  # (T, 2)

        # 3. 取最后一个位置作为 goal 候选点
        goal_xy = traj[-1]

        # 4. 在 lane_graph 中找最近的 lane_id
        lane_graph = scene["lane_graph"]
        goal_lane_id = find_nearest_lane(goal_xy, lane_graph)

        # 5. 用 BFS / Dijkstra 找从当前 ego lane 到 goal 的 lane_id path
        ego_lane_id = find_nearest_lane(traj[0], lane_graph)
        path = find_shortest_lane_path(ego_lane_id, goal_lane_id, lane_graph)

        return goal_lane_id, path
