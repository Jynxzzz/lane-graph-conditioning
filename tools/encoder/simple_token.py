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


class SimpleEncoder:
    def __init__(self, max_len=128, vocab_size=1024, discretize_bins=16, radius=50.0):
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.discretize_bins = discretize_bins
        self.radius = radius

    def encode(self, scenario, cfg):
        tokens = []
        lane_token_map = {}
        ego, ego_pos, ego_heading = extract_ego_info(scenario, frame_idx=0)

        w2e = build_local_transform(ego_pos, ego_heading)
        # 在调用 safe_draw_lane_graph 前加上
        sdc_xy = np.array([ego_pos[0], ego_pos[1]])
        ego_lane_id = find_ego_lane_id(sdc_xy, scenario["lane_graph"])
        G, _ = build_waterflow_graph(scenario["lane_graph"], ego_lane_id)

        lane_graph = scenario.get("lane_graph", {}).get("lanes", {})

        # debug_break("[DEBUG] Break to inspect lane graph structure")
        for lane_id in G.nodes:
            # 获取自身及邻居
            successors = list(G.successors(lane_id))
            predecessors = list(G.predecessors(lane_id))

            # 如果不够邻居，跳过或补零
            if len(successors) == 0 or len(predecessors) == 0:
                continue

            # === 朝向角（朝向谁） ===
            center_vec = compute_lane_heading(scenario, lane_id)
            pred_vecs = [compute_lane_heading(scenario, pid) for pid in predecessors]
            succ_vecs = [compute_lane_heading(scenario, sid) for sid in successors]

            # === 将角度转为离散 token
            token_center = angle2token(center_vec, bins=16)
            token_pred = angle2token(pred_vecs[0], bins=16)  # 取一个前驱
            token_succ = angle2token(succ_vecs[0], bins=16)

            # === 构造 token triplet
            token_triplet = (token_pred, token_center, token_succ)
            tokens.append(token_triplet)
            lane_token_map[lane_id] = token_triplet

        flat_tokens = [t for triplet in tokens for t in triplet]
        counter = Counter(flat_tokens)
        debug_print("🧩 Token 分布：", "begin!")
        for token, count in sorted(counter.items()):
            logging.info(f"Token {token}: {count} 次")

        debug_print("🧩 Token 分布：", "end!")
        return tokens, lane_token_map
