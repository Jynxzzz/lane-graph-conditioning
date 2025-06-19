# tools/encoder/simple_token.py

# render_frame.py
# ✅ 完整版 render_bev_frame.py + 正确 heading 对齐逻辑
import json
import logging
import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from _dev.encoder_debug import encode_lanes_debug
from _dev.render_frame import build_local_transform, extract_ego_info
from jynxzzzdebug import debug_break, debug_print, explore_dict, setup_logger
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D
from tools.debug_scene_structure import explore_scene, print_scene_structure
from tools.scene_loader import (
    load_random_scene_from_list,
    load_scene_data,
    load_selected_scene_list,
)

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

        ego, ego_pos, ego_heading = extract_ego_info(scenario, 0)
        logging.info(f"[ENCODE] ego position: {ego_pos}, heading: {ego_heading}")

        w2e = build_local_transform(ego_pos, ego_heading)
        logging.info(f"[ENCODE] world to ego transform: {w2e}")
        lane_graph = scenario.get("lane_graph", {}).get("lanes", {})
        logging.info(f"[ENCODE] lane_graph has {len(lane_graph)} lanes")

        for lane_id, lane_pts in lane_graph.items():
            if not isinstance(lane_pts, np.ndarray) or lane_pts.shape[0] < 2:
                continue

            local_pts = w2e(lane_pts[:, :2])
            dists = np.linalg.norm(local_pts, axis=1)

            if np.any(dists < self.radius):  # ✅ 仅保留在视野内的 lane
                vec = local_pts[-1] - local_pts[0]
                angle = np.arctan2(vec[1], vec[0])
                token = int((angle + np.pi) / (2 * np.pi) * self.discretize_bins)

                lane_token_map[lane_id] = token  # ✅ 保存 id→token 映射
                tokens.append(token)
                logging.info(f"[ENCODE] lane {lane_id} → token {token}")
        logging.info(f"token map: {lane_token_map}")

        return tokens, lane_token_map
