# render_frame.py
# ✅ 完整版 render_bev_frame.py + 正确 heading 对齐逻辑
import json
import logging
import math
import os
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from _dev.encoder_debug import encode_lanes_debug
from _dev.render_frame import (
    build_local_transform,
    draw_agents,
    draw_ego_box,
    draw_heading_vector,
    draw_lane_tokens,
    draw_outer_circle,
    draw_traffic_light_tokens,
    draw_traffic_lights,
    extract_ego_info,
    init_canvas,
)
from datasets.diffusion.lane_graph_utils import (
    build_lane_tokens,
    build_waterflow_graph,
    find_lane_id_for_traj,
)
from jynxzzzdebug import debug_break, debug_print, explore_dict, setup_logger

# tools/encoder/simple_token.py
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D
from tools.debug_scene_structure import explore_scene, print_scene_structure
from tools.scene_loader import (
    load_random_scene_from_list,
    load_scene_data,
    load_selected_scene_list,
)
from utils.traj_processing import extract_sdc_and_neighbors
from utils.vis_traj import draw_trajectories


# transform_hooks.py
def extract_trajectories(sample):
    scenario = sample["scenario"]
    ego, ego_pos, ego_heading_deg = extract_ego_info(scenario, frame_idx=0)
    w2e = build_local_transform(ego_pos, ego_heading_deg)

    result = extract_sdc_and_neighbors(scenario, max_distance=50.0, frame_idx=0)

    sdc_traj = torch.tensor(w2e(result["sdc_traj"]), dtype=torch.float32)
    neighbor_trajs = [
        torch.tensor(w2e(np.array(traj)), dtype=torch.float32)
        for traj in result["neighbor_trajs"].values()
    ]

    sample["sdc_traj"] = sdc_traj
    sample["neighbor_trajs"] = (
        pad_sequence(neighbor_trajs, batch_first=True)
        if neighbor_trajs
        else torch.zeros((0, 91, 2))
    )
    return sample


def add_lane_graph_context(sample):
    """
    对 sample 增加 lane 图结构 G、token、轨迹对应 lane_id
    """
    traj = sample["sdc_traj"].numpy()  # (91, 2)
    lane_graph_raw = sample["lane_graph"]  # from raw scenario
    ego_pos = traj[0]  # 假设起点就是 ego 位置

    # 1. 限制感知范围构图（3-hop，或 circle 限定）
    G = build_waterflow_graph(lane_graph_raw, ego_pos=ego_pos, radius=50.0)

    # 2. 构造 lane 节点属性
    lane_tokens = build_lane_tokens(G, lane_graph_raw)

    # 3. 给每个轨迹点匹配所在的 lane
    traj_lane_ids = find_lane_id_for_traj(traj, lane_graph_raw)

    # 4. 加入 sample 中
    sample["graph"] = G
    sample["lane_tokens"] = lane_tokens
    sample["traj_lane_ids"] = traj_lane_ids
    return sample
