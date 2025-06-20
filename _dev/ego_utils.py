import json
import logging
import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from jynxzzzdebug import debug_break, debug_print, explore_dict, setup_logger
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D
from tools.debug_scene_structure import explore_scene, print_scene_structure
from tools.lane_graph.lane_explorer import find_ego_lane_id, get_lane_traversal
from tools.lane_graph.lane_graph_builder import (
    build_subgraph_with_features,
    get_nearby_lane_ids,
)
from tools.scene_loader import (
    load_random_scene_from_list,
    load_scene_data,
    load_selected_scene_list,
)
from tools.traffic_light_manager.traffic_light_builder import (
    annotate_light_to_graph,
    extract_light_info,
    get_controlled_lights,
)

from _dev.encoder_debug import encode_lanes_debug
from _dev.render_frame import build_local_transform, extract_ego_info


def build_ego_centered_context(scene, frame_idx, radius=50.0):
    ego = scene["objects"][scene["av_idx"]]
    ego_pos = (
        float(ego["position"][frame_idx]["x"]),
        float(ego["position"][frame_idx]["y"]),
    )
    heading = float(ego["heading"][frame_idx])

    # === Step 1: 构造世界坐标到 ego 局部坐标的转换函数
    w2e = build_local_transform(ego_pos, heading)
    lane_graph = scene["lane_graph"]

    # === Step 2: 找出周围 lane ids
    debug_print("build_ego_centered_context", "=== Step 2: 找到周围 lane ids ===")

    # nearby_ids = get_nearby_lane_ids(lane_graph, w2e, radius=radius)
    ego_lane_id = find_ego_lane_id(ego_pos, lane_graph)
    logging.info(f"找到 ego lane id: {ego_lane_id}")

    main_lane_ids = get_lane_traversal(lane_graph, ego_lane_id, max_depth=3)
    logging.info(
        f"找到 {len(main_lane_ids)} 条主车道，包含 {len(lane_graph['lanes'])} 条总车道"
    )
    light_info = get_controlled_lights(scene, main_lane_ids)
    debug_print(
        "build_ego_centered_context",
        f"找到 {len(main_lane_ids)} 条主车道，包含 {len(light_info)} 个交通灯",
    )

    debug_break("=== Step 2: 找到周围 lane ids ===")

    # === Step 3: 构建子图 + 提取 lane 特征
    G_local = build_subgraph_with_features(lane_graph, nearby_ids)

    # === Step 4: 找交通灯状态，标注到图上
    light_dict = extract_light_info(scene)
    annotate_light_to_graph(G_local, light_dict)

    return {
        "ego_pos": ego_pos,
        "heading": heading,
        "lane_graph": G_local,
        "light_info": light_dict,
    }
