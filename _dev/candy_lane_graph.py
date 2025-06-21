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
from jynxzzzdebug import debug_break, debug_print, explore_dict, setup_logger
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D
from tools.debug_scene_structure import explore_scene, print_scene_structure
from tools.scene_loader import (
    load_random_scene_from_list,
    load_scene_data,
    load_selected_scene_list,
)

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

# tools/encoder/simple_token.py


def plot_lane_graph(scene, frame_idx=0, radius=50.0, save_path="lane_debug.png"):
    # === 初始化画布
    # logging.info(f"{explore_dict(scene)}")
    # logging.info(f"scene keys: {scene.keys()}")
    # logging.info(f"lane graph keys: {scene.get('lane_graph', {}).keys()}")
    fig, ax = init_canvas(frame_idx, radius)
    # debug_break("init_canvas")

    # === 提取 ego 位置信息 & 坐标变换矩阵
    ego, ego_pos, ego_heading_deg = extract_ego_info(scene, frame_idx)
    w2e = build_local_transform(ego_pos, ego_heading_deg)

    # === 绘制 ego 位置（可选）
    draw_ego_box(ax)
    draw_heading_vector(ax, ego_pos, ego_heading_deg, w2e)

    draw_agents(ax, scene.get("objects", []), scene["av_idx"], frame_idx, w2e)
    # logging.info(
    #     f"Frame {frame_idx} | Ego Pos = {ego_pos}, Heading = {ego_heading_deg:.2f}°"
    # )
    #
    draw_lane_tokens(
        ax, scene.get("lane_graph", {}), scene.get("lane_token_map", {}), w2e, radius
    )

    draw_outer_circle(ax, radius)

    # logging.info(f"traffic lights: {scene.get('traffic_lights', [])}")
    draw_traffic_lights(ax, scene.get("traffic_lights", []), frame_idx, w2e)

    draw_traffic_light_tokens(
        ax,
        scene.get("traffic_light_tokens", []),
        scene.get("traffic_light_token_map", {}),
        w2e,
        radius,
    )
    # === 获取车道结构
    lane_graph = scene.get("lane_graph", {})
    lane_token_map = scene.get("lane_token_map", {})

    for lane_id, pts in lane_graph.get("lanes", {}).items():
        pts = np.array(pts)
        if pts.shape[0] < 2:
            continue  # 跳过异常 lane

        # 坐标变换到 ego frame
        pts = w2e(pts[:, :2])
        xs, ys = pts[:, 0], pts[:, 1]
        ax.plot(xs, ys, linestyle="-", linewidth=1.2)

        # 画上 token_id 作为标签
        # token_id = lane_token_map.get(lane_id, None)
        # if token_id is not None:
        #     mid = len(xs) // 2
        #     # 💥关键：安全地放置 text，避免触发 matplotlib 自动缩放机制
        #     if abs(xs[mid]) < radius and abs(ys[mid]) < radius:
        #         try:
        #             ax.text(xs[mid], ys[mid], str(token_id), fontsize=6, color="blue")
        #             # ax.text(xs[mid], ys[mid], str(token), fontsize=fontsize, color="blue")
        #         except Exception as e:
        #             logging.warning(
        #                 f"[text skipped] lane {lane_id} token render error: {e}"
        #             )

    # === 保存图像
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    print(f"Lane graph saved to {save_path}")
