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


def draw_agent_heading_arrow(
    ax, pos_world, heading_deg, w2e, length=3.0, color="darkgreen", zorder=4
):
    """
    参数：
        pos_world: (x, y)，世界坐标下 agent 的当前坐标
        heading_deg: 朝向角度（世界坐标系下）
        w2e: 世界坐标到 ego 坐标的转换函数
    """
    import math

    import numpy as np

    # 朝向角度 → 弧度
    heading_rad = math.radians(heading_deg)

    # 在世界坐标中构造朝向点
    hx = pos_world[0] + length * math.cos(heading_rad)
    hy = pos_world[1] + length * math.sin(heading_rad)

    pt_start = np.array([pos_world])  # 世界坐标起点
    pt_end = np.array([[hx, hy]])  # 世界坐标终点

    # 转为 ego frame
    pt_start_ego = w2e(pt_start)[0]
    pt_end_ego = w2e(pt_end)[0]

    # 画箭头
    ax.arrow(
        pt_start_ego[0],
        pt_start_ego[1],
        pt_end_ego[0] - pt_start_ego[0],
        pt_end_ego[1] - pt_start_ego[1],
        head_width=0.6,
        head_length=1.2,
        fc=color,
        ec=color,
        linewidth=1.0,
        zorder=zorder,
        alpha=0.8,
    )


def compute_agent_heading(agent, frame_idx, fallback=0.0):
    try:
        pos_now = agent["position"][frame_idx]
        pos_prev = agent["position"][frame_idx - 1]

        # 直接在世界坐标下计算
        dx = pos_now["x"] - pos_prev["x"]
        dy = pos_now["y"] - pos_prev["y"]

        if dx == 0 and dy == 0:
            return fallback

        return math.degrees(math.atan2(dy, dx))
    except Exception as e:
        logging.warning(f"[compute_agent_heading] failed: {e}")
        return fallback


def draw_agents_as_boxes(ax, agents, av_idx, frame_idx, w2e, draw_social_space=True):
    type_styles = {
        "vehicle": {"length": 4.8, "width": 2.0, "color": "blue", "buffer": 0.5},
        "pedestrian": {"length": 0.6, "width": 0.6, "color": "green", "buffer": 0.3},
        "cyclist": {"length": 1.8, "width": 0.6, "color": "orange", "buffer": 0.3},
    }

    for i, agent in enumerate(agents):
        if i == av_idx:
            continue
        try:
            pos = agent["position"][frame_idx]
            valid = agent["valid"][frame_idx]
            if not valid or pos["x"] < -9000:
                continue

            agent_type = agent.get("type", "vehicle").lower()
            style = type_styles.get(agent_type, type_styles["vehicle"])

            x_local, y_local = w2e([[pos["x"], pos["y"]]])[0]
            # 获取世界坐标
            pos = agent["position"][frame_idx]
            pos_world = (
                agent["position"][frame_idx]["x"],
                agent["position"][frame_idx]["y"],
            )
            heading_deg = compute_agent_heading(agent, frame_idx)
            # draw_agent_heading_arrow(ax, pos_world, heading_deg, w2e=w2e)

            # 主体 box
            rect = Rectangle(
                (x_local - style["width"] / 2, y_local - style["length"] / 2),
                style["width"],
                style["length"],
                edgecolor=style["color"],
                facecolor="none",
                linewidth=1.0,
                zorder=3,
            )
            ax.add_patch(rect)

            # 可选：社交 buffer（半透明）
            if draw_social_space:
                rect_buf = Rectangle(
                    (
                        x_local - (style["width"] / 2 + style["buffer"]),
                        y_local - (style["length"] / 2 + style["buffer"]),
                    ),
                    style["width"] + 2 * style["buffer"],
                    style["length"] + 2 * style["buffer"],
                    edgecolor=None,
                    facecolor=style["color"],
                    alpha=0.15,
                    zorder=2,
                )
                ax.add_patch(rect_buf)

        except Exception as e:
            logging.warning(f"[draw_agents_as_boxes] agent {i} skipped: {e}")


def draw_lane_area(ax, scene, w2e, lane_width=3.6, color="orange", alpha=0.25):
    """
    绘制带宽度的车道区域，可用于背景高亮

    参数：
        ax: matplotlib 轴对象
        scene: 包含 lane_graph 的场景字典
        w2e: 世界坐标系 -> ego 坐标系的变换函数
        lane_width: 车道宽度，单位 meter
        color: 填充颜色
        alpha: 透明度
    """
    import logging

    import numpy as np
    from matplotlib.patches import Polygon

    lane_graph = scene.get("lane_graph", {})
    num_lanes = 0

    for lane_id, centerline in lane_graph.get("lanes", {}).items():
        pts = np.array(centerline)
        if pts.shape[0] < 2:
            continue

        # 中心线转 ego frame
        pts = w2e(pts[:, :2])
        lefts, rights = [], []

        # 根据切线方向构造左右边界
        for i in range(len(pts) - 1):
            p1, p2 = pts[i], pts[i + 1]
            dir_vec = p2 - p1
            norm = np.linalg.norm(dir_vec)
            if norm == 0:
                continue
            perp = np.array([-dir_vec[1], dir_vec[0]]) / norm  # 法向量
            offset = (lane_width / 2.0) * perp

            lefts.append(p1 + offset)
            rights.append(p1 - offset)

            # 加上最后一个点偏移（末端封闭）
            if i == len(pts) - 2:
                lefts.append(p2 + offset)
                rights.append(p2 - offset)

        # 合并成闭环 polygon（左侧 + 右侧反向）
        if len(lefts) >= 2 and len(rights) >= 2:
            polygon = np.vstack([lefts, rights[::-1]])
            patch = Polygon(polygon, closed=True, color=color, alpha=alpha, linewidth=0)
            ax.add_patch(patch)
            num_lanes += 1

    logging.info(f"[draw_lane_area] ✅ 共绘制 {num_lanes} 个车道荧光区域")


import matplotlib.cm as cm
import matplotlib.colors as mcolors


def draw_lane_centerlines(ax, scene, w2e, colormap="tab20", alpha=0.7, linewidth=1.5):
    """
    绘制彩色车道中心线，每条 lane_id 分配不同颜色

    参数：
        ax: matplotlib 轴对象
        scene: 包含 lane_graph 的场景字典
        w2e: 世界坐标系 -> ego 坐标系的变换函数
        colormap: matplotlib 的 colormap 名称（如 tab20 / hsv / jet）
    """
    import logging

    import numpy as np

    lane_graph = scene.get("lane_graph", {})
    lane_ids = list(lane_graph.get("lanes", {}).keys())
    num_lanes = len(lane_ids)

    # 建立 color map 映射
    cmap = cm.get_cmap(colormap, num_lanes)
    color_map = {lane_id: mcolors.to_hex(cmap(i)) for i, lane_id in enumerate(lane_ids)}

    drawn = 0
    for lane_id, pts in lane_graph.get("lanes", {}).items():
        pts = np.array(pts)
        if pts.shape[0] < 2:
            continue

        pts = w2e(pts[:, :2])
        xs, ys = pts[:, 0], pts[:, 1]
        ax.plot(
            xs,
            ys,
            linestyle="-",
            linewidth=linewidth,
            color=color_map[lane_id],
            alpha=alpha,
            zorder=0,
        )
        drawn += 1

    logging.info(
        f"[draw_lane_centerlines] ✅ 彩色绘制 {drawn} 条 lane centerlines using '{colormap}'"
    )


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
    # ✅ 方式一：原生 Python 获取所有类型的唯一值
    # types = [agent.get("type", "unknown") for agent in scene["objects"]]
    # unique_types = set(types)
    # logging.info(f"✅ 当前场景共包含 agent 类型: {unique_types}")

    # === 绘制 ego 位置（可选）
    draw_ego_box(ax)

    draw_heading_vector(ax, ego_pos, ego_heading_deg, w2e)

    draw_agents(ax, scene.get("objects", []), scene["av_idx"], frame_idx, w2e)
    # logging.info(
    #     f"Frame {frame_idx} | Ego Pos = {ego_pos}, Heading = {ego_heading_deg:.2f}°"
    # )
    #
    draw_agents_as_boxes(
        ax, scene["objects"], av_idx=scene["av_idx"], frame_idx=frame_idx, w2e=w2e
    )

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
    # === draw lane background first
    # draw_lane_area(ax, scene, w2e, lane_width=3.6, color="orange", alpha=0.25)
    draw_lane_centerlines(ax, scene, w2e)
    draw_trajectories(ax, scene, frame_idx, w2e)
    #

    # === 保存图像
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    print(f"Lane graph saved to {save_path}")
