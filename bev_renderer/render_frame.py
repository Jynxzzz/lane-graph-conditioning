# render_frame.py
import logging
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D

# ✅ 完整版 render_bev_frame.py + 正确 heading 对齐逻辑


logging.basicConfig(level=logging.INFO)


def world_to_ego(points, ego_pos, ego_heading_deg):
    heading_rad = math.radians(ego_heading_deg)
    adjusted_heading = heading_rad - np.pi / 2  # Waymo heading: 0° = 北，转成 X+ 朝前

    dxdy = points - np.array(ego_pos)
    c, s = np.cos(-adjusted_heading), np.sin(-adjusted_heading)
    R = np.array([[c, -s], [s, c]])
    return dxdy @ R.T


# ✅ 完整版 render_bev_frame()（含 heading + velocity 可视化，逻辑无 bug）

import logging
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D

logging.basicConfig(level=logging.INFO)
from .draw_utils import (
    draw_agents,
    draw_ego_box,
    draw_heading_vector,
    draw_lane_graph,
    draw_velocity_vector,
)


def render_bev_frame(scene, frame_idx=0, radius=50.0, save_path="bev_debug.png"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor("white")
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect("equal")
    ax.set_title(f"BEV Debug Frame {frame_idx}")
    ax.grid(True)

    # === 提取 ego 状态 ===
    ego = scene["objects"][scene["av_idx"]]
    ego_pos = (ego["position"][frame_idx]["x"], ego["position"][frame_idx]["y"])
    ego_heading_deg = ego["heading"][frame_idx]
    vx = ego["velocity"][frame_idx]["x"]
    vy = ego["velocity"][frame_idx]["y"]

    logging.info(
        f"Frame {frame_idx} | Pos = {ego_pos}, Heading = {ego_heading_deg:.2f}°"
    )

    # === 局部转换函数（必须等 ego_pos/heading 有了后定义） ===
    def w2e(points):
        return world_to_ego(points, ego_pos, ego_heading_deg)

    # === Velocity 向量（蓝色）===
    world_vel_pt = np.array([[ego_pos[0] + vx, ego_pos[1] + vy]])
    vec_vel_local = w2e(world_vel_pt) - w2e(np.array([[ego_pos[0], ego_pos[1]]]))
    dvx, dvy = vec_vel_local[0] * 10
    ax.arrow(0, 0, dvx, dvy, fc="blue", ec="blue", alpha=0.6, zorder=4)
    # ax.text(dvx * 1.1, dvy * 1.1, "velocity", color="blue", fontsize=9)

    # === 车道线 + 朝向箭头 ===
    for lane in scene["lane_graph"]["lanes"].values():
        pts = np.array(lane)[:, :2]
        local = w2e(pts)
        ax.plot(local[:, 0], local[:, 1], "k-", linewidth=0.5, zorder=1)
        if len(local) >= 2:
            x0, y0 = local[0]
            x1, y1 = local[1]
            ax.arrow(
                x0,
                y0,
                x1 - x0,
                y1 - y0,
                head_width=0.4,
                head_length=0.7,
                color="orange",
                linewidth=0.5,
                zorder=2,
            )

    # === Agent ===
    for i, agent in enumerate(scene["objects"]):
        if i == scene["av_idx"]:
            continue
        pos_list = agent.get("position", [])
        valid = agent.get("valid", [])
        if frame_idx >= len(pos_list):
            continue
        pos = pos_list[frame_idx]
        if not valid[frame_idx] or pos["x"] < -9000:
            continue
        point = np.array([[pos["x"], pos["y"]]])
        point_local = w2e(point)
        ax.scatter(point_local[0, 0], point_local[0, 1], color="red", s=10, zorder=2)

    # === heading 向量（绿色）===
    heading_rad = math.radians(ego_heading_deg)
    hx = ego_pos[0] + np.cos(heading_rad)
    hy = ego_pos[1] + np.sin(heading_rad)

    world_heading_pt = np.array([[hx, hy]])
    ego_heading_pt = np.array([[ego_pos[0], ego_pos[1]]])

    vec_local = w2e(world_heading_pt) - w2e(ego_heading_pt)
    dx, dy = vec_local[0] * 10
    ax.arrow(
        0,
        0,
        dx,
        dy,
        head_width=1.0,
        head_length=1.5,
        fc="green",
        ec="green",
        linewidth=1.2,
        zorder=4,
    )
    ax.text(dx * 1.1, dy * 1.1, "heading", color="green", fontsize=9)
    logging.info(f"Heading vector in ego frame: ({dx:.2f}, {dy:.2f})")

    # === ego 蓝框，含旋转 ===
    ego_length = 4.8
    ego_width = 2.0
    vec = vec_local[0]
    theta = math.atan2(vec[1], vec[0])  # 朝向角
    logging.info(
        f"[Ego Rect Rotation] angle from heading vec = {math.degrees(theta):.2f}°"
    )

    transform = Affine2D().rotate_around(0, 0, theta)
    rect = Rectangle(
        (-ego_length / 2, -ego_width / 2),
        ego_length,
        ego_width,
        edgecolor="blue",
        facecolor="none",
        linewidth=1.5,
        zorder=3,
    )
    rect.set_transform(transform + ax.transData)
    ax.add_patch(rect)

    # === 外围圆圈 ===
    ax.add_patch(
        Circle(
            (0, 0), radius, edgecolor="gray", facecolor="none", linestyle="--", zorder=1
        )
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ BEV 可视化完成：{save_path}")
