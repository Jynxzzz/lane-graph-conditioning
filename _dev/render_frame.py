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
# from bev_renderer.draw_utils import (
#     draw_agents,
#     draw_ego_box,
#     draw_heading_vector,
#     draw_lane_graph,
#     draw_traffic_lights,
#     draw_velocity_vector,
# )


def init_canvas(frame_idx, radius):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor("white")
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect("equal")
    ax.set_title(f"BEV Debug Frame {frame_idx}")
    ax.grid(True)
    return fig, ax


def extract_ego_info(scene, frame_idx):
    ego = scene["objects"][scene["av_idx"]]

    # === 提取坐标 ===
    ego_pos = (
        float(ego["position"][frame_idx]["x"]),
        float(ego["position"][frame_idx]["y"]),
    )

    # === 提取 heading ===
    heading_raw = ego["heading"][frame_idx]
    if isinstance(heading_raw, tuple):
        print(
            f"[⚠️ WARNING] heading is tuple at frame {frame_idx}, value = {heading_raw}"
        )
        heading = float(heading_raw[0])
    elif isinstance(heading_raw, list):
        print(
            f"[⚠️ WARNING] heading is list at frame {frame_idx}, value = {heading_raw}"
        )
        heading = float(heading_raw[0])
    else:
        heading = float(heading_raw)

    print(f"[DEBUG] Frame {frame_idx} heading: {heading:.2f}, type: {type(heading)}")
    return ego, ego_pos, heading


# def extract_ego_info(scene, frame_idx):
#     ego = scene["objects"][scene["av_idx"]]
#     ego_pos = (ego["position"][frame_idx]["x"], ego["position"][frame_idx]["y"])
#
#     heading = ego["heading"][frame_idx]
#     if isinstance(heading, tuple):
#         print(f"[⚠️ WARNING] heading is tuple at frame {frame_idx}, value = {heading}")
#         heading = heading[0]  # 解包tuple
#     ego_heading_deg = float(heading)
#     print("[DEBUG] ego heading:", ego_heading_deg)
#     print("[DEBUG] ego type:", type(ego_heading_deg))
#
#     return ego, ego_pos, ego_heading_deg


def build_local_transform(ego_pos, heading_deg):
    def w2e(points):
        return world_to_ego(points, ego_pos, heading_deg)

    return w2e


def draw_velocity_vector(ax, ego, ego_pos, w2e):
    vx = ego["velocity"][0]["x"]
    vy = ego["velocity"][0]["y"]
    world_vel_pt = np.array([[ego_pos[0] + vx, ego_pos[1] + vy]])
    vec = w2e(world_vel_pt) - w2e(np.array([[ego_pos[0], ego_pos[1]]]))
    dvx, dvy = vec[0] * 10
    ax.arrow(0, 0, dvx, dvy, fc="blue", ec="blue", alpha=0.6, zorder=4)


# def draw_heading_vector(ax, pos, heading_deg, w2e):
#     heading_rad = math.radians(heading_deg)
#     hx = pos[0] + np.cos(heading_rad)
#     hy = pos[1] + np.sin(heading_rad)
#     world_heading = np.array([[hx, hy]])
#     ego_heading = np.array([[pos[0], pos[1]]])
#     vec = w2e(world_heading) - w2e(ego_heading)
#     dx, dy = vec[0] * 10
#     ax.arrow(
#         0,
#         0,
#         dx,
#         dy,
#         head_width=1.0,
#         head_length=1.5,
#         fc="green",
#         ec="green",
#         linewidth=1.2,
#         zorder=4,
#     )
#     ax.text(dx * 1.1, dy * 1.1, "heading", color="green", fontsize=9)
#     logging.info(f"Heading vector in ego frame: ({dx:.2f}, {dy:.2f})")
#     return vec[0]


def draw_heading_vector(ax, ego_pos, heading_deg, w2e):
    print(
        f"[DEBUG draw_heading_vector] heading_deg = {heading_deg}, type = {type(heading_deg)}"
    )
    heading_rad = math.radians(heading_deg)
    hx = ego_pos[0] + np.cos(heading_rad)
    hy = ego_pos[1] + np.sin(heading_rad)
    world_heading_pt = np.array([[hx, hy]])
    ego_heading_pt = np.array([[ego_pos[0], ego_pos[1]]])
    vec = w2e(world_heading_pt) - w2e(ego_heading_pt)
    dx, dy = vec[0] * 10
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


def draw_lanes(ax, lane_graph, w2e):
    for lane in lane_graph.get("lanes", {}).values():
        try:
            pts = np.array(lane)[:, :2]
            local = w2e(pts)
            ax.plot(local[:, 0], local[:, 1], "k-", linewidth=0.5, zorder=1)
        except Exception as e:
            logging.warning(f"❌ Lane drawing error: {e}")


def draw_agents(ax, agents, av_idx, frame_idx, w2e):
    for i, agent in enumerate(agents):
        if i == av_idx:
            continue
        try:
            pos = agent["position"][frame_idx]
            valid = agent["valid"][frame_idx]
            if not valid or pos["x"] < -9000:
                continue
            point_local = w2e(np.array([[pos["x"], pos["y"]]]))
            ax.scatter(
                point_local[0, 0], point_local[0, 1], color="red", s=10, zorder=2
            )
        except Exception as e:
            logging.warning(f"agent {i} skipped: {e}")


def draw_ego_box(ax, w2e):
    ego_length, ego_width = 4.8, 2.0
    vec = np.array([1.0, 0.0])
    theta = math.atan2(vec[1], vec[0])
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


def draw_outer_circle(ax, radius):
    ax.add_patch(
        Circle(
            (0, 0), radius, edgecolor="gray", facecolor="none", linestyle="--", zorder=1
        )
    )


def save_canvas(fig, save_path):
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ BEV 渲染完成：{save_path}")


def render_bev_frame(scene, frame_idx=0, radius=50.0, save_path="bev_debug.png"):
    fig, ax = init_canvas(frame_idx, radius)
    ego, ego_pos, ego_heading_deg = extract_ego_info(scene, frame_idx)
    w2e = build_local_transform(ego_pos, ego_heading_deg)

    draw_velocity_vector(ax, ego, ego_pos, w2e)
    draw_heading_vector(ax, ego_pos, ego_heading_deg, w2e)
    draw_ego_box(ax, w2e)
    draw_lanes(ax, scene.get("lane_graph", {}), w2e)
    draw_agents(ax, scene.get("objects", []), scene["av_idx"], frame_idx, w2e)

    draw_outer_circle(ax, radius)
    logging.info(f"traffic lights: {scene.get('traffic_lights', [])}")
    draw_traffic_lights(ax, scene.get("traffic_lights", []), frame_idx, w2e)

    save_canvas(fig, save_path)


def draw_traffic_lights(ax, traffic_lights, frame_idx, w2e):
    if frame_idx >= len(traffic_lights):
        return

    state_color_map = {
        1: "red",  # ARROW_STOP
        2: "orange",  # ARROW_CAUTION
        3: "green",  # ARROW_GO
        4: "red",  # STOP
        5: "orange",  # CAUTION
        6: "green",  # GO
        7: "red",  # FLASHING_STOP
        8: "orange",  # FLASHING_CAUTION
    }

    for tls in traffic_lights[frame_idx]:
        state = tls["state"]
        stop_point = tls.get("stop_point")
        if stop_point is None:
            continue

        color = state_color_map.get(state, "gray")
        x, y = stop_point["x"], stop_point["y"]
        local = w2e(np.array([[x, y]]))
        px, py = local[0]

        ax.plot(px, py, "o", color=color, markersize=8, alpha=0.7)
        ax.text(px + 0.5, py + 0.5, f"{state}", fontsize=6, color=color)


# def render_bev_frame(scene, frame_idx=0, radius=50.0, save_path="bev_debug.png"):
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.set_facecolor("white")
#     ax.set_xlim(-radius, radius)
#     ax.set_ylim(-radius, radius)
#     ax.set_aspect("equal")
#     ax.set_title(f"BEV Debug Frame {frame_idx}")
#     ax.grid(True)
#
#     # === 提取 ego 状态 ===
#     ego = scene["objects"][scene["av_idx"]]
#     ego_pos = (ego["position"][frame_idx]["x"], ego["position"][frame_idx]["y"])
#     ego_heading_deg = ego["heading"][frame_idx]
#     vx = ego["velocity"][frame_idx]["x"]
#     vy = ego["velocity"][frame_idx]["y"]
#
#     logging.info(
#         f"Frame {frame_idx} | Pos = {ego_pos}, Heading = {ego_heading_deg:.2f}°"
#     )
#
#     # === 局部转换函数（必须等 ego_pos/heading 有了后定义） ===
#     def w2e(points):
#         return world_to_ego(points, ego_pos, ego_heading_deg)
#
#     # === Velocity 向量（蓝色）===
#     world_vel_pt = np.array([[ego_pos[0] + vx, ego_pos[1] + vy]])
#     vec_vel_local = w2e(world_vel_pt) - w2e(np.array([[ego_pos[0], ego_pos[1]]]))
#     dvx, dvy = vec_vel_local[0] * 10
#     ax.arrow(0, 0, dvx, dvy, fc="blue", ec="blue", alpha=0.6, zorder=4)
#     # ax.text(dvx * 1.1, dvy * 1.1, "velocity", color="blue", fontsize=9)
#
#     # === 车道线 + 朝向箭头 ===
#     for lane in scene["lane_graph"]["lanes"].values():
#         pts = np.array(lane)[:, :2]
#         local = w2e(pts)
#         ax.plot(local[:, 0], local[:, 1], "k-", linewidth=0.5, zorder=1)
#         if len(local) >= 2:
#             x0, y0 = local[0]
#             x1, y1 = local[1]
#             ax.arrow(
#                 x0,
#                 y0,
#                 x1 - x0,
#                 y1 - y0,
#                 head_width=0.4,
#                 head_length=0.7,
#                 color="orange",
#                 linewidth=0.5,
#                 zorder=2,
#             )
#
#     # === Agent ===
#     for i, agent in enumerate(scene["objects"]):
#         if i == scene["av_idx"]:
#             continue
#         pos_list = agent.get("position", [])
#         valid = agent.get("valid", [])
#         if frame_idx >= len(pos_list):
#             continue
#         pos = pos_list[frame_idx]
#         if not valid[frame_idx] or pos["x"] < -9000:
#             continue
#         point = np.array([[pos["x"], pos["y"]]])
#         point_local = w2e(point)
#         ax.scatter(point_local[0, 0], point_local[0, 1], color="red", s=10, zorder=2)
#
#     # === heading 向量（绿色）===
#     heading_rad = math.radians(ego_heading_deg)
#     hx = ego_pos[0] + np.cos(heading_rad)
#     hy = ego_pos[1] + np.sin(heading_rad)
#
#     world_heading_pt = np.array([[hx, hy]])
#     ego_heading_pt = np.array([[ego_pos[0], ego_pos[1]]])
#
#     vec_local = w2e(world_heading_pt) - w2e(ego_heading_pt)
#     dx, dy = vec_local[0] * 10
#     ax.arrow(
#         0,
#         0,
#         dx,
#         dy,
#         head_width=1.0,
#         head_length=1.5,
#         fc="green",
#         ec="green",
#         linewidth=1.2,
#         zorder=4,
#     )
#     ax.text(dx * 1.1, dy * 1.1, "heading", color="green", fontsize=9)
#     logging.info(f"Heading vector in ego frame: ({dx:.2f}, {dy:.2f})")
#
#     # === ego 蓝框，含旋转 ===
#     ego_length = 4.8
#     ego_width = 2.0
#     vec = vec_local[0]
#     theta = math.atan2(vec[1], vec[0])  # 朝向角
#     logging.info(
#         f"[Ego Rect Rotation] angle from heading vec = {math.degrees(theta):.2f}°"
#     )
#
#     transform = Affine2D().rotate_around(0, 0, theta)
#     rect = Rectangle(
#         (-ego_length / 2, -ego_width / 2),
#         ego_length,
#         ego_width,
#         edgecolor="blue",
#         facecolor="none",
#         linewidth=1.5,
#         zorder=3,
#     )
#     rect.set_transform(transform + ax.transData)
#     ax.add_patch(rect)
#
#     # === 外围圆圈 ===
#     ax.add_patch(
#         Circle(
#             (0, 0), radius, edgecolor="gray", facecolor="none", linestyle="--", zorder=1
#         )
#     )
#
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150)
#     plt.close()
#     print(f"✅ BEV 可视化完成：{save_path}")
