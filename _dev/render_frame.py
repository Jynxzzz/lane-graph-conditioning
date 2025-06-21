# render_frame.py
import logging
import math

import matplotlib.pyplot as plt
import numpy as np
from jynxzzzdebug import debug_break, debug_print, explore_dict, setup_logger
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D

from _dev.encoder_debug import encode_lanes_debug

# ✅ 完整版 render_bev_frame.py + 正确 heading 对齐逻辑


logging = setup_logger("render_frame", "logs/render_frame.log")


def within_radius(x, y, radius):
    return abs(x) < radius and abs(y) < radius


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
    debug_print("extract_ego_info", "start extracting ego info")

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
    logging.info(
        f"[DEBUG] Building local transform for ego at {ego_pos} with heading {heading_deg}°"
    )

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
    logging.info(
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


def draw_lanes(ax, lane_graph, w2e, lane_token_map=None):
    for lane_id, lane in lane_graph.get("lanes", {}).items():
        try:
            pts = np.array(lane)[:, :2]
            local = w2e(pts)
            ax.plot(local[:, 0], local[:, 1], "k-", linewidth=0.5, zorder=1)

            if local.shape[0] == 0:
                continue

            # === 起点位置（文本 anchor）
            x, y = local[0]

            # === 显示 lane id
            # ax.text(
            #     x,
            #     y + 0.5,
            #     f"{lane_id}",
            #     fontsize=5,
            #     color="black",
            #     ha="center",
            #     va="center",
            #     bbox=dict(
            #         facecolor="white",
            #         alpha=0.6,
            #         edgecolor="none",
            #         boxstyle="round,pad=0.1",
            #     ),
            #     zorder=5,
            # )

            # # === 显示 token，如果有
            # if lane_token_map and lane_id in lane_token_map:
            #     token = lane_token_map[lane_id]
            #     ax.text(
            #         x,
            #         y + 1.0,
            #         f"t{token}",
            #         fontsize=5,
            #         color="red",
            #         ha="center",
            #         va="center",
            #         bbox=dict(facecolor="white", alpha=0.4, edgecolor="none"),
            #         zorder=5,
            #     )

        except Exception as e:
            logging.warning(f"❌ Lane {lane_id} drawing error: {e}")


# def draw_lanes(ax, lane_graph, w2e):
#     for lane in lane_graph.get("lanes", {}).values():
#         try:
#             pts = np.array(lane)[:, :2]
#             local = w2e(pts)
#             ax.plot(local[:, 0], local[:, 1], "k-", linewidth=0.5, zorder=1)
#         except Exception as e:
#             logging.warning(f"❌ Lane drawing error: {e}")


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


# 所有 lane/agent 都已经经过 w2e 转换
# 那我们只需要画一个朝 +x 方向的蓝框
def draw_ego_box(ax, length=4.8, width=2.0):
    rect = Rectangle(
        (-width / 2, -length / 2),  # 横向是短边，纵向是长边
        width,  # ➡️ x 方向：2.0m
        length,  # ⬆️ y 方向：4.8m
        edgecolor="blue",
        facecolor="none",
        linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(rect)


# def draw_ego_box(ax, w2e):
#     ego_length, ego_width = 4.8, 2.0
#     vec = np.array([1.0, 0.0])
#     theta = math.atan2(vec[1], vec[0])
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


# visualize_utils.py
import matplotlib.pyplot as plt


def draw_traffic_light_tokens(
    ax, traffic_light_tokens, token_map, w2e=None, radius=50.0
):
    for i, token in enumerate(traffic_light_tokens):
        try:
            x, y = token.x, token.y
            if w2e is not None:
                x, y = w2e(np.array([[x, y]]))[0]

            if abs(x) > radius or abs(y) > radius:
                continue

            token_id = token_map.get(i, None)

            # === 显示 Traffic Light 名称（上）
            if within_radius(x, y, radius):
                ax.text(
                    x,
                    y - 3,
                    f"TL{i}",
                    fontsize=6,
                    color="darkred",
                    ha="center",
                    va="center",
                    bbox=dict(
                        facecolor="white",
                        alpha=0.6,
                        edgecolor="gray",
                        boxstyle="round,pad=0.2",
                    ),
                    zorder=5,
                )

            # === 显示 Token ID（下）
            if within_radius(x, y, radius) and token_id is not None:
                ax.text(
                    x,
                    y - 6,
                    f"t{token_id}",
                    fontsize=7,
                    color="green",
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="white", alpha=0.4, edgecolor="none"),
                    zorder=5,
                )

        except Exception as e:
            logging.warning(f"[draw_traffic_light_tokens] skipped: {e}")


def draw_lane_tokens(ax, lane_graph, lane_token_map, w2e=None, fontsize=6, radius=50):
    for lane_id, lane_pts in lane_graph.get("lanes", {}).items():
        token = lane_token_map.get(lane_id)
        if token is None:
            continue

        lane_pts = np.array(lane_pts)
        if lane_pts.ndim != 2 or lane_pts.shape[0] == 0 or lane_pts.shape[1] < 2:
            logging.warning(
                f"[draw_lane_tokens] lane {lane_id} → 非法坐标 shape {lane_pts.shape}"
            )
            continue

        if w2e is not None:
            try:
                transformed_pts = w2e(lane_pts[:, :2])
            except Exception as e:
                logging.error(f"[❌ ERROR] lane {lane_id} transform failed: {e}")
                continue

            if transformed_pts.ndim != 2 or transformed_pts.shape[0] == 0:
                logging.warning(
                    f"[⚠️ skipped] lane {lane_id} transformed result invalid: shape={transformed_pts.shape}"
                )
                continue

            lane_pts = transformed_pts
        # 显示 lane id（上方偏移 + 黑色）
        x, y = lane_pts[0, 0], lane_pts[0, 1]  # 起点位置
        if within_radius(x, y, radius):
            ax.text(
                x,
                y - 9.0,
                f"{lane_id}",
                fontsize=6,
                color="black",
                ha="center",
                va="center",
                bbox=dict(
                    facecolor="white",
                    alpha=0.6,
                    edgecolor="gray",
                    boxstyle="round,pad=0.2",
                ),
                zorder=5,
            )

        # 显示 token（下方偏移 + 蓝色）

        if lane_token_map and lane_id in lane_token_map and within_radius(x, y, radius):
            token = lane_token_map[lane_id]
            ax.text(
                x,
                y - 12,
                f"t{token}",
                fontsize=7,  # 稍大点
                color="blue",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.4, edgecolor="none"),
                zorder=5,
            )
        xs, ys = lane_pts[:, 0], lane_pts[:, 1]
        mid = len(xs) // 2
        ax.plot(xs, ys, linewidth=1, color="gray", alpha=0.6)

        # 💥关键：安全地放置 text，避免触发 matplotlib 自动缩放机制
        # if abs(xs[mid]) < radius and abs(ys[mid]) < radius:
        #     try:
        #         ax.text(xs[mid], ys[mid], str(token), fontsize=fontsize, color="blue")
        #     except Exception as e:
        #         logging.warning(
        #             f"[text skipped] lane {lane_id} token render error: {e}"
        #         )


# def draw_lane_tokens(ax, lane_graph, lane_token_map, w2e=None, fontsize=6):
#     for lane_id, lane_pts in lane_graph.get("lanes", {}).items():
#         token = lane_token_map.get(lane_id)
#         if token is None:
#             continue
#         lane_pts = np.array(lane_pts)
#         logging.info(f"[draw_lane_tokens] lane {lane_id} has token {token}")
#         if lane_pts.ndim != 2 or lane_pts.shape[0] == 0 or lane_pts.shape[1] < 2:
#             logging.warning(
#                 f"[draw_lane_tokens] lane {lane_id} → 非法坐标 shape {lane_pts.shape}"
#             )
#             continue
#         if w2e is not None:
#             logging.info(f"[draw_lane_tokens] lane {lane_id} w2e transform")
#             try:
#                 transformed_pts = w2e(lane_pts[:, :2])
#                 logging.info(
#                     f"[draw_lane_tokens] lane {lane_id} transformed shape: {transformed_pts.shape}"
#                 )
#             except Exception as e:
#                 logging.error(f"[❌ ERROR] lane {lane_id} transform failed: {e}")
#                 continue
#
#             if transformed_pts.ndim != 2 or transformed_pts.shape[0] == 0:
#                 logging.warning(
#                     f"[⚠️ skipped] lane {lane_id} transformed result invalid: shape={transformed_pts.shape}"
#                 )
#                 continue
#
#             lane_pts = transformed_pts
#         xs, ys = lane_pts[:, 0], lane_pts[:, 1]
#         mid = len(xs) // 2
#         ax.text(xs[mid], ys[mid], str(token), fontsize=fontsize, color="blue")
#         ax.plot(xs, ys, linewidth=1, color="gray", alpha=0.6)


def render_bev_frame(
    scene, frame_idx=0, radius=50.0, save_path="bev_debug.png", mode="default"
):

    fig, ax = init_canvas(frame_idx, radius)
    ego, ego_pos, ego_heading_deg = extract_ego_info(scene, frame_idx)
    w2e = build_local_transform(ego_pos, ego_heading_deg)

    draw_velocity_vector(ax, ego, ego_pos, w2e)
    draw_heading_vector(ax, ego_pos, ego_heading_deg, w2e)
    draw_ego_box(ax)
    draw_lanes(
        ax,
        scene.get("lane_graph", {}),
        w2e,
        lane_token_map=scene.get("lane_token_map", {}),
    )
    draw_agents(ax, scene.get("objects", []), scene["av_idx"], frame_idx, w2e)

    draw_lane_tokens(
        ax, scene.get("lane_graph", {}), scene.get("lane_token_map", {}), w2e, radius
    )

    draw_outer_circle(ax, radius)
    logging.info(f"traffic lights: {scene.get('traffic_lights', [])}")
    draw_traffic_lights(ax, scene.get("traffic_lights", []), frame_idx, w2e)
    # if mode == "encode":
    #     fig, ax = plt.subplots(figsize=(8, 8))  # 💡 新增画布
    #     tokens = encode_lanes_debug(scene["lane_graph"], w2e, ax=ax)
    #     print(f"👁️ Lane tokens: {tokens}")
    #     plt.axis("equal")  # optional：保持比例
    #     plt.title("Lane Token Visualization")
    #     plt.show()  # 💡 不要忘记展示

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
