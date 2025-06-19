# draw_utils.py
import logging
import math

import numpy as np
from matplotlib.patches import Rectangle


def draw_velocity_vector(ax, pos, vx, vy, w2e):
    world_pt = np.array([[pos[0] + vx, pos[1] + vy]])
    vec = w2e(world_pt) - w2e(np.array([[pos[0], pos[1]]]))
    dx, dy = vec[0] * 10
    ax.arrow(0, 0, dx, dy, fc="blue", ec="blue", alpha=0.6, zorder=4)


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
        px, py = w2e(x, y)

        ax.plot(px, py, "o", color=color, markersize=8, alpha=0.7)
        ax.text(px + 0.5, py + 0.5, f"{state}", fontsize=6, color=color)


def draw_lane_graph(ax, lanes, w2e):
    for lane in lanes.values():
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


def draw_agents(ax, objects, av_idx, frame_idx, w2e):
    for i, agent in enumerate(objects):
        if i == av_idx:
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


def draw_heading_vector(ax, pos, heading_deg, w2e):
    heading_rad = math.radians(heading_deg)
    hx = pos[0] + np.cos(heading_rad)
    hy = pos[1] + np.sin(heading_rad)
    world_heading = np.array([[hx, hy]])
    ego_heading = np.array([[pos[0], pos[1]]])
    vec = w2e(world_heading) - w2e(ego_heading)
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
    ax.text(dx * 1.1, dy * 1.1, "heading", color="green", fontsize=9)
    logging.info(f"Heading vector in ego frame: ({dx:.2f}, {dy:.2f})")
    return vec[0]


def draw_ego_box(ax, heading_vec):
    ego_length = 4.8
    ego_width = 2.0
    theta = math.atan2(heading_vec[1], heading_vec[0])
    logging.info(f"[Ego Rect Rotation] angle = {math.degrees(theta):.2f}°")
    transform = ax.transData + ax.transData.get_affine().rotate_around(0, 0, theta)
    rect = Rectangle(
        (-ego_length / 2, -ego_width / 2),
        ego_length,
        ego_width,
        edgecolor="blue",
        facecolor="none",
        linewidth=1.5,
        zorder=3,
    )
    rect.set_transform(transform)
    ax.add_patch(rect)
