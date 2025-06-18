# render_bev_video.py (完整版)
import math

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle


def world_to_ego(points, ego_pos, ego_heading):
    dxdy = points - np.array(ego_pos)
    c, s = np.cos(-ego_heading), np.sin(-ego_heading)
    R = np.array([[c, -s], [s, c]])
    return dxdy @ R.T


def draw_ego(ax, length=4.8, width=2.0, buffer_length=10.0):
    rect = Rectangle(
        (-length / 2, -width / 2),
        length,
        width,
        edgecolor="blue",
        facecolor="none",
        linewidth=1.5,
        zorder=4,
    )
    ax.add_patch(rect)
    ax.arrow(
        0,
        0,
        buffer_length,
        0,
        head_width=1.0,
        head_length=1.5,
        fc="blue",
        ec="blue",
        linewidth=1.0,
        zorder=4,
    )


def render_bev_video(scene, output_path="ego_bev_9sec.mp4", radius=50.0):
    fig, ax = plt.subplots(figsize=(6, 6))

    ego = scene["objects"][scene["av_idx"]]
    ego_pos_seq = [(p["x"], p["y"]) for p in ego["position"]]
    ego_heading_seq = ego["heading"]

    def draw_frame(frame_idx):
        ax.clear()
        ax.set_facecolor("white")  # 防止黑屏
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_aspect("equal")
        ax.set_title(f"BEV View - Frame {frame_idx}")
        ax.grid(True)

        ego_pos = ego_pos_seq[frame_idx]
        heading = math.radians(ego_heading_seq[frame_idx])

        # ==== Lane ====
        for lane in scene["lane_graph"]["lanes"].values():
            pts = np.array(lane)[:, :2]
            local = world_to_ego(pts, ego_pos, heading)
            ax.plot(local[:, 0], local[:, 1], "k-", linewidth=0.5, zorder=1)

        # ==== Other agents ====
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
            point_local = world_to_ego(point, ego_pos, heading)
            ax.scatter(
                point_local[0, 0], point_local[0, 1], color="red", s=10, zorder=2
            )

        # ==== Ego vehicle ====
        draw_ego(ax)
        ax.add_patch(
            Circle(
                (0, 0),
                radius,
                edgecolor="gray",
                facecolor="none",
                linestyle="--",
                zorder=1,
            )
        )

    ani = animation.FuncAnimation(fig, draw_frame, frames=90, interval=100)
    ani.save(output_path, writer="ffmpeg", dpi=150)
    print(f"✅ BEV 视频已导出：{output_path}")
    plt.close()


if __name__ == "__main__":
    import pickle

    with open(
        "/home/xingnan/VideoDataInbox/scenario_dreamer_waymo/train/training.tfrecord-00000-of-01000_9.pkl",
        "rb",
    ) as f:
        scene = pickle.load(f)

    render_bev_video(scene, output_path="ego_bev_9sec.mp4")
    print("ego_bev_9sec.mp4")
