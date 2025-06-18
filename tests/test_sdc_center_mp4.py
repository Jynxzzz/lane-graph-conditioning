import math

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def world_to_ego(points, ego_pos, ego_heading):
    """
    将世界坐标点集 points 转换为以 ego 为中心的局部坐标系（朝向向上）

    Args:
        points (np.ndarray): shape (N, 2)，世界坐标点
        ego_pos (tuple): (x, y)，ego 当前世界坐标位置
        ego_heading (float): 弧度，ego 朝向角，正北为 0，逆时针为正（Waymo 提供的是这样的）

    Returns:
        np.ndarray: shape (N, 2)，变换后的相对坐标（以 ego 为原点，方向对齐）
    """
    # 平移坐标
    dxdy = points - np.array(ego_pos)  # 每个点相对 ego 的位移向量

    # 构造旋转矩阵（逆旋转，让 ego 朝向 Y+）
    cos_h = np.cos(-ego_heading)
    sin_h = np.sin(-ego_heading)
    R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])  # shape (2, 2)

    # 应用旋转矩阵
    local = dxdy @ R.T  # shape (N, 2)
    return local


def render_bev_video(scene, output_path="bev_video.mp4", radius=50.0):
    fig, ax = plt.subplots(figsize=(6, 6))

    ego = scene["objects"][scene["av_idx"]]
    ego_pos_seq = [(p["x"], p["y"]) for p in ego["position"]]
    ego_heading_seq = ego["heading"]

    def draw_frame(frame_idx):
        ax.clear()
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_aspect("equal")
        ax.set_title(f"BEV View - Frame {frame_idx}")
        ax.grid(True)

        # === 获取当前 ego pose ===
        ego_pos = ego_pos_seq[frame_idx]
        # heading = ego_heading_seq[frame_idx]
        heading = math.radians(ego_heading_seq[frame_idx])

        # ==== Lanes ====
        for lane in scene["lane_graph"]["lanes"].values():
            pts = np.array(lane)[:, :2]
            local = world_to_ego(pts, ego_pos, heading)
            ax.plot(local[:, 0], local[:, 1], "k-", linewidth=0.5)

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
            print(
                f"[DEBUG Frame {frame_idx}] ego_pos={ego_pos}, heading={heading} (type: {type(heading)})"
            )
            if frame_idx == 89:
                plt.savefig("debug_frame89.png", dpi=150)

            point_local = world_to_ego(point, ego_pos, heading)
            ax.scatter(point_local[0, 0], point_local[0, 1], color="red", s=10)

        ax.scatter(0, 0, color="blue", s=30, label="Ego")
        ax.arrow(0, 0, 5, 0, head_width=2, color="blue")
        ax.add_patch(
            Circle((0, 0), radius, edgecolor="gray", facecolor="none", linestyle="--")
        )

    ani = animation.FuncAnimation(fig, draw_frame, frames=90, interval=100)
    ani.save(output_path, writer="ffmpeg", dpi=150)
    print(f"✅ BEV 视频已导出：{output_path}")
    plt.close()


if __name__ == "__main__":
    import pickle

    import numpy as np

    # === 正确加载场景（.pkl） ===

    with open(
        "/home/xingnan/VideoDataInbox/scenario_dreamer_waymo/train/training.tfrecord-00000-of-01000_9.pkl",
        "rb",
    ) as f:
        scene = pickle.load(f)

    render_bev_video(scene, output_path="ego_bev_9sec.mp4")

    print("ego_bev_9sec.mp4")
