import pickle

import matplotlib.pyplot as plt

# === 1. 坐标转换模块: world_to_ego.py ===
import numpy as np
from jynxzzzdebug import debug_break, setup_logger

# === 加载数据 ===
# with open(
#     "/home/xingnan/VideoDataInbox/scenario_dreamer_waymo/train/training.tfrecord-00000-of-01000_9.pkl",
#     "rb",
# ) as f:
#     scene = pickle.load(f)
#
# logger = setup_logger("waymo_scene", "logs/test.log")
#
# logger.info(f"Scene keys: {scene.keys()}")
# logger.info(f"Lane graph keys: {scene['lane_graph'].keys()}")
#
# === 文件结构初始化 ===
# 项目路径: EgoDreamer/
# 我们现在构建核心模块: world_to_ego.py, render_bev.py, token_builder.py, transformer.py


def world_to_ego(points, ego_pos, ego_heading):
    """
    将世界坐标点集 points 转换为以 ego 为中心的局部坐标系
    points: np.ndarray, shape (N, 2)
    ego_pos: tuple, (x, y)
    ego_heading: float, 弧度制
    """
    dxdy = points - np.array(ego_pos)
    c, s = np.cos(-ego_heading), np.sin(-ego_heading)
    R = np.array([[c, -s], [s, c]])
    local = dxdy @ R.T
    return local


# === 2. BEV 渲染器: render_bev.py ===
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def render_bev(scene, ego_pos, ego_heading, radius=50.0):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Ego-Centered Bird's Eye View")
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.grid(True)
    ax.set_aspect("equal")

    # ==== 绘制 Lane ====
    for lane in scene["lane_graph"]["lanes"].values():
        lane_pts = np.array(lane)[:, :2]
        local = world_to_ego(lane_pts, ego_pos, ego_heading)
        ax.plot(local[:, 0], local[:, 1], "k-", linewidth=0.5)

    # ==== 绘制其他 Agent ====
    for i, agent in enumerate(scene["objects"]):
        traj = agent.get("position", [])
        valid = agent.get("valid", [])
        if not traj or not valid:
            continue
        if i == scene["av_idx"]:
            continue  # skip ego here
        pos = traj[0]  # 当前帧位置
        if pos["x"] < -9000:
            continue
        agent_xy = np.array([[pos["x"], pos["y"]]])
        agent_local = world_to_ego(agent_xy, ego_pos, ego_heading)
        ax.scatter(agent_local[0, 0], agent_local[0, 1], color="red", s=10)

    # ==== 绘制 Ego 自身 ====
    ax.scatter(0, 0, color="blue", s=30, label="Ego")
    ax.arrow(0, 0, 5, 0, head_width=2, color="blue")
    ax.add_patch(
        Circle((0, 0), radius, edgecolor="gray", facecolor="none", linestyle="--")
    )

    ax.legend()
    plt.tight_layout()
    plt.savefig("bev_debug.png", dpi=150)
    plt.close()


# === 3. 示例入口（demo_bev.py） ===
# 用于调用测试，之后我们接入 token embedding + transformer

if __name__ == "__main__":
    # === 正确加载场景（.pkl） ===

    with open(
        "/home/xingnan/VideoDataInbox/scenario_dreamer_waymo/train/training.tfrecord-00000-of-01000_9.pkl",
        "rb",
    ) as f:
        scene = pickle.load(f)

    ego = scene["objects"][scene["av_idx"]]
    pos0 = ego["position"][0]
    heading = ego["heading"][0]
    ego_pos = (pos0["x"], pos0["y"])

    render_bev(scene, ego_pos, heading)
    print("✅ BEV 可视化完成 (bev_debug.png)")

    ego = scene["objects"][scene["av_idx"]]
    pos0 = ego["position"][0]
    heading = ego["heading"][0]
    ego_pos = (pos0["x"], pos0["y"])

    render_bev(scene, ego_pos, heading)
    print("✅ BEV 可视化完成 (bev_debug.png)")
