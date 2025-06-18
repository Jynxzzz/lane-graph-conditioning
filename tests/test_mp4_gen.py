import pickle

import matplotlib.pyplot as plt
import numpy as np
from jynxzzzdebug import debug_break, setup_logger

# === 加载数据 ===
with open(
    "/home/xingnan/VideoDataInbox/scenario_dreamer_waymo/train/training.tfrecord-00000-of-01000_9.pkl",
    "rb",
) as f:
    scene = pickle.load(f)

logger = setup_logger("waymo_scene", "logs/test.log")

logger.info(f"Scene keys: {scene.keys()}")
logger.info(f"Lane graph keys: {scene['lane_graph'].keys()}")


def draw_lane_graph(ax, scene):
    for crosswalk in scene["lane_graph"]["crosswalks"].values():
        x = crosswalk[:, 0]
        y = crosswalk[:, 1]
        ax.fill(x, y, color="green", alpha=0.3)

    for lane in scene["lane_graph"]["lanes"].values():
        ax.plot(lane[:, 0], lane[:, 1], "k-", linewidth=0.5)

    for edge in scene["lane_graph"]["road_edges"].values():
        ax.plot(edge[:, 0], edge[:, 1], "b--", linewidth=0.5)


import matplotlib.animation as animation
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))

# —— 预处理 agent 轨迹（跟之前一样） ——
agent_trajs = []
for i, agent in enumerate(scene["objects"]):
    if not isinstance(agent, dict):
        continue
    pos = agent.get("position", [])
    valid = agent.get("valid", [])
    if not pos or not valid:
        continue
    valid_mask = np.array(valid[: len(pos)]).astype(bool)
    traj = [
        (p["x"], p["y"]) if v and p["x"] > -9000 else None
        for p, v in zip(pos, valid_mask)
    ]
    agent_trajs.append({"id": i, "is_ego": i == scene["av_idx"], "traj": traj})


# —— 每一帧更新内容 ——
def update(frame_idx):
    ax.clear()
    ax.set_title(f"Frame {frame_idx}")
    ax.axis("equal")
    ax.grid(True)

    # ==== Lane graph 背景图 ====
    draw_lane_graph(ax, scene)

    # ==== Agent 动态轨迹 ====
    for agent in agent_trajs:
        traj = agent["traj"]
        if frame_idx >= len(traj):
            continue
        point = traj[frame_idx]
        if point is None:
            continue
        color = "blue" if agent["is_ego"] else "red"
        size = 30 if agent["is_ego"] else 10
        ax.scatter(point[0], point[1], c=color, s=size, alpha=0.8)

        # 可选拖尾线
        tail = [p for p in traj[: frame_idx + 1] if p is not None]
        if len(tail) >= 2:
            xs, ys = zip(*tail)
            ax.plot(xs, ys, color=color, alpha=0.3, linewidth=1)


ani = animation.FuncAnimation(fig, update, frames=90, interval=100)
ani.save("lane_scene_anim.gif", writer="pillow", dpi=150)
plt.close()
