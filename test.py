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

# === 可视化 Lane Graph ===
for crosswalk in scene["lane_graph"]["crosswalks"].values():
    x = crosswalk[:, 0]
    y = crosswalk[:, 1]
    plt.fill(x, y, color="green", alpha=0.3)

for lane in scene["lane_graph"]["lanes"].values():
    plt.plot(lane[:, 0], lane[:, 1], "k-", linewidth=0.5)

for edge in scene["lane_graph"]["road_edges"].values():
    plt.plot(edge[:, 0], edge[:, 1], "b--", linewidth=0.5)

# === 可视化 Agent 轨迹 ===
for i, agent in enumerate(scene["objects"]):
    if not isinstance(agent, dict):
        logger.warning(f"[Agent {i}] Unexpected type: {type(agent)}")
        continue

    pos_list = agent.get("position", [])
    valid_mask = agent.get("valid", [])

    if not pos_list or not valid_mask:
        continue

    logger.info(f"[Agent {i}] Keys: {agent.keys()}")
    logger.info(f"[Agent {i}] First pos entry: {pos_list[0]}")

    valid = np.array(valid_mask).astype(bool)
    if len(valid) == 0:
        continue

    # 裁剪长度一致
    pos_list = pos_list[: len(valid)]
    valid = valid[: len(pos_list)]

    try:
        # 提取有效轨迹点
        x = [p["x"] for i, p in enumerate(pos_list) if valid[i] and p["x"] > -9000]
        y = [p["y"] for i, p in enumerate(pos_list) if valid[i] and p["y"] > -9000]

        if len(x) < 2:
            continue

        color = "blue" if i == scene["av_idx"] else "red"
        linewidth = 2 if i == scene["av_idx"] else 0.8
        plt.plot(x, y, color=color, alpha=0.7, linewidth=linewidth)

        if i == scene["av_idx"]:
            plt.scatter(x[0], y[0], c="green", s=20, label="Ego Start")
            plt.scatter(x[-1], y[-1], c="red", s=20, label="Ego End")

    except Exception as e:
        logger.warning(f"[Agent {i}] Failed to plot: {e}")
        continue

plt.axis("equal")
plt.title("Waymo Scene Visualization")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("scene_viz.png", dpi=300)
plt.show()
