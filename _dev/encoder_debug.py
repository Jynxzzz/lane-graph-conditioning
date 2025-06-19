import matplotlib.pyplot as plt
import numpy as np
from jynxzzzdebug import debug_break, explore_dict, setup_logger

logging = setup_logger("encode_lanes", "logs/encode_lanes.log")


def encode_lanes_debug(lane_graph, w2e, ax=None, radius=50.0, discretize_bins=16):
    tokens = []

    for lane_id, lane_pts in lane_graph["lanes"].items():
        # logging.info(f"explore lane: {explore_dict(lane_graph)}")
        if not isinstance(lane_pts, np.ndarray) or lane_pts.shape[0] < 2:
            logging.warning(
                f"[⚠️ skipped] lane {lane_id} is invalid. type={type(lane_pts)}, shape={getattr(lane_pts, 'shape', None)}"
            )
            continue

        # 转换为 ego 坐标
        local_pts = w2e(lane_pts[:, :2])

        logging.info(f"[DEBUG] lane_graph has {len(lane_graph)} lanes")
        logging.debug(f"[🚧 debug] lanes keys: {list(lane_graph['lanes'].keys())[:5]}")
        for lane_id, lane_pts in lane_graph["lanes"].items():
            if not isinstance(lane_pts, np.ndarray) or lane_pts.shape[0] < 2:
                logging.warning(
                    f"[⚠️ skipped] lane {lane_id} is invalid. type={type(lane_pts)}, shape={getattr(lane_pts, 'shape', None)}"
                )
                continue

            logging.info(f"[🧠 encode] lane {lane_id}: shape={lane_pts.shape}")
            # 下面才是 encode 的逻辑
        dists = np.linalg.norm(local_pts, axis=1)
        if np.any(dists < radius):
            # 方向向量
            vec = local_pts[-1] - local_pts[0]
            angle = np.arctan2(vec[1], vec[0])
            angle_deg = np.degrees(angle)
            token = int((angle + np.pi) / (2 * np.pi) * discretize_bins)

            tokens.append(token)

            if ax is not None:
                ax.plot(local_pts[:, 0], local_pts[:, 1], color="black", linewidth=1)
                ax.arrow(
                    local_pts[0, 0],
                    local_pts[0, 1],
                    vec[0],
                    vec[1],
                    head_width=1,
                    head_length=2,
                    fc="orange",
                    ec="orange",
                )
                ax.text(
                    local_pts[0, 0] + 1.5,  # 轻微右偏，防止重叠
                    local_pts[0, 1] + 0.8,  # 上移一点
                    str(token),
                    fontsize=12,  # 放大字体
                    color="red",
                    weight="bold",  # 加粗更明显
                    zorder=10,  # 确保在最顶层
                    bbox=dict(
                        facecolor="white", edgecolor="none", alpha=0.7
                    ),  # 增加白底遮挡
                )

    return tokens
