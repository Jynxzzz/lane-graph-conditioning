import matplotlib.pyplot as plt
import numpy as np
from jynxzzzdebug import debug_break, setup_logger
from matplotlib.patches import Rectangle

logging = setup_logger("lane_graph", "logs/lane_graph.log")


import logging

# def draw_lane_graph(ax, scene):
#     for crosswalk in scene["lane_graph"]["crosswalks"].values():
#         x = crosswalk[:, 0]
#         y = crosswalk[:, 1]
#         ax.fill(x, y, color="green", alpha=0.3)
#
#     for lane in scene["lane_graph"]["lanes"].values():
#         ax.plot(lane[:, 0], lane[:, 1], "k-", linewidth=0.5)
#
#     for edge in scene["lane_graph"]["road_edges"].values():
#         ax.plot(edge[:, 0], edge[:, 1], "b--", linewidth=0.5)
import numpy as np


def draw_lanes_near_sdc(scene, ax, frame_idx=0, radius=30.0):
    try:
        lanes = scene["lane_graph"].get("lanes", {})

        # 取出 ego 的位置（当前帧）
        ego_id = scene["av_idx"]
        ego_agent = scene["objects"][ego_id]
        pos = ego_agent.get("position", [])
        if frame_idx >= len(pos) or pos[frame_idx] is None:
            logging.warning("⚠️ Ego position not available at this frame")
            return

        sdc_pos = np.array([pos[frame_idx]["x"], pos[frame_idx]["y"]])

        for key, lane in lanes.items():
            if not isinstance(lane, np.ndarray) or lane.shape[1] < 2:
                logging.warning(f"⛔️ Invalid lane format at key {key}")
                continue

            # 判断是否在 radius 范围内
            dists = np.linalg.norm(lane[:, :2] - sdc_pos[:2], axis=1)
            if np.any(dists < radius):
                ax.plot(lane[:, 0], lane[:, 1], "k-", linewidth=1.0)
                logging.debug(f"✅ Lane {key} is within {radius}m of SDC")

    except Exception as e:
        logging.error(f"❌ Error drawing lanes: {e}")


def draw_lane_graph(ax, scene):
    # try:
    #     crosswalks = scene["lane_graph"].get("crosswalks", {})
    #     for key, crosswalk in crosswalks.items():
    #         if isinstance(crosswalk, np.ndarray) and crosswalk.shape[1] >= 2:
    #             x = crosswalk[:, 0]
    #             y = crosswalk[:, 1]
    #             ax.fill(x, y, color="green", alpha=0.3)
    #         else:
    #             logging.warning(
    #                 f"⛔️ Invalid crosswalk format at key {key}: {crosswalk}"
    #             )
    # except Exception as e:
    #     logging.error(f"❌ Error drawing crosswalks: {e}")
    #
    try:
        lanes = scene["lane_graph"].get("lanes", {})
        for key, lane in lanes.items():
            logging.debug(f"lane key: {key}, lane data: {lane}")

            if isinstance(lane, np.ndarray) and lane.shape[1] >= 2:
                ax.plot(lane[:, 0], lane[:, 1], "k-", linewidth=0.5)

            else:
                logging.warning(f"⛔️ Invalid lane format at key {key}: {lane}")

    except Exception as e:
        logging.error(f"❌ Error drawing lanes: {e}")

    # try:
    #     edges = scene["lane_graph"].get("road_edges", {})
    #     for key, edge in edges.items():
    #         if isinstance(edge, np.ndarray) and edge.shape[1] >= 2:
    #             ax.plot(edge[:, 0], edge[:, 1], "b--", linewidth=0.5)
    #         else:
    #             logging.warning(f"⛔️ Invalid road edge format at key {key}: {edge}")
    # except Exception as e:
    #     logging.error(f"❌ Error drawing road_edges: {e}")
