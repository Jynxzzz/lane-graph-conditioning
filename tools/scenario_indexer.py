# tools/scenario_indexer.py
import os
import pickle
from collections import Counter
from enum import IntEnum

from jynxzzzdebug import debug_break, setup_logger

logging = setup_logger("scenario_indexer", "logs/scenario_indexer.log")

# === 1. 常量定义（文件顶部） ===
TRAFFIC_LIGHT_STATE = {
    0: "UNKNOWN",
    1: "ARROW_STOP",
    2: "ARROW_CAUTION",
    3: "ARROW_GO",
    4: "STOP",
    5: "CAUTION",
    6: "GO",
    7: "FLASHING_STOP",
    8: "FLASHING_CAUTION",
}


# def summarize_traffic_light_states(traffic_lights_frames):
#     from collections import Counter, defaultdict
#
#     lane_state_history = defaultdict(list)
#     global_state_counter = Counter()
#     all_states_set = set()
#
#     for frame in traffic_lights_frames:
#         for light in frame:
#             lane_id = light.get("lane")
#             state_idx = light.get("state")
#             state_str = TRAFFIC_LIGHT_STATE.get(state_idx, "INVALID")
#
#             lane_state_history[lane_id].append(state_str)
#             global_state_counter[state_str] += 1
#             all_states_set.add(state_str)
#
#     has_green = "GREEN" in all_states_set
#     has_red = "RED" in all_states_set
#     has_yellow = "YELLOW" in all_states_set
#     has_only_green = all_states_set == {"GREEN"}
#     has_all_states = all(s in all_states_set for s in ["GREEN", "RED", "YELLOW"])
#
#     return {
#         "total_lights": len(lane_state_history),
#         "state_counts": dict(global_state_counter),
#         "has_green": has_green,
#         "has_red": has_red,
#         "has_yellow": has_yellow,
#         "has_only_green": has_only_green,
#         "has_all_states": has_all_states,
#         "per_lane_states": dict(lane_state_history),
#         "state_description": (
#             "state_counts 表示每种交通灯状态在所有帧中出现的次数，"
#             "不等同于灯的数量，而是所有帧中出现频率的累计。"
#             "数据频率为 10Hz，因此较大的数值表示状态持续较久或多个灯反复出现。"
#         ),
#     }


def analyze_scenario(fpath, verbose=False):
    try:
        with open(fpath, "rb") as f:
            data = pickle.load(f)

        tags = []

        # === 信号类型标注 ===
        stop_signs = data.get("lane_graph", {}).get("stop_signs", [])
        if stop_signs:
            tags.append("stop_sign")

        traffic_lights = data.get("traffic_lights", [])
        has_light = any(len(f) > 0 for f in traffic_lights)
        if has_light:
            tags.append("traffic_light")

        if not tags:
            tags = ["others"]

        # === 计算交通统计量 ===
        traffic_stats = compute_traffic_stats(data)
        light_summary = summarize_traffic_light_states(
            data.get("traffic_lights", []), grid_size=0.5
        )
        # unique_lights, state_counts, lane2state, debug_locations = (
        #     analyze_traffic_lights(data.get("traffic_lights", []), grid_size=0.5)
        # )
        #
        # light_summary = {
        #     "total_lights": unique_lights,
        #     "traffic_light_state_freq_by_frame": state_counts,
        #     "has_green": "GO" in state_counts,
        #     "has_red": "STOP" in state_counts,
        #     "has_yellow": "CAUTION" in state_counts,
        #     "has_only_green": "GO" in state_counts and len(state_counts) == 1,
        #     "has_all_states": all(k in state_counts for k in ["GO", "STOP", "CAUTION"]),
        #     "per_lane_states": lane2state,
        # }
        #
        # if light_summary["has_only_green"]:
        #     tags.append("only_green_lights")
        # if not light_summary["has_red"]:
        #     tags.append("no_red_light")
        # if light_summary["has_all_states"]:
        #     tags.append("complex_light_pattern")
        # if len(light_summary["traffic_light_state_freq_by_frame"]) == 0:
        #     tags.append("no_light_state_info")

        # === 拥堵判定 ===
        def get_congestion_level(density, speed):
            if density > 40:
                return "severe"
            elif density > 25 or speed < 2.0:
                return "heavy"
            elif density > 15 or speed < 4.0:
                return "moderate"
            else:
                return "free"

        congestion_level = get_congestion_level(
            traffic_stats["density_veh_per_km"], traffic_stats["avg_speed"]
        )
        is_congested = congestion_level != "free"
        if is_congested:
            tags.append("congested")

        return {
            "path": fpath,
            "tags": tags,
            "corrupted": False,
            "stop_signs": len(stop_signs),
            "volume": traffic_stats["volume"],
            "avg_speed": round(traffic_stats["avg_speed"], 2),
            "road_length_km": round(traffic_stats["road_length_km"], 2),
            "density_veh_per_km": round(traffic_stats["density_veh_per_km"], 2),
            "scene_summary": {
                "has_signal": "traffic_light" in tags,
                "has_stop_sign": "stop_sign" in tags,
                "is_congested": "congested" in tags,
                "has_only_green": light_summary["has_only_green"],
                "has_all_states": light_summary["has_all_states"],
                "light_complexity": (
                    "complex"
                    if light_summary["has_all_states"]
                    else "green_only" if light_summary["has_only_green"] else "partial"
                ),
                "has_no_red": "no_red_light" in tags,
                "light_state_count": len(light_summary["state_counts"]),
                "lane_with_lights": len(light_summary["per_lane_states"]),
            },
            "traffic_lights": light_summary["total_lights"],
            "traffic_light_states": light_summary["state_counts"],
            "traffic_light_summary": {
                "has_green": light_summary["has_green"],
                "has_red": light_summary["has_red"],
                "has_yellow": light_summary["has_yellow"],
                "has_only_green": light_summary["has_only_green"],
                "has_all_states": light_summary["has_all_states"],
                "per_lane_states": light_summary["per_lane_states"],
                "state_description": light_summary["state_description"],
                **(
                    {
                        "debug_light_locations": light_summary["debug_light_locations"],
                        "state_history_per_lane": light_summary[
                            "state_history_per_lane"
                        ],
                    }
                    if verbose
                    else {}
                ),
            },
        }
    except Exception as e:
        return {
            "path": fpath,
            "tags": ["corrupted"],
            "corrupted": True,
            "stop_signs": 0,
            "traffic_lights": 0,
            "volume": 0,
            "avg_speed": 0,
            "road_length_km": 0,
            "density_veh_per_km": 0,
            "error": str(e),
        }


import numpy as np


def summarize_traffic_light_states(
    traffic_lights_frames, grid_size=0.5, frame_idx=None
):
    from collections import Counter, defaultdict

    unique_lights = set()
    state_counter = Counter()
    lane_id2latest_state = {}
    lane_id2state_seq = defaultdict(list)
    debug_light_locations = defaultdict(list)
    all_states_set = set()

    frames = (
        traffic_lights_frames
        if frame_idx is None
        else [traffic_lights_frames[frame_idx]]
    )
    logging.info(f"[Analyze] total frames: {len(frames)}")

    for frame in frames:
        for light in frame:
            lane_id = light.get("lane")
            state = light.get("state")
            state_str = TRAFFIC_LIGHT_STATE.get(state, "INVALID")

            # === 网格化 stop_point 去重 ===
            sp = light.get("stop_point", {})
            x = sp.get("x", 0.0)
            y = sp.get("y", 0.0)
            gx = int(x / grid_size)
            gy = int(y / grid_size)
            key = (lane_id, gx, gy)
            unique_lights.add(key)

            # === 状态统计 ===
            state_counter[state_str] += 1
            lane_id2latest_state[lane_id] = state_str
            lane_id2state_seq[lane_id].append(state_str)
            all_states_set.add(state_str)

            debug_light_locations[lane_id].append((x, y))

            logging.debug(
                f"[TrafficLight] lane={lane_id}, state={state_str}, pos=({x:.2f},{y:.2f})"
            )

    logging.info(f"[Analyze] Unique lights (lane + grid): {len(unique_lights)}")

    has_green = "GO" in all_states_set
    has_red = "STOP" in all_states_set
    has_yellow = "CAUTION" in all_states_set
    has_only_green = all_states_set == {"GO"}
    has_all_states = all(s in all_states_set for s in ["GO", "STOP", "CAUTION"])

    return {
        "total_lights": len(unique_lights),
        "state_counts": dict(state_counter),
        "has_green": has_green,
        "has_red": has_red,
        "has_yellow": has_yellow,
        "has_only_green": has_only_green,
        "has_all_states": has_all_states,
        "per_lane_states": dict(lane_id2latest_state),
        "state_history_per_lane": dict(lane_id2state_seq),
        "debug_light_locations": dict(debug_light_locations),
        "state_description": (
            "state_counts shows the total frequency of each traffic light state "
            "across all frames (not the number of lights), "
            "while state_history_per_lane records the per-frame state sequence for each lane_id."
        ),
    }


# def analyze_traffic_lights(traffic_lights, frame_idx=None, grid_size=0.5):
#     from collections import Counter, defaultdict
#
#     unique_lights = set()
#     state_counter = Counter()
#     lane_id2state = {}
#     debug_light_locations = defaultdict(list)
#
#     frames = traffic_lights if frame_idx is None else [traffic_lights[frame_idx]]
#     logging.info(f"[Analyze] total frames: {len(frames)}")
#
#     for frame in frames:
#         for light in frame:
#             lane_id = light.get("lane")
#             state = light.get("state")
#             state_str = TRAFFIC_LIGHT_STATE.get(state, "INVALID")
#
#             # === 网格化 stop_point ===
#             sp = light.get("stop_point", {})
#             x = sp.get("x", 0.0)
#             y = sp.get("y", 0.0)
#             gx = int(x / grid_size)
#             gy = int(y / grid_size)
#
#             key = (lane_id, gx, gy)
#             unique_lights.add(key)
#
#             # === 状态统计 ===
#             state_counter[state_str] += 1
#             lane_id2state[lane_id] = state_str
#
#             # === Debug: 记录原始 stop_point 位置（便于后续分析误差范围）
#             debug_light_locations[lane_id].append((x, y))
#
#             logging.debug(
#                 f"[TrafficLight] lane={lane_id}, state={state_str}, pos=({x:.2f},{y:.2f})"
#             )
#
#     logging.info(f"[Analyze] Unique lights (lane + grid): {len(unique_lights)}")
#
#     return len(unique_lights), dict(state_counter), lane_id2state, debug_light_locations


# def analyze_traffic_lights(traffic_lights, frame_idx=None):
#     from collections import Counter
#
#     unique_lights = set()
#     state_counter = Counter()
#     lane_id2state = {}
#
#     frames = traffic_lights if frame_idx is None else [traffic_lights[frame_idx]]
#     logging.info(f"length of traffic_lights frames: {len(frames)}")
#
#     for frame in frames:
#         for light in frame:
#             lane_id = light.get("lane")
#             state = light.get("state")
#             state_str = TRAFFIC_LIGHT_STATE.get(state, "INVALID")
#
#             # === 唯一灯数量统计 ===
#             unique_lights.add(lane_id)
#
#             # === 状态计数器（绿、红、黄） ===
#             state_counter[state_str] += 1
#
#             # === 每个 lane 对应的最新状态 ===
#             # 注意：如果有多个状态，只记录最后出现的（你可以改成列表形式）
#             lane_id2state[lane_id] = state_str
#
#             logging.info(f"Traffic light state: {state_str} for lane {lane_id}")
#
#     logging.info(f"Unique lane-based traffic lights: {len(unique_lights)}")
#
#     return len(unique_lights), dict(state_counter), lane_id2state


def compute_road_length_km(lanes: dict):
    total = 0.0
    for lane in lanes.values():
        if isinstance(lane, np.ndarray) and lane.shape[0] >= 2:
            diffs = lane[1:] - lane[:-1]
            lengths = np.linalg.norm(diffs, axis=1)
            total += np.sum(lengths)
    return total / 1000  # m → km


import math
from collections import defaultdict


def compute_traffic_stats(scene):
    import math
    from collections import Counter, defaultdict

    objects = scene.get("objects", [])
    lanes = scene.get("lane_graph", {}).get("lanes", {})

    type_counter = defaultdict(int)
    total_speed = 0.0
    speed_count = 0

    for obj in objects:
        obj_type = obj.get("type", "unset").lower()
        type_counter[obj_type] += 1
        if obj_type == "vehicle":
            velocity_list = obj.get("velocity", [])
            if isinstance(velocity_list, list) and len(velocity_list) > 0:
                v = velocity_list[0]
                if isinstance(v, dict) and "x" in v and "y" in v:
                    vx, vy = v["x"], v["y"]
                    if abs(vx) < 1000 and abs(vy) < 1000:
                        speed = math.hypot(vx, vy)
                        total_speed += speed
                        speed_count += 1

    volume = type_counter["vehicle"]
    avg_speed = total_speed / speed_count if speed_count > 0 else 0.0
    road_length_km = compute_road_length_km(lanes)
    density_veh_per_km = volume / road_length_km if road_length_km > 0 else 0.0

    stop_signs = len(scene.get("lane_graph", {}).get("stop_signs", []))

    # === Traffic light 分析 ===
    # unique_traffic_light_count, traffic_light_state_counter, traffic_light_map = (
    #     analyze_traffic_lights(traffic_lights, frame_idx=0)
    # )

    return {
        "stop_signs": stop_signs,
        "volume": volume,
        "avg_speed": avg_speed,
        "road_length_km": road_length_km,
        "density_veh_per_km": density_veh_per_km,
        "type_distribution": dict(type_counter),
    }
