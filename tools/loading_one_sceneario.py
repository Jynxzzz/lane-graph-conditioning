import logging
import pickle
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from bev_renderer.lane_graph import draw_lane_graph, draw_lanes_near_sdc
from jynxzzzdebug import debug_break, setup_logger

DEFAULT_SCENE_PATH = "/home/xingnan/VideoDataInbox/scenario_dreamer_waymo/train/training.tfrecord-00000-of-01000_5.pkl"


deug = False


def load_scene_data(path: Optional[str] = None) -> Dict:
    """加载指定路径的 scene 数据，如果未指定则使用默认测试路径"""
    path = path or DEFAULT_SCENE_PATH  # fallback
    try:
        with open(path, "rb") as f:
            scene = pickle.load(f)
        logging.info(f"✅ Loaded scene from: {path}")
        return scene
    except Exception as e:
        logging.error(f"❌ Failed to load scene from {path}: {e}")
        return {}


import logging
import os
import pickle
import random
from typing import Dict, Optional

DEFAULT_SCENE_DIR = "/home/xingnan/VideoDataInbox/scenario_dreamer_waymo/train"


def load_random_scene(path: Optional[str] = None) -> Dict:
    """从指定目录中随机加载一个 scene.pkl 文件"""
    scene_dir = path or DEFAULT_SCENE_DIR

    try:
        pkl_files = [f for f in os.listdir(scene_dir) if f.endswith(".pkl")]
        if not pkl_files:
            logging.warning(f"📭 No .pkl files found in {scene_dir}")
            return {}

        chosen_file = random.choice(pkl_files)
        full_path = os.path.join(scene_dir, chosen_file)

        with open(full_path, "rb") as f:
            scene = pickle.load(f)

        logging.info(f"🎯 Loaded random scene: {chosen_file}")
        return scene

    except Exception as e:
        logging.error(f"❌ Error loading random scene from {scene_dir}: {e}")
        return {}


def analyze_scenario_directory(path: Optional[str] = None):
    stats = {
        "total": 0,
        "with_stop_sign": 0,
        "with_traffic_light": 0,
        "others": 0,
        "corrupted": 0,
    }

    scene_dir = path or DEFAULT_SCENE_DIR

    for fname in os.listdir(scene_dir):
        if not fname.endswith(".pkl"):
            continue

        fpath = os.path.join(scene_dir, fname)
        try:
            with open(fpath, "rb") as f:
                scenario = pickle.load(f)

            stats["total"] += 1

            has_stop = bool(scenario.get("lane_graph", {}).get("stop_signs"))
            has_light = any(len(f) > 0 for f in scenario.get("traffic_lights", []))
            # has_light = bool(scenario.get("traffic_lights"))

            if has_light:
                stats["with_traffic_light"] += 1
            elif has_stop:
                stats["with_stop_sign"] += 1
            else:
                stats["others"] += 1
        except Exception as e:
            logging.warning(f"❌ Failed to load {fname}: {e}")
            stats["corrupted"] += 1

    return stats


if __name__ == "__main__":
    result = analyze_scenario_directory()
    for k, v in result.items():
        print(f"{k}: {v}")
