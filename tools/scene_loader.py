import logging
import pickle
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from bev_renderer.lane_graph import draw_lane_graph, draw_lanes_near_sdc
from jynxzzzdebug import debug_break, setup_logger

DEFAULT_SCENE_PATH = "/home/xingnan/VideoDataInbox/scenario_dreamer_waymo/train/training.tfrecord-00000-of-01000_5.pkl"


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


def load_selected_scene_list(list_path: str) -> list:
    """
    从 txt 或 jsonl 文件中加载场景路径列表
    每行一个路径（可以是相对路径）
    """
    try:
        with open(list_path, "r") as f:
            scene_list = [line.strip() for line in f if line.strip()]
        return scene_list
    except Exception as e:
        logging.error(f"❌ Error loading scene list from {list_path}: {e}")
        return []


def load_random_scene_from_list(scene_list: list, base_dir: str) -> dict:
    """
    从给定路径列表中随机选择一个 scene.pkl 加载
    - scene_list: 场景相对路径列表
    - base_dir: 所有路径的根目录
    """
    try:
        if not scene_list:
            logging.warning("📭 Scene list is empty")
            return {}

        chosen = random.choice(scene_list)
        full_path = os.path.join(base_dir, chosen)

        with open(full_path, "rb") as f:
            scene = pickle.load(f)

        logging.info(f"🎯 Loaded scene: {chosen}")
        return scene

    except Exception as e:
        logging.error(f"❌ Error loading scene from list: {e}")
        return {}


import logging
import os
import pickle
import random

if __name__ == "__main__":
    result = analyze_scenario_directory()
    for k, v in result.items():
        print(f"{k}: {v}")
