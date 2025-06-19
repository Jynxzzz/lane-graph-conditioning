import json
import logging
import os
import pickle

from jynxzzzdebug import debug_break, debug_print, explore_dict, setup_logger
from tools.debug_scene_structure import explore_scene, print_scene_structure
from tools.scene_loader import (
    load_random_scene_from_list,
    load_scene_data,
    load_selected_scene_list,
)

from _dev.render_frame import render_bev_frame

# === 创建输出目录 ===
output_dir = "test_rendered_frames"
os.makedirs(output_dir, exist_ok=True)
import random

random.seed(42)  # ✅ 全局可复现
# load random scene
# scenario= load_scene_data()
# # 加载 green-only 列表
green_only_list = load_selected_scene_list(
    "/home/xingnan/scenario-dreamer/green_only_list.txt"
)
#
# # 随机读取一个 green-only 场景
scenario = load_random_scene_from_list(
    green_only_list,
    base_dir="/home/xingnan/VideoDataInbox/scenario_dreamer_waymo/train",
)


print_scene_structure(scenario, frame_idx=0)
debug_break()
debug_print("=== Debugging scene structure ===", "begin!")
explore_scene(scenario, frame_idx=0)
# explore_dict(scenario)


# === 渲染多帧（比如 0~89）===
for frame_idx in range(3):
    try:
        save_path = os.path.join(output_dir, f"frame_{frame_idx:03d}.png")
        render_bev_frame(scenario, frame_idx=frame_idx, save_path=save_path)
    except Exception as e:
        print(f"[❌ ERROR] Failed at frame {frame_idx}: {e}")
# for frame_idx in range(90):  # 9s数据，每秒10帧
#     save_path = os.path.join(output_dir, f"frame_{frame_idx:03d}.png")
#     render_bev_frame(scenario, frame_idx=frame_idx, save_path=save_path)

print("✅ 所有帧渲染完成！")
