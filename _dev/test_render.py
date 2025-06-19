import json
import logging
import os
import pickle

from jynxzzzdebug import debug_break, debug_print, explore_dict, setup_logger
from tools.debug_scene_structure import explore_scene, print_scene_structure
from tools.loading_one_sceneario import load_random_scene, load_scene_data

from _dev.render_frame import render_bev_frame

# === 创建输出目录 ===
output_dir = "test_rendered_frames"
os.makedirs(output_dir, exist_ok=True)

scenario = load_random_scene()


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
