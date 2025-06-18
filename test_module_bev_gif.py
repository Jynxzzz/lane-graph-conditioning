import json
import logging
import os
import pickle

from bev_renderer.render_frame import render_bev_frame

# === 配置日志 ===
logging.basicConfig(level=logging.INFO)

with open(
    "/home/xingnan/VideoDataInbox/scenario_dreamer_waymo/train/training.tfrecord-00000-of-01000_9.pkl",
    "rb",
) as f:
    scene = pickle.load(f)


# === 创建输出目录 ===
output_dir = "rendered_frames"
os.makedirs(output_dir, exist_ok=True)

# === 渲染多帧（比如 0~89）===
for frame_idx in range(90):  # 9s数据，每秒10帧
    save_path = os.path.join(output_dir, f"frame_{frame_idx:03d}.png")
    render_bev_frame(scene, frame_idx=frame_idx, save_path=save_path)

print("✅ 所有帧渲染完成！")
