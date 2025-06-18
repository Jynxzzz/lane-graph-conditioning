import os
import pickle

import cv2
from render_utils.opencv_renderer import render_bev_opencv

# === 加载场景数据 ===
with open("data/training.tfrecord-00000-of-01000_9.pkl", "rb") as f:
    scene = pickle.load(f)

# === 输出路径 ===
output_dir = "rendered_frames"
os.makedirs(output_dir, exist_ok=True)

# === 渲染帧图像 ===
for i in range(90):
    img = render_bev_opencv(scene, frame_idx=i)
    save_path = os.path.join(output_dir, f"frame_{i:03d}.png")
    cv2.imwrite(save_path, img)

print("✅ 所有帧已成功渲染完成！")
