import os

import imageio

image_dir = "test_candy_rendered_frames"
frames = []
for i in range(90):
    filename = os.path.join(image_dir, f"frame_{i:03d}.png")
    if os.path.exists(filename):
        frames.append(imageio.imread(filename))

gif_path = "bev_debug_lane_encoded_candy.gif"
imageio.mimsave(gif_path, frames, fps=10)
print(f"✅ GIF 已保存：{gif_path}")
