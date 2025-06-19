import os

# === 创建输出目录 ===
import random

import hydra
from jynxzzzdebug import debug_break, debug_print, explore_dict, setup_logger
from omegaconf import DictConfig, OmegaConf

from _dev.candy_lane_graph import plot_lane_graph
from _dev.render_frame import render_bev_frame
from tools.debug_scene_structure import explore_scene, print_scene_structure

# 调用 encoder
from tools.encoder import build_encoder
from tools.scene_loader import (
    load_random_scene_from_list,
    load_scene_data,
    load_selected_scene_list,
)

random.seed(42)  # ✅ 全局可复现
# load random scene
# scenario= load_scene_data()
# # 加载 green-only 列表
logging = setup_logger("main", "logs/main.log")
from _dev.render_frame import render_bev_frame


# === 创建输出目录 ===
@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    output_dir = "test_candy_rendered_frames"
    os.makedirs(output_dir, exist_ok=True)
    logging.info("🚀 启动 Scenario Dreamer 测试...")

    random.seed(cfg.seed)
    # === 加载世界（你也可以换成 load_selected_scene）
    green_only_list = load_selected_scene_list(
        "/home/xingnan/scenario-dreamer/green_only_list.txt"
    )
    #
    # # 随机读取一个 green-only 场景
    scenario = load_random_scene_from_list(
        green_only_list,
        base_dir="/home/xingnan/VideoDataInbox/scenario_dreamer_waymo/train",
    )

    # === 构造 encoder 实例
    encoder = build_encoder(cfg.encoder.name)

    # encoding
    tokens, lane_token_map = encoder.encode(scenario, cfg.encoder)
    scenario["lane_token_map"] = lane_token_map
    scenario["lane_tokens"] = tokens

    logging.info(f"编码:10结果：{tokens[:10]}")
    logging.info(f"编码结果长度：{len(tokens)}")

    debug_print("=== Debugging scene structure ===", "begin!")

    for frame_idx in range(90):
        try:
            save_path = os.path.join(output_dir, f"frame_{frame_idx:03d}.png")
            # render_bev_frame(
            #     scenario, frame_idx=frame_idx, save_path=save_path, mode="encode"
            # )
            # render_bev_frame(scenario, frame_idx=frame_idx, save_path=save_path)
            plot_lane_graph(scenario, frame_idx=frame_idx, save_path=save_path)
        except Exception as e:
            print(f"[❌ ERROR] Failed at frame {frame_idx}: {e}")


if __name__ == "__main__":
    main()
