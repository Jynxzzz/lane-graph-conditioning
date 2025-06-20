# main_model.py

import os
import random

import hydra
import numpy as np
from jynxzzzdebug import generate_paths, setup_logger
from omegaconf import DictConfig

from _dev.candy_lane_graph import extract_ego_info, plot_lane_graph
from tools.encoder import build_encoder
from tools.lane_graph.lane_explorer import build_waterflow_graph, find_ego_lane_id
from tools.lane_graph.lane_graph_builder import build_lane_graph
from tools.scene_loader import load_random_scene_from_list, load_selected_scene_list
from utils.test_utils import test_multiple_scenarios, test_single_scenario

logger = setup_logger("model_main", "logs/model_main.log")


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.test.run_batch:
        test_multiple_scenarios(cfg)
    else:
        test_single_scenario(cfg)
    # random.seed(cfg.seed)
    # output_dir = "model_outputs"
    # save_path = cfg.test.test_frame_path
    # # 拼接完整文件名
    # os.makedirs(save_path, exist_ok=True)
    #
    # logger.info("🚗 启动建模流程...")
    # frame_idx = cfg.test.test_frame_idx
    #
    # filename = f"lane_debug_{frame_idx:03d}.png"  # 可改成其他名字，比如带 ID 的
    # save_path = os.path.join(save_path, filename)
    # os.makedirs(cfg.test.test_frame_path, exist_ok=True)
    #
    # # === 加载场景 ===
    # scene_list = load_selected_scene_list(cfg.scene.scene_list_path)
    # scenario = load_random_scene_from_list(scene_list, base_dir=cfg.scene.base_dir)
    #
    # # === 编码器
    # encoder = build_encoder(cfg.encoder.name)
    # tokens, lane_token_map = encoder.encode(scenario, cfg.encoder)
    #
    # # === 加入编码结果
    # scenario["lane_tokens"] = tokens
    # scenario["lane_token_map"] = lane_token_map
    #
    # plot_lane_graph(scenario, frame_idx=frame_idx, save_path=save_path)
    # # === 模型训练 / 推理流程
    # # TODO: 调用你的训练函数，比如：
    # # train_model(G, tokens, lane_token_map, ego_lane_id, ...)


if __name__ == "__main__":
    main()
