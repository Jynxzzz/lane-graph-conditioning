# main_model.py

import os
import random

import hydra
import numpy as np
from jynxzzzdebug import setup_logger
from omegaconf import DictConfig

from tools.encoder import build_encoder
from tools.lane_graph.lane_explorer import find_ego_lane_id
from tools.lane_graph.lane_graph_builder import build_lane_graph
from tools.scene_loader import load_random_scene_from_list, load_selected_scene_list

logger = setup_logger("model_main", "logs/model_main.log")


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    random.seed(cfg.seed)
    output_dir = "model_outputs"
    os.makedirs(output_dir, exist_ok=True)

    logger.info("🚗 启动建模流程...")

    # === 加载场景 ===
    scene_list = load_selected_scene_list(cfg.scene.scene_list_path)
    scenario = load_random_scene_from_list(scene_list, base_dir=cfg.scene.base_dir)

    # === 编码器
    encoder = build_encoder(cfg.encoder.name)
    tokens, lane_token_map = encoder.encode(scenario, cfg.encoder)

    # === 加入编码结果
    scenario["lane_tokens"] = tokens
    scenario["lane_token_map"] = lane_token_map

    # === 构建 lane graph
    G = build_lane_graph(scenario)

    # === 找到 ego 所在 lane
    ego_pos = scenario["objects"][0]["center"][:2]
    ego_lane_id = find_ego_lane_id(ego_pos, scenario["lane_graph"])

    logger.info(f"✅ SDC 当前 lane_id = {ego_lane_id}")
    # 👉 你可以在这里接入建模逻辑，如提取邻居 lane、构建子图、送入模型等

    # === 模型训练 / 推理流程
    # TODO: 调用你的训练函数，比如：
    # train_model(G, tokens, lane_token_map, ego_lane_id, ...)


if __name__ == "__main__":
    main()
