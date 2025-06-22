# main_model.py

import json
import os
import random
from pathlib import Path

import hydra
import numpy as np
from _dev.candy_lane_graph import extract_ego_info, plot_lane_graph
from jynxzzzdebug import (
    debug_break,
    debug_print,
    explore_dict,
    generate_paths,
    setup_logger,
)
from omegaconf import DictConfig
from tools.encoder import build_encoder
from tools.lane_graph.lane_explorer import build_waterflow_graph, find_ego_lane_id
from tools.lane_graph.lane_graph_builder import build_lane_graph
from tools.scene_loader import load_random_scene_from_list, load_selected_scene_list

from utils.traj_processing import extract_sdc_and_neighbors

logging = setup_logger("test_utils", "logs/test_utils.log")


def test_multiple_scenarios(cfg: DictConfig):
    random.seed(cfg.seed)
    output_dir = Path(cfg.test.test_batch_frame_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 场景列表加载与打乱
    scene_list = load_selected_scene_list(cfg.scene.scene_list_path)
    random.shuffle(scene_list)

    # 初始化编码器
    encoder = build_encoder(cfg.encoder.name)

    # token 分布统计输出路径
    token_stats_file = output_dir / "token_stats.jsonl"
    with open(token_stats_file, "w") as stats_f:

        for idx, scene_name in enumerate(scene_list):
            try:
                scenario = load_random_scene_from_list(
                    [scene_name], base_dir=cfg.scene.base_dir
                )

                if scenario.get("corrupted", False):
                    logging.warning(f"⛔ 跳过损坏场景: {scene_name}")
                    continue

                # === 编码车道
                lane_tokens, lane_token_map = encoder.encode_lanes(scenario)
                scenario["lane_tokens"] = lane_tokens
                scenario["lane_token_map"] = lane_token_map

                # === 编码红绿灯
                traffic_tokens, traffic_token_map = encoder.encode_traffic_lights(
                    scenario, frame_idx=0
                )
                scenario["traffic_light_tokens"] = traffic_tokens
                scenario["traffic_light_token_map"] = traffic_token_map

                # === 保存图像
                save_path = output_dir / f"lane_debug_{idx:03d}.png"
                plot_lane_graph(
                    scenario,
                    radius=cfg.scene.radius,
                    frame_idx=0,
                    save_path=str(save_path),
                )

                # === token 统计
                token_count = {}
                for t in lane_tokens:
                    token_count[str(t)] = token_count.get(str(t), 0) + 1
                stats_entry = {
                    "scene": scene_name,
                    "token_counts": token_count,
                    "num_tokens": len(lane_tokens),
                }
                stats_f.write(json.dumps(stats_entry) + "\n")

                logging.info(f"✅ 场景 {scene_name} 编码完成，图像保存在 {save_path}")

            except Exception as e:
                logging.warning(f"⚠️ 处理场景 {scene_name} 失败: {e}")
                continue

    logging.info(f"🎉 所有场景完成！统计已保存到: {token_stats_file}")


def test_single_scenario(cfg: DictConfig):
    import os
    import random
    from pathlib import Path

    random.seed(cfg.seed)
    output_dir = Path(cfg.test.test_single_frame_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("🚗 启动单场景测试流程...")

    # === 加载并选定一个场景 ===
    scene_list = load_selected_scene_list(cfg.scene.scene_list_path)
    scenario = load_random_scene_from_list(scene_list, base_dir=cfg.scene.base_dir)

    if scenario.get("corrupted", False):
        logging.warning("⚠️ 当前场景损坏，跳过")
        return

    result = extract_sdc_and_neighbors(scenario, max_distance=20.0, frame_idx=0)

    # logging.info(f"🚗 SDC轨迹长度: {len(result['sdc_traj'])}")
    # logging.info(f"🚙 周围邻居数: {len(result['neighbor_ids'])}")
    # for nid, traj in result["neighbor_trajs"].items():
    #     logging.info(f"  🔹 Neighbor {nid} 轨迹长度: {len(traj)}")

    # === 构建编码器
    encoder = build_encoder(cfg.encoder.name)
    # objects traj:
    agent_tokens = encoder.encode_agents(scenario, frame_idx=0)

    # === 编码车道
    tokens, lane_token_map = encoder.encode_lanes(scenario)
    scenario["lane_tokens"] = tokens
    scenario["lane_token_map"] = lane_token_map

    # === 编码红绿灯
    traffic_tokens, traffic_token_map = encoder.encode_traffic_lights(
        scenario, frame_idx=0
    )
    scenario["traffic_light_tokens"] = traffic_tokens
    scenario["traffic_light_token_map"] = traffic_token_map

    # === 保存图像
    frame_idx = cfg.test.test_frame_idx
    filename = f"lane_debug_{frame_idx:03d}.png"
    save_path = output_dir / filename
    plot_lane_graph(
        scenario, radius=cfg.scene.radius, frame_idx=frame_idx, save_path=str(save_path)
    )

    logging.info(f"✅ 单场景 lane 图保存至: {save_path}")
    logging.info(f"🔢 编码 token 总数: {len(tokens)}")
