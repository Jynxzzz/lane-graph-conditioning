import os

# === 创建输出目录 ===
import random

import hydra
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from _dev.candy_lane_graph import extract_ego_info, plot_lane_graph
from _dev.ego_utils import build_ego_centered_context
from _dev.render_frame import build_local_transform, render_bev_frame
from jynxzzzdebug import debug_break, debug_print, explore_dict, setup_logger
from omegaconf import DictConfig, OmegaConf
from tools.debug_scene_structure import explore_scene, print_scene_structure

# 调用 encoder
from tools.encoder import build_encoder
from tools.lane_graph.lane_explorer import (  # find_all_preds,
    build_directional_graph,
    build_waterflow_graph,
    extract_ego_subgraph,
    find_ego_lane_id,
)
from tools.lane_graph.lane_graph_builder import (  # safe_draw_lane_graph,
    build_lane_graph,
    get_nearby_lane_ids,
    plot_lane_graph_dual,
)
from tools.scene_loader import (
    load_random_scene_from_list,
    load_scene_data,
    load_selected_scene_list,
)
from utils.graph_pop_animation import animate_graph_growth

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
    import numpy as np

    ego, ego_pos, ego_heading = extract_ego_info(scenario, frame_idx=0)

    w2e = build_local_transform(ego_pos, ego_heading)
    # 在调用 safe_draw_lane_graph 前加上
    ego_local = w2e(np.array([ego_pos]))[0]  # 转为相对 ego 的坐标系
    sdc_xy = np.array([ego_pos[0], ego_pos[1]])

    ego_lane_id = find_ego_lane_id(sdc_xy, scenario["lane_graph"])
    logging.info(f"sdc lane id: {ego_lane_id}")
    debug_print("main", f"ego lane id: {ego_lane_id}")
    # === 获取 ego frame 下的 lane 起点
    G = build_lane_graph(scenario)
    G_full = G
    G_sub = build_waterflow_graph(scenario["lane_graph"], ego_lane_id)
    G_sub, node_stages = build_waterflow_graph(scenario["lane_graph"], ego_lane_id)
    pos = nx.spring_layout(G_sub, seed=42)
    # 调用生成动画函数
    # animate_graph_growth(G_sub, pos, node_stages, save_path="lane_wave.gif")
    # debug_break("=== Debugging lane graph structure ===")
    pos = {}
    for lane_id in G.nodes:
        lane_pts = scenario["lane_graph"]["lanes"].get(lane_id)
        if lane_pts is not None and len(lane_pts) > 0:
            local_pts = w2e(lane_pts[:, :2])
            pos[lane_id] = tuple(local_pts[0])  # ✅ now in SDC-centered coord
    nearby_ids = get_nearby_lane_ids(scenario["lane_graph"], w2e, radius=25.0)

    plot_lane_graph_dual(
        G_full,
        G_sub,
        real_pos=pos,  # 你原来算好的 pos
        ego_pos=ego_local,  # 如果有 w2e 转过的 ego_pos
        ego_lane_id=102,
        save_prefix="lane_graph_sdc",
    )

    # debug_break("=== Debugging lane graph structure ===")
    for frame_idx in range(1):
        try:
            save_path = os.path.join(output_dir, f"frame_{frame_idx:03d}.png")
            # test lane graph
            debug_print("=== Debugging lane graph structure ===", "begin!")
            # build_ego_centered_context(scenario, frame_idx, radius=50.0)
            #
            # debug_print("===  Debugging lane graph structure ===", "end!")

            # render_bev_frame(
            #     scenario, frame_idx=frame_idx, save_path=save_path, mode="encode"
            # )
            # render_bev_frame(scenario, frame_idx=frame_idx, save_path=save_path)
            plot_lane_graph(scenario, frame_idx=frame_idx, save_path=save_path)
        except Exception as e:
            print(f"[❌ ERROR] Failed at frame {frame_idx}: {e}")


if __name__ == "__main__":
    main()
