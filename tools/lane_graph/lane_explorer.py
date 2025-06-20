import os

import networkx as nx
import numpy as np
from jynxzzzdebug import debug_break, debug_print, explore_dict, setup_logger

logging = setup_logger("lane_explorer", "logs/lane_explorer.log")

# def extract_ego_subgraph(G, ego_lane_id, max_hops=3):
#     """
#     从 ego 所在 lane 向外扩展 max_hops 层得到子图。
#     只保留方向正确的边（假设是 DiGraph）
#     """
#     nodes = set([ego_lane_id])
#     frontier = set([ego_lane_id])
#     logging.info(f"开始从 {ego_lane_id} 扩展子图，最大跳数 {max_hops}")
#
#     for _ in range(max_hops):
#         next_frontier = set()
#
#         logging.info(f"当前 frontier: {frontier}")
#
#         for n in frontier:
#             debug_print("extract_ego_subgraph", f"扩展节点: {n}")
#             neighbors = list(G.successors(n))  # 只向前扩展
#             next_frontier.update(neighbors)
#             logging.info(f"从 {n} 扩展到 {len(neighbors)} 个邻居: {neighbors}")
#         logging.info(f"扩展到 {len(next_frontier)} 个节点")
#         nodes.update(next_frontier)
#         frontier = next_frontier
#
#     return G.subgraph(nodes).copy()

import logging

import networkx as nx


def find_all_preds(lane_graph, target_id, max_hops=2):
    preds = set()
    frontier = {target_id}
    for _ in range(max_hops):
        new_frontier = set()
        for to_id in frontier:
            for from_id, to_list in lane_graph["suc_pairs"].items():
                if to_id in to_list:
                    if from_id not in preds:
                        logging.info(f"[Pred] {from_id} → {to_id}")
                        preds.add(from_id)
                        new_frontier.add(from_id)
        frontier = new_frontier
    return preds


def find_all_sucs_and_neighbors(lane_graph, ego_lane_id, max_hops=2):
    G = nx.DiGraph()
    visited = set()
    frontier = {ego_lane_id}

    for _ in range(max_hops):
        new_frontier = set()
        for curr in frontier:
            visited.add(curr)

            # === 后继边
            for suc in lane_graph["suc_pairs"].get(curr, []):
                if suc not in visited:
                    logging.info(f"[Suc] {curr} → {suc}")
                    G.add_edge(curr, suc, type="suc")
                    new_frontier.add(suc)

            # === 左右邻接道
            for dir_key in ["left_pairs", "right_pairs"]:
                for sid in lane_graph[dir_key].get(curr, []):
                    if sid not in visited:
                        logging.info(f"[{dir_key[:4]}] {curr} → {sid}")
                        G.add_edge(curr, sid, type=dir_key[:4])
                        new_frontier.add(sid)

        frontier = new_frontier
    return G


def build_waterflow_graph(lane_graph, ego_lane_id, max_hops=2):
    G = nx.DiGraph()
    G.add_node(ego_lane_id)

    # === Stage 1: 上游 predecessor → ego
    preds = find_all_preds(lane_graph, ego_lane_id, max_hops=max_hops)
    for p in preds:
        G.add_edge(p, ego_lane_id, type="pre")

    # === Stage 2: ego → successors + neighbors
    G_suc = find_all_sucs_and_neighbors(lane_graph, ego_lane_id, max_hops=max_hops)
    G.update(G_suc)

    return G


def build_directional_graph(lane_graph, ego_lane_id):
    G = nx.DiGraph()
    G.add_node(ego_lane_id)

    # === Stage 1: only add one-level predecessor → ego
    for from_id, to_list in lane_graph["suc_pairs"].items():
        if ego_lane_id in to_list:
            logging.info(f"找到 {from_id} 的后继是 {ego_lane_id}")
            G.add_edge(from_id, ego_lane_id, type="pre_to_sdc")
            # ❗ 不要给 pred 加 left/right
            # 也不要继续往前找 pred 的 pred！

    # === Stage 2: from SDC, add right, left, suc
    for suc_id in lane_graph["suc_pairs"].get(ego_lane_id, []):
        logging.info(f"添加 {ego_lane_id} 的后继 {suc_id}")
        G.add_edge(ego_lane_id, suc_id, type="sdc_to_suc")

    for direction in ["left_pairs", "right_pairs"]:
        logging.info(f"添加 {ego_lane_id} 的 {direction[:-6]} lane")
        for to_id in lane_graph[direction].get(ego_lane_id, []):
            logging.info(f"添加 {ego_lane_id} 的 {direction[:-6]} lane {to_id}")
            G.add_edge(ego_lane_id, to_id, type=f"sdc_to_{direction[:4]}")

    return G


def extract_ego_subgraph(lane_graph, ego_lane_id, max_hops=3):
    """
    从 lane_graph 中提取以 ego_lane_id 为起点的子图（只保留 max_hops 内的路径）。
    """
    import networkx as nx

    G = nx.DiGraph()

    # 1. 加入所有节点
    for lane_id, polyline in lane_graph["lanes"].items():
        logging.info(f"添加 lane {lane_id} 到图中")
        G.add_node(lane_id, polyline=polyline)

        # 加入 successor 边
    for from_id, to_list in lane_graph["suc_pairs"].items():
        logging.info(f"添加 {from_id} 的后继边到图中")
        for to_id in to_list:
            G.add_edge(from_id, to_id, type="suc")

    # 加入 left 边
    for from_id, to_list in lane_graph["left_pairs"].items():
        logging.info(f"添加 {from_id} 的左侧边到图中")
        for to_id in to_list:
            G.add_edge(from_id, to_id, type="left")

    # 加入 right 边
    for from_id, to_list in lane_graph["right_pairs"].items():
        logging.info(f"添加 {from_id} 的右侧边到图中")
        for to_id in to_list:
            G.add_edge(from_id, to_id, type="right")

    # （可选）加入 predecessor 边
    # for from_id, to_list in lane_graph["pre_pairs"].items():
    #     for to_id in to_list:
    #         G.add_edge(from_id, to_id, type="pre")

    # 3. 从 ego_lane_id 开始 BFS 拓展 max_hops
    visited = set()
    frontier = {ego_lane_id}
    nodes = set([ego_lane_id])

    for _ in range(max_hops):
        next_frontier = set()
        for node in frontier:
            visited.add(node)
            neighbors = list(G.successors(node))
            next_frontier.update(neighbors)
        nodes.update(next_frontier)
        frontier = next_frontier - visited

    # 4. 构造子图并返回
    return G.subgraph(nodes).copy()


def find_ego_lane_id(sdc_pos, lane_graph, threshold=2.0):
    """
    从所有 lane 中找到 SDC 所在的 lane_id
    - 使用最近点距离 + 投影判断
    """
    min_dist = float("inf")
    ego_lane_id = None
    for lane_id, lane_pts in lane_graph["lanes"].items():
        dists = np.linalg.norm(lane_pts[:, :2] - sdc_pos, axis=1)
        if dists.min() < min_dist and dists.min() < threshold:
            min_dist = dists.min()
            ego_lane_id = lane_id
    return ego_lane_id


def get_lane_traversal(lane_graph, ego_lane_id, max_depth=3):
    """
    构建 ego 所在 lane 的上下游结构（有向图）
    """
    visited = set()
    queue = [(ego_lane_id, 0)]
    traversal_ids = set()

    while queue:
        lane_id, depth = queue.pop(0)
        if lane_id in visited or depth > max_depth:
            continue
        visited.add(lane_id)
        traversal_ids.add(lane_id)

        # 添加后继（可加方向限制）
        successors = lane_graph["suc_pairs"].get(lane_id, [])
        for suc in successors:
            queue.append((suc, depth + 1))

    return traversal_ids


def get_controlled_lights(scenario, lane_ids):
    """
    从场景中找出控制这些 lane 的 traffic light id 和状态
    """
    controlled = {}
    for light in scenario["traffic_lights"]:
        control_lanes = light["lane_ids"]  # 某些格式下是控制的 lane_id 列表
        for lane_id in lane_ids:
            if lane_id in control_lanes:
                controlled[light["id"]] = light["state"]
    return controlled
