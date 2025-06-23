# obsitraj/datasets/components/lane_graph.py

import networkx as nx
import numpy as np


def extract_lane_graph(scene: dict) -> dict:
    from collections import defaultdict

    import networkx as nx
    import numpy as np

    lane_dict = scene["lane_graph"]["lanes"]
    successors = scene["lane_graph"].get("successors", {})
    left_pairs = scene["lane_graph"].get("left_neighbors", {})
    right_pairs = scene["lane_graph"].get("right_neighbors", {})

    # === 自动构建 predecessor（爸爸们）
    predecessors = defaultdict(list)
    for src, dsts in successors.items():
        for dst in dsts:
            predecessors[dst].append(src)

    G = nx.DiGraph()
    centerlines = {}
    node_positions = {}
    node_features = {}

    for lane_id, points in lane_dict.items():
        centerline = np.array(points)  # shape: (N, 2)
        if centerline.shape[0] < 2:
            continue  # 跳过无效lane
        G.add_node(lane_id)
        centerlines[lane_id] = centerline
        node_positions[lane_id] = centerline.mean(axis=0)

        # 可选 node 特征（方向向量、长度）
        direction = centerline[-1] - centerline[0]
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        node_features[lane_id] = {
            "direction": direction,
            "length": np.linalg.norm(centerline[-1] - centerline[0]),
        }

    # === 添加边
    for src, dsts in successors.items():
        for dst in dsts:
            if src in G and dst in G:
                G.add_edge(src, dst, type="successor")

    for src, dsts in predecessors.items():
        for dst in dsts:
            if src in G and dst in G:
                G.add_edge(src, dst, type="predecessor")

    for src, dsts in left_pairs.items():
        for dst in dsts:
            if src in G and dst in G:
                G.add_edge(src, dst, type="left_adj")

    for src, dsts in right_pairs.items():
        for dst in dsts:
            if src in G and dst in G:
                G.add_edge(src, dst, type="right_adj")

    return {
        "graph": G,
        "lane_id_list": list(G.nodes),
        "centerlines": centerlines,
        "node_positions": node_positions,
        "node_features": node_features,
        "successors": successors,
        "predecessors": dict(predecessors),
        "left_neighbors": left_pairs,
        "right_neighbors": right_pairs,
    }


# def extract_lane_graph(scene: dict) -> dict:
#     import networkx as nx
#     import numpy as np
#
#     G = nx.DiGraph()
#     lane_dict = scene.get("lane_graph", {}).get("lanes", {})
#
#     for lane_id, pts in lane_dict.items():
#         pts = np.array(pts)
#         if pts.shape[0] < 2:
#             continue  # 跳过太短的 lane
#
#         # 加入节点 + 属性
#         G.add_node(lane_id, centerline=pts[:, :2])  # 💡 只保留 x, y
#         # 暂不加 edge
#
#     return {
#         "graph": G,
#         "node_features": None,  # 后面可加入方向向量、类型等
#         "node_positions": None,  # 可设为 pts.mean(axis=0)
#         "lane_id_list": list(G.nodes),
#     }
