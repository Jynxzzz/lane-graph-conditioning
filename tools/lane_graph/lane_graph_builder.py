# lane_graph_builder.py
import os

import networkx as nx
import numpy as np
from jynxzzzdebug import debug_break, debug_print, explore_dict, setup_logger

logging = setup_logger("lane_graph_builder", "logs/lane_graph_builder.log")


def extract_ego_subgraph(G, ego_lane_id, max_hops=3):
    """
    从 ego 所在 lane 向外扩展 max_hops 层得到子图。
    只保留方向正确的边（假设是 DiGraph）
    """
    nodes = set([ego_lane_id])
    frontier = set([ego_lane_id])
    logging.info(f"开始从 {ego_lane_id} 扩展子图，最大跳数 {max_hops}")

    for _ in range(max_hops):
        next_frontier = set()

        logging.info(f"当前 frontier: {frontier}")

        for n in frontier:
            debug_print("extract_ego_subgraph", f"扩展节点: {n}")
            neighbors = list(G.successors(n))  # 只向前扩展
            next_frontier.update(neighbors)
            logging.info(f"从 {n} 扩展到 {len(neighbors)} 个邻居: {neighbors}")
        logging.info(f"扩展到 {len(next_frontier)} 个节点")
        nodes.update(next_frontier)
        frontier = next_frontier

    return G.subgraph(nodes).copy()


def build_lane_graph(scenario):
    G = nx.DiGraph()

    pre_pairs = scenario["lane_graph"]["pre_pairs"]
    suc_pairs = scenario["lane_graph"]["suc_pairs"]
    left_pairs = scenario["lane_graph"]["left_pairs"]
    right_pairs = scenario["lane_graph"]["right_pairs"]

    all_lanes = (
        set(pre_pairs.keys())
        | set(suc_pairs.keys())
        | set(left_pairs.keys())
        | set(right_pairs.keys())
    )

    for lane_id in all_lanes:
        G.add_node(lane_id)

    # 添加边（按方向）
    for lane_id, preds in pre_pairs.items():
        for pred in preds:
            G.add_edge(pred, lane_id, type="pre")

    for lane_id, sucs in suc_pairs.items():
        for suc in sucs:
            G.add_edge(lane_id, suc, type="suc")

    for lane_id, lefts in left_pairs.items():
        for left in lefts:
            G.add_edge(lane_id, left, type="left")

    for lane_id, rights in right_pairs.items():
        for right in rights:
            G.add_edge(lane_id, right, type="right")

    return G


import matplotlib.pyplot as plt
import networkx as nx


def build_subgraph_with_features(lane_graph, nearby_ids):
    G = nx.DiGraph()
    for lane_id in nearby_ids:
        attr = lane_graph["lane_attrs"].get(lane_id, {})
        G.add_node(lane_id)
        G.nodes[lane_id]["centerline"] = lane_graph["lanes"][lane_id]
        G.nodes[lane_id]["turn_direction"] = attr.get("turn_direction", "none")
        G.nodes[lane_id]["has_traffic_light"] = attr.get("has_traffic_light", False)

    # 添加方向边
    for lane_id in nearby_ids:
        for suc in lane_graph["suc_pairs"].get(lane_id, []):
            if suc in nearby_ids:
                G.add_edge(lane_id, suc, type="suc")
    return G


def get_nearby_lane_ids(lane_graph, w2e, radius=50.0):
    nearby_ids = []
    for lane_id, lane_pts in lane_graph["lanes"].items():
        local_pts = w2e(lane_pts[:, :2])
        dists = np.linalg.norm(local_pts, axis=1)
        if np.any(dists < radius):
            nearby_ids.append(lane_id)
    return nearby_ids


def plot_lane_graph_dual(
    G, G_sub, real_pos, ego_pos=None, ego_lane_id=None, save_prefix="lane_graph"
):
    import os

    import matplotlib.pyplot as plt
    import networkx as nx

    def _draw_one(G, pos, title, save_path, show_edge_label=True):
        plt.figure(figsize=(8, 8))
        node_colors = []
        for node in G.nodes():
            if node == ego_lane_id:
                node_colors.append("orange")
            else:
                node_colors.append("lightblue")

        nx.draw(
            G,
            pos=pos,
            with_labels=False,
            node_color=node_colors,
            node_size=400,
            edge_color="gray",
            arrows=True,
            connectionstyle="arc3,rad=0.25",
        )
        # === 安全手动绘制边的 label（如 left / right / succ）
        # offset_map = {
        #     "left": (-0.2, 0.1),
        #     "right": (0.2, 0.1),
        #     "succ": (0.0, -0.2),
        #     "pred": (0.0, 0.2),
        # }
        if show_edge_label:
            edge_labels = nx.get_edge_attributes(G, "type")
            for (u, v), label in edge_labels.items():
                # 如果太远，跳过
                if u in pos and v in pos:
                    dist = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
                    if dist > 20:  # 可调
                        continue
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
                    plt.text(
                        xm,
                        ym,
                        label,
                        fontsize=6,
                        color="red",
                        ha="center",
                        va="center",
                        bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
                    )
        # if show_edge_label:
        #     edge_labels = nx.get_edge_attributes(G, "type")
        #     for (u, v), label in edge_labels.items():
        #         if u in pos and v in pos:
        #             x0, y0 = pos[u]
        #             x1, y1 = pos[v]
        #             xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        #             # dx, dy = offset_map.get(label, (0, 0))
        #             plt.text(
        #                 xm,
        #                 ym,
        #                 label,
        #                 fontsize=6,
        #                 color="red",
        #                 ha="center",
        #                 va="center",
        #                 bbox=dict(facecolor="white", alpha=0.4, edgecolor="none"),
        #             )

        # === 边标签（连接关系，如 succ / left / right）
        # === edge label
        labels = {}
        for node in G.nodes():
            if node == ego_lane_id:
                labels[node] = f"{node}\n(ego)"
            else:
                labels[node] = str(node)

        # 自动把 lane label 画到 node 中心
        nx.draw_networkx_labels(
            G, pos=pos, labels=labels, font_size=6, font_color="black"
        )

        # if ego_pos is not None:
        #     plt.plot(ego_pos[0], ego_pos[1], "ro", label="Ego Pos")
        #     plt.legend()

        plt.title(title)
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    # === 1. 真实 BEV 视图（以起点 w2e 为位置）
    save_real = f"{save_prefix}_real.png"
    spring_pos = nx.spring_layout(G_sub, seed=42, k=1.0)
    _draw_one(
        G_sub,
        # real_pos,
        spring_pos,
        "📍 Real Layout (Ego-local map)",
        save_real,
        show_edge_label=True,
    )

    # === 2. 拓扑结构视图（自动展开）
    spring_pos = nx.spring_layout(G, seed=42, k=1)
    save_struct = f"{save_prefix}_struct.png"
    _draw_one(
        G,
        spring_pos,
        "📎 Graph Structure Layout",
        save_struct,
        show_edge_label=False,
    )

    print(f"[✅] Saved dual graph views to:\n - {save_real}\n - {save_struct}")


# def safe_draw_lane_graph(
#     G, pos, save_path="lane_graph_sdc.png", ego_pos=None, ego_lane_id=None
# ):
#     import matplotlib.pyplot as plt
#
#     plt.figure(figsize=(10, 10))
#
#     # === 画出所有节点 ===
#     node_colors = []
#     for node in G.nodes():
#         if node == ego_lane_id:
#             node_colors.append("orange")  # 高亮 ego 所在车道
#         else:
#             node_colors.append("lightblue")
#
#     nx.draw(
#         G,
#         pos=pos,
#         with_labels=True,
#         node_size=500,
#         font_size=3,
#         edge_color="gray",
#         arrows=True,
#         connectionstyle="arc3,rad=0.4",
#     )
#     # === 手动标注每个节点的 lane_id ===
#     for node_id, (x, y) in pos.items():
#         label = f"{node_id}"
#         if node_id == ego_lane_id:
#             label += " (ego)"
#         plt.text(
#             x,
#             y + 0.5,
#             label,
#             fontsize=6,
#             ha="center",
#             va="center",
#             bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
#         )
#
#     # === 画边标签（pre, suc, left, right） ===
#     edge_labels = nx.get_edge_attributes(G, "type")
#     for (u, v), label in edge_labels.items():
#         if u in pos and v in pos:
#             x0, y0 = pos[u]
#             x1, y1 = pos[v]
#             xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
#             plt.text(
#                 xm,
#                 ym,
#                 label,
#                 fontsize=8,
#                 color="red",
#                 ha="center",
#                 va="center",
#                 bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
#             )
#     if ego_pos is not None:
#         plt.plot(ego_pos[0], ego_pos[1], "ro", label="Ego Pos")
#         plt.legend()
#
#     plt.axis("equal")
#     plt.title(f"Lane Graph around SDC @ {ego_pos}")
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close()


# def safe_draw_lane_graph(G, scenario, pos, save_path="lane_graph.png"):
#     plt.figure(figsize=(12, 12))
#
#     # 画图结构
#     nx.draw(
#         G,
#         pos=pos,
#         with_labels=True,
#         node_size=300,
#         font_size=4,
#         edge_color="gray",
#         arrows=True,
#         connectionstyle="arc3,rad=0.5",  # 弯曲一点点防止重叠
#     )
#
#     # 改用文字手动标注边类型
#     edge_labels = nx.get_edge_attributes(G, "type")
#     nx.draw_networkx_edge_labels(
#         G,
#         pos,
#         edge_labels=edge_labels,
#         font_size=4,
#         font_color="red",
#         rotate=False,
#         bbox=dict(facecolor="white", edgecolor="none", alpha=0),
#     )
#
#     # for (u, v), label in edge_labels.items():
#     #     if u in pos and v in pos:
#     #         x0, y0 = pos[u]
#     #         x1, y1 = pos[v]
#     #         xm, ym = (x0 + x1) / 2, (y0 + y1) / 2  # 中点位置
#     #         dx, dy = 0.5, 0.5  # 微调
#     #         plt.text(
#     #             xm + dx,  # 加小偏移
#     #             ym + dy,
#     #             label,
#     #             fontsize=8,
#     #             color="red",
#     #             ha="center",
#     #             va="center",
#     #             bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
#     #         )
#     # plt.text(
#     #     xm,
#     #     ym,
#     #     label,
#     #     fontsize=8,
#     #     color="red",
#     #     ha="center",
#     #     va="center",
#     #     bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
#     # )
#
#     plt.axis("equal")
#     plt.title("Lane Graph with Relation Labels")
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close()
