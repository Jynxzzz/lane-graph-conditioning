import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter


def animate_graph_growth(G, pos, node_stages, save_path="graph_growth.gif"):
    fig, ax = plt.subplots(figsize=(8, 6))
    shown_nodes = set()
    shown_edges = set()

    def update(frame):
        ax.clear()
        current_nodes = node_stages[frame]
        shown_nodes.update(current_nodes)

        # 更新边（只能画 shown_nodes 内部的边）
        for u, v, data in G.edges(data=True):
            if u in shown_nodes and v in shown_nodes:
                shown_edges.add((u, v))

        nx.draw_networkx_nodes(
            G, pos, nodelist=shown_nodes, ax=ax, node_color="skyblue"
        )
        nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in shown_nodes}, ax=ax)
        nx.draw_networkx_edges(
            G, pos, edgelist=list(shown_edges), ax=ax, edge_color="gray"
        )

        ax.set_title(f"Step {frame+1}/{len(node_stages)}")
        ax.axis("off")

    ani = FuncAnimation(
        fig, update, frames=len(node_stages), interval=1000, repeat=False
    )
    ani.save(save_path, writer=PillowWriter(fps=1))
    print(f"✅ Saved to {save_path}")
