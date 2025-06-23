import matplotlib.pyplot as plt
import torch
from datasets.diffusion.components.compose import Compose
from datasets.diffusion.components.lane_graph import extract_lane_graph
from datasets.diffusion.components.transform import ExtractModelInput, ToBEV, ToTensor
from datasets.diffusion.trajectory_diffusion_dataset import TrajectoryDiffusionDataset
from jynxzzzdebug import debug_break, debug_print, explore_dict, setup_logger
from torch.utils.data import DataLoader
from torchvision import transforms

logging = setup_logger(
    "trajectory_diffusion", "logs/trajectory_diffusion.log", log_level="DEBUG"
)


import torch


def visualize_sample(sample):
    sdc = torch.tensor(sample["scenario"]["sdc_traj_bev"])
    raw_neighbors = sample["scenario"].get("neighbors_traj_bev", [])

    neighbors = []
    for i, traj in enumerate(raw_neighbors):
        try:
            if isinstance(traj, list) and all(
                torch.is_tensor(p) and p.ndim == 2 for p in traj
            ):
                stacked = torch.cat(traj, dim=0)
                neighbors.append(stacked)
            else:
                logging.warning(f"❌ 第 {i} 个邻居轨迹格式不合法: {traj}")
        except Exception as e:
            logging.warning(f"⚠️ 第 {i} 个邻居轨迹拼接失败: {e}")

    # 画图
    plt.plot(sdc[:, 0], sdc[:, 1], label="SDC", color="red", linewidth=2)
    for i, traj in enumerate(neighbors):
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.5, label=f"N{i}")
    plt.legend()
    plt.axis("equal")
    plt.title("SDC + Neighbors Trajectory (BEV)")
    plt.show()


if __name__ == "__main__":

    dataset = TrajectoryDiffusionDataset(
        list_path="scene_lists/green_only_list.txt",
        base_dir="/home/xingnan/VideoDataInbox/scenario_dreamer_waymo/train/",
        transform=Compose(
            [
                ToBEV(),
                ExtractModelInput(history_len=10, future_len=20),
                ToTensor(),
            ]
        ),
    )
    print(f"📦 数据集共计 {len(dataset)} 个场景")

    sample = dataset[0]
    logging.info(f"key s in sample: {sample.keys()}")

    logging.info(sample["scenario"]["sdc_traj_bev"])
    logging.info(f"SDC 轨迹 shape: {sample['scenario']['sdc_traj_bev']}")
    # G = lane_data["graph"]
    # print(f"🌐 Lane Graph 构建完成，节点数: {len(G.nodes)}, 边数: {len(G.edges)}")

    # 示例：看几个节点内容
    # for lane_id in list(G.nodes)[:3]:
    #     print(
    #         f"  Lane {lane_id} → centerline shape: {G.nodes[lane_id]['centerline'].shape}"
    #     )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for sample in dataloader:
        visualize_sample(sample)
        break  # 只跑一帧看结构
