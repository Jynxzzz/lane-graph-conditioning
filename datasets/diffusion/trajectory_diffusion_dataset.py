# trajectory_diffusion_dataset.py
import os
import pickle

from jynxzzzdebug import setup_logger

logging = setup_logger(
    "trajectory_diffusion_dataset", "logs/trajectory_diffusion_dataset.log"
)

import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDiffusionDataset(Dataset):
    def __init__(self, list_path, base_dir, transform=None):
        with open(list_path, "r") as f:
            self.valid_scene_list = [line.strip() for line in f if line.strip()]
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.valid_scene_list)

    def __getitem__(self, idx):
        rel_path = self.valid_scene_list[idx]
        full_path = os.path.join(self.base_dir, rel_path)
        with open(full_path, "rb") as f:
            scenario = pickle.load(f)

        if scenario.get("corrupted", False):
            raise ValueError(f"⚠️ Scenario {rel_path} is corrupted.")

        sample = {
            "scenario": scenario,
            "rel_path": rel_path,
            "idx": idx,
        }
        return self.transform(sample) if self.transform else sample

    # def __getitem__(self, idx):
    #     rel_path = self.valid_scene_list[idx]
    #     full_path = os.path.join(self.base_dir, rel_path)
    #
    #     with open(full_path, "rb") as f:
    #         scenario = pickle.load(f)
    #
    #     if scenario.get("corrupted", False):
    #         raise ValueError(f"⚠️ Scenario {rel_path} is corrupted.")
    #
    #     # === 提取 ego 信息并构建 world-to-ego 坐标变换函数 ===
    #     ego, ego_pos, ego_heading_deg = extract_ego_info(scenario, frame_idx=0)
    #     w2e = build_local_transform(ego_pos, ego_heading_deg)
    #
    #     # === SDC 轨迹 ===
    #     result = extract_sdc_and_neighbors(scenario, max_distance=50.0, frame_idx=0)
    #     sdc_traj_world = np.array(result["sdc_traj"], dtype=np.float32)  # (91, 2)
    #     sdc_traj_local = w2e(sdc_traj_world)  # (91, 2)
    #     sdc_traj_local = torch.tensor(sdc_traj_local, dtype=torch.float32)
    #
    #     # === 邻居轨迹 ===
    #     neighbor_trajs_local = []
    #     for traj in result["neighbor_trajs"].values():
    #         traj = np.array(traj, dtype=np.float32)
    #         traj_local = w2e(traj)
    #         neighbor_trajs_local.append(torch.tensor(traj_local, dtype=torch.float32))
    #
    #     if len(neighbor_trajs_local) == 0:
    #         neighbor_trajs = torch.zeros((0, 91, 2))  # no neighbor fallback
    #     else:
    #         neighbor_trajs = pad_sequence(
    #             neighbor_trajs_local, batch_first=True
    #         )  # (N, 91, 2)
    #
    #     sample = {
    #         "sdc_traj": sdc_traj_local,  # (91, 2)
    #         "neighbor_trajs": neighbor_trajs,  # (N, 91, 2)
    #         "rel_path": rel_path,
    #     }
    #
    #     logging.debug(
    #         f"[Dataset] ✅ sdc: {sdc_traj_local.shape}, neighbors: {neighbor_trajs.shape}"
    #     )
    #     return self.transform(sample) if self.transform else sample
