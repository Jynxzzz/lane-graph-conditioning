import numpy as np
import torch


def world_to_ego(points, ego_pos, heading_deg):
    """
    批量处理世界坐标点 → Ego 坐标（以 ego_pos 为原点，heading_deg 为 x 正方向）
    points: (N, 2) numpy array or torch tensor
    """
    if isinstance(points, torch.Tensor):
        points = points.numpy()

    # 平移
    shifted = points - np.array(ego_pos)

    # 旋转
    theta = -np.radians(heading_deg)  # 注意是负角度（世界→ego）
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

    local = shifted @ rot.T  # shape (N, 2)
    return torch.tensor(local, dtype=torch.float32)
