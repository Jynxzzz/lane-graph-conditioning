from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class TrafficLightToken:
    id: int
    frame_idx: int
    x: float
    y: float
    state: int
    controlled_lane: Optional[int] = None

    # ✅ 新增字段：方向向量（用于绘图偏移 label）
    dx: float = 0.0  # x 方向分量
    dy: float = 1.0  # y 方向分量


@dataclass
class LaneToken:
    id: int  # token 编号
    lane_id: str  # 原始 lane 编号
    centerline: np.ndarray  # (N, 2)
    heading_token: int  # 离散方向 token
    pred_id: List[str] = field(default_factory=list)  # 前驱 lane_id 列表
    succ_id: List[str] = field(default_factory=list)  # 后继 lane_id 列表
    left_id: Optional[str] = None  # 左邻
    right_id: Optional[str] = None  # 右邻
    is_start: bool = False  # 是否是 ego 起点
    is_goal: bool = False  # 是否是目标 lane（未来加入）
    on_GT_path: bool = False  # 是否在 GT 路径上（未来加入）
    has_traffic_light: bool = False
    has_stop_sign: bool = False
    ego_xy: Optional[np.ndarray] = None  # ego 位置
    w2e: Optional[np.ndarray] = None  # 世界转 ego 坐标系变换
