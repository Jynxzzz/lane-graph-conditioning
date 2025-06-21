from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class TrafficLightToken:
    id: int
    frame_idx: int  # 出现在哪一帧（用于时序建模）
    x: float  # 当前 token 在车辆坐标系下的位置
    y: float
    state: int  # 0=red, 1=yellow, 2=green
    controlled_lane: Optional[int] = None  # 控制的 lane_id（可用于图结构）

    # ✅ 可选方向向量（label 渲染偏移方向）
    dx: float = 0.0
    dy: float = 1.0


@dataclass
class LaneToken:
    id: int  # token 编号（全局唯一）
    lane_id: str  # 原始地图中的 lane ID
    centerline: np.ndarray  # 中心线坐标 (N, 2)
    heading_token: int  # 离散化方向 token 编号（用于方向预测）

    # 拓扑结构
    pred_id: List[str] = field(default_factory=list)
    succ_id: List[str] = field(default_factory=list)
    left_id: Optional[str] = None
    right_id: Optional[str] = None

    # 语义标签
    is_start: bool = False
    is_goal: bool = False
    on_GT_path: bool = False
    has_traffic_light: bool = False
    has_stop_sign: bool = False

    # 坐标系辅助（如绘图或 frame 对齐）
    ego_xy: Optional[np.ndarray] = None  # 在 ego frame 下的起始点位置
    w2e: Optional[np.ndarray] = None  # 世界 → ego frame 的坐标变换矩阵


@dataclass
class SDCToken:
    id: int  # token ID
    x: float
    y: float
    heading: float  # 朝向（单位：弧度）
    speed: float  # m/s
    accel: float  # m/s²


@dataclass
class AgentToken:
    id: int  # token 编号
    x: float
    y: float
    vx: float  # x方向速度（单位 m/s）
    vy: float  # y方向速度
    relative_to_sdc: bool = True  # 是否为相对 ego 车的局部坐标系
