# scenario_dreamer/encoders/base_encoder.py

from abc import ABC, abstractmethod


class BaseEncoder(ABC):
    @abstractmethod
    def encode_lanes(self, scene):
        """提取 lane 图结构 → token 列表"""
        pass

    @abstractmethod
    def encode_traffic_lights(self, scene, frame_idx):
        """提取 traffic light 信息 → token 列表"""
        pass

    @abstractmethod
    def encode_agents(self, scene, frame_idx):
        """提取其他车辆轨迹信息 → token 列表"""
        pass
