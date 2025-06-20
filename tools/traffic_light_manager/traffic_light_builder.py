import os

import networkx as nx


def extract_light_info(scene):
    light_info = {}
    for light_group in scene.get("traffic_lights", []):
        for light in light_group:
            lane_id = light["lane"]
            state = light["state"]  # 6=绿灯，3=红灯（你可加映射表）
            light_info[lane_id] = state
    return light_info


def annotate_light_to_graph(G, light_info):
    for lane_id in G.nodes:
        if lane_id in light_info:
            G.nodes[lane_id]["light_state"] = light_info[lane_id]
        else:
            G.nodes[lane_id]["light_state"] = -1  # 无灯


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
