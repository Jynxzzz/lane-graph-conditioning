import logging
import os

from jynxzzzdebug import debug_break


def print_scene_structure(scene, frame_idx=0):
    # logging.info("=== Debugging scene structure ===")
    # logging.info(f"scene.keys: {scene.keys()}")
    # logging.info(f"objects: {len(scene['objects'])} agents")
    # logging.info(f"av_idx: {scene['av_idx']}")
    #
    # ego = scene["objects"][scene["av_idx"]]
    # logging.info(f"ego keys: {list(ego.keys())}")
    # logging.info(f"ego position[0]: {ego['position'][0]}")
    # logging.info(f"ego heading[0]: {ego['heading'][0]}")
    #
    # objects = scene["objects"]
    # # logging obj type
    # logging.info(f"objects[0] type: {(objects[0]['type'])}")
    # # logging velocity
    # logging.info(f"objects[0] velocity: {objects[0]['velocity']}")
    # traffic lights
    traffic_lights = scene.get("traffic_lights", [])
    logging.info(f"traffic_lights: {traffic_lights}")
    lights = set()
    light_stop_points = set()
    for frame in traffic_lights:
        logging.info(f"frame has {len(frame)} traffic lights")
        for light in frame:
            if isinstance(frame, list) and len(frame) > 0:
                logging.info(f"frame has {len(frame)} traffic lights")
                stop_point = light.get("stop_point", {})
                lights.add(light["lane"])
                stop_xy = (stop_point.get("x"), stop_point.get("y"))
                light_stop_points.add(stop_xy)
                logging.info(f"unique traffic lights: {len(lights)}")
            else:
                logging.warning("traffic_lights frame is not a list or is empty")


def explore_scene(scene, frame_idx=0, max_depth=2, prefix=""):
    """
    递归探索 scene 的结构，打印每层 key 和 value 的基本信息。
    """
    if max_depth <= 0 or not isinstance(scene, dict):
        return
    for k, v in scene.items():
        if isinstance(v, dict):
            logging.info(f"{prefix}{k}: dict with {len(v)} keys")
            explore_scene(v, frame_idx, max_depth - 1, prefix + "  ")
        elif isinstance(v, list):
            logging.info(
                f"{prefix}{k}: list of len={len(v)} type={type(v[0]) if v else 'empty'}"
            )
            if v and isinstance(v[0], dict):
                logging.info(f"{prefix}  {k}[0] keys: {list(v[0].keys())}")
        else:
            logging.info(f"{prefix}{k}: {type(v).__name__}")
