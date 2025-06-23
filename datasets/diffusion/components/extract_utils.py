import json
import logging


def extract_sdc_and_neighbors(scenario, max_distance=30.0, frame_idx=0):
    sdc_id = scenario["av_idx"]
    sdc = scenario["objects"][sdc_id]

    sdc_traj = [
        [float(p["x"]), float(p["y"])]
        for p, valid in zip(sdc["position"], sdc["valid"])
        if valid
    ]

    sdc_pos = sdc["position"][frame_idx]
    neighbors = []

    for i, obj in enumerate(scenario["objects"]):
        if i == sdc_id:
            continue
        if not obj["valid"][frame_idx]:
            continue

        obj_pos = obj["position"][frame_idx]
        dx = obj_pos["x"] - sdc_pos["x"]
        dy = obj_pos["y"] - sdc_pos["y"]
        dist = (dx**2 + dy**2) ** 0.5

        if dist < max_distance:
            neighbors.append(i)

    neighbor_trajs = {}
    for nid in neighbors:
        obj = scenario["objects"][nid]
        traj = [
            [float(p["x"]), float(p["y"])]
            for p, valid in zip(obj["position"], obj["valid"])
            if valid
        ]
        neighbor_trajs[nid] = traj

    return {
        "sdc_id": sdc_id,
        "sdc_traj": sdc_traj,
        "neighbor_ids": neighbors,
        "neighbor_trajs": neighbor_trajs,
    }
