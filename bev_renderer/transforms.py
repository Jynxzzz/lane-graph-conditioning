# transforms.py
import math

import numpy as np


def world_to_ego(points, ego_pos, ego_heading_deg):
    theta = -math.radians(ego_heading_deg)
    translation = np.array(ego_pos)
    rot_matrix = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    return (points - translation) @ rot_matrix.T
