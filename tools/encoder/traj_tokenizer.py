from jynxzzzdebug import debug_break, debug_print, explore_dict, setup_logger

logging = setup_logger("traj_processing", "logs/traj_processing.log")


def encode_traj_to_tokens(sdc_traj, neighbors, grid_size=0.5, grid_range=100.0):
    """
    将 SDC 和邻居车辆轨迹转为 token 序列。

    - 每个点 (x, y) 会被量化为一个整数 token。
    - 可用于喂给 Transformer 或 AutoEncoder 等模型。

    Args:
        sdc_traj: List of [x, y]
        neighbors: Dict[int, List[[x, y]]]
        grid_size: 每个 token 所代表的物理距离（单位米）
        grid_range: 假设的最大坐标范围，用于构建 token id 范围

    Returns:
        List[int]: 编码后的 token 列表
    """
    tokens = []

    def quantize(x, y):
        # 将坐标离散化成整数格点
        ix = int((x + grid_range / 2) / grid_size)
        iy = int((y + grid_range / 2) / grid_size)
        return iy * int(grid_range / grid_size) + ix  # 合并为单一 token

    # 编码 SDC 轨迹
    for pt in sdc_traj:
        token = quantize(pt[0], pt[1])
        tokens.append(token)

    # 编码邻居车辆轨迹
    for nid, traj in neighbors.items():
        for pt in traj:
            token = quantize(pt[0], pt[1])
            tokens.append(token)

    logging.info(
        f"Encoded {len(sdc_traj)} SDC points and {len(neighbors)} neighbors into {len(tokens)} tokens."
    )
    return tokens
