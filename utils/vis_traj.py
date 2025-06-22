from jynxzzzdebug import debug_break, debug_print, explore_dict, setup_logger

logging = setup_logger("traj_processing", "logs/traj_processing.log")

from _dev.encoder_debug import encode_lanes_debug
from _dev.render_frame import (
    build_local_transform,
    draw_agents,
    draw_ego_box,
    draw_heading_vector,
    draw_lane_tokens,
    draw_outer_circle,
    draw_traffic_light_tokens,
    draw_traffic_lights,
    extract_ego_info,
    init_canvas,
)
from matplotlib.lines import Line2D
from tools.debug_scene_structure import explore_scene, print_scene_structure
from tools.scene_loader import (
    load_random_scene_from_list,
    load_scene_data,
    load_selected_scene_list,
)

from utils.traj_processing import extract_sdc_and_neighbors

# 文件路径：utils/vis_traj.py
# 文件位置建议：utils/vis_traj.py


# def draw_trajectory_with_interval_dots(
#     ax,
#     traj_xy,
#     w2e,
#     color="blue",
#     base_alpha=0.4,
#     dot_interval=10,
#     label=None,
#     highlight_color="orange",
#     zorder=2,
#     linewidth=1.5,
#     scatter_size=20,
# ):
#     """
#     绘制轨迹曲线，并在每隔 dot_interval 处用不同颜色打点。
#
#     参数：
#         traj_xy: (N, 2) array，世界坐标轨迹
#         w2e: 世界 → ego frame 的转换函数
#         color: 主轨迹颜色
#         highlight_color: 每隔 interval 个点的高亮色
#         base_alpha: 默认圆点透明度
#     """
#     import numpy as np
#
#     traj_xy = np.array(traj_xy)
#     if traj_xy.ndim != 2 or traj_xy.shape[1] < 2:
#         logging.warning(
#             f"[draw_trajectory_with_interval_dots] 异常 shape: {traj_xy.shape}"
#         )
#         return
#
#     # === 转换坐标系 ===
#     xy_local = w2e(traj_xy[:, :2])
#
#     # === 连线 ===
#     ax.plot(
#         xy_local[:, 0],
#         xy_local[:, 1],
#         linestyle="--",
#         color=color,
#         linewidth=linewidth,
#         alpha=base_alpha,
#         label=label,
#         zorder=zorder,
#     )
#
#     # === 糖葫芦点 ===
#     ax.scatter(
#         xy_local[:, 0],
#         xy_local[:, 1],
#         c=color,
#         s=scatter_size,
#         edgecolors="gray",
#         linewidths=0.5,
#         alpha=base_alpha,
#         zorder=zorder,
#     )
#
#     # === 每隔 interval 的点高亮显示 ===
#     highlight_xy = xy_local[::dot_interval]
#     ax.scatter(
#         highlight_xy[:, 0],
#         highlight_xy[:, 1],
#         c=highlight_color,
#         s=scatter_size * 1.2,
#         edgecolors="black",
#         linewidths=0.6,
#         alpha=0.9,
#         zorder=zorder + 1,
#     )


# custom_legend =
# custom_legend = [
#     Line2D(
#         [0],
#         [0],
#         marker="o",
#         color="w",
#         label="SDC",
#         markerfacecolor="pink",
#         markersize=8,
#         markeredgecolor="black",
#     ),
#     Line2D(
#         [0],
#         [0],
#         marker="o",
#         color="w",
#         label="ConVs",
#         markerfacecolor="gray",
#         markersize=6,
#         markeredgecolor="black",
#         alpha=0.6,
#     ),
#     Line2D(
#         [0],
#         [0],
#         marker="o",
#         color="w",
#         label="SDC 1s Interval Marker (10Hz)",
#         markerfacecolor="gold",
#         markeredgecolor="black",
#         markersize=8,
#     ),
#     Line2D(
#         [0],
#         [0],
#         marker="o",
#         color="w",
#         label="ConV 1s Interval Marker (10Hz)",
#         markerfacecolor="gold",
#         markeredgecolor="black",
#         markersize=8,
#     ),
# ]


custom_legend = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="SDC Traj (10Hz)",  # 0.1 sec间隔
        markerfacecolor="pink",
        markeredgecolor="black",
        markersize=6,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="ConV Trajs (10Hz)",  # 0.1 sec间隔
        markerfacecolor="gray",
        markeredgecolor="black",
        markersize=6,
        alpha=0.6,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="SDC traj (1Hz)",  # 每 1 秒一个点
        markerfacecolor="gold",
        markeredgecolor="black",
        markersize=7,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="gray",  # 注意这行的 color 会作为线的颜色显示出来
        label="ConV traj (1Hz)",  # 每 1 秒一个点
        markerfacecolor="skyblue",
        markeredgecolor="gray",
        linestyle="--",
        linewidth=1.0,
        markersize=5,
        alpha=0.5,
    ),
]


def draw_trajectories(ax, scene, frame_idx, w2e):
    import numpy as np

    from utils.traj_processing import extract_sdc_and_neighbors

    traj_info = extract_sdc_and_neighbors(scene, frame_idx=frame_idx)
    sdc_traj = traj_info["sdc_traj"]
    neighbor_trajs = traj_info["neighbor_trajs"]

    # ==== SDC 轨迹 ====
    xy = np.array(sdc_traj)[:, :2]
    xy_local = w2e(xy)

    # ax.plot(xy_local[:, 0], xy_local[:, 1], "r-", linewidth=2, label="SDC")

    draw_trajectory_with_interval_dots(
        ax,
        traj_xy=sdc_traj,
        w2e=w2e,
        color="red",
        highlight_color="gold",
        dot_interval=10,
        label="SDC",
        scatter_size=30,
        zorder=3,
    )

    ax.scatter(
        xy_local[:, 0],
        xy_local[:, 1],
        c="pink",
        s=15,
        edgecolors="black",
        linewidths=0.6,
        alpha=0.9,
        zorder=3,
    )

    # ==== 邻居轨迹 ====
    logging.info(f"[DEBUG] 有 {len(neighbor_trajs)} 个邻车轨迹")
    for nid, traj in neighbor_trajs.items():
        logging.info(f"[{nid}] 轨迹点数: {len(traj)}")
        if not isinstance(traj, (list, np.ndarray)):
            logging.warning(f"[WARN] 非法邻居轨迹类型: {type(traj)} → 已跳过")
            continue

        xy = np.array(traj)
        if xy.ndim != 2 or xy.shape[1] < 2:
            logging.warning(f"[WARN] 异常邻居轨迹 shape={xy.shape} → 已跳过")
            continue

        # xy_local = w2e(xy[:, :2])
        # ax.plot(xy_local[:, 0], xy_local[:, 1], "b--", alpha=0.5)
        # ax.scatter(
        #     xy_local[:, 0],
        #     xy_local[:, 1],
        #     c="gray",
        #     s=20,
        #     edgecolors="gray",
        #     linewidths=0.4,
        #     alpha=0.4,
        #     zorder=2,
        # )
        # 自定义图例
        draw_trajectory_with_interval_dots(
            ax,
            traj_xy=traj,
            w2e=w2e,
            color="gray",
            base_alpha=0.4,
            dot_interval=10,
            label=None,
            highlight_color="skyblue",
            zorder=2,
            linewidth=1.0,
            scatter_size=16,
        )


# def draw_trajectories(ax, scene, frame_idx, w2e):
#     import numpy as np
#
#     from utils.traj_processing import extract_sdc_and_neighbors
#
#     traj_info = extract_sdc_and_neighbors(scene, frame_idx=frame_idx)
#     sdc_traj = traj_info["sdc_traj"]
#     neighbor_trajs = traj_info["neighbor_trajs"]
#
#     # 画 SDC 轨迹（红色实线）
#     xy = np.array(sdc_traj)[:, :2]
#     xy_local = w2e(xy)
#     ax.plot(xy_local[:, 0], xy_local[:, 1], "r-", linewidth=2, label="SDC")
#     for traj in neighbor_trajs:
#         if not isinstance(traj, (list, np.ndarray)):
#             logging.warning(f"[WARN] 非法邻居轨迹类型: {type(traj)} → 已跳过")
#             continue
#
#         xy = np.array(traj)
#         if xy.ndim != 2 or xy.shape[1] < 2:
#             logging.warning(f"[WARN] 异常邻居轨迹 shape={xy.shape} → 已跳过")
#             continue
#
#         xy_local = w2e(xy[:, :2])
#         ax.plot(xy_local[:, 0], xy_local[:, 1], "b--", alpha=0.5)
#
#     ax.legend()


def draw_trajectory_with_interval_dots(
    ax,
    traj_xy,
    w2e,
    color="blue",
    base_alpha=0.4,
    dot_interval=10,
    label=None,
    highlight_color="orange",
    zorder=2,
    linewidth=1.5,
    scatter_size=20,
):
    """
    绘制轨迹曲线，并在每隔 dot_interval 处用不同颜色打点。

    参数：
        traj_xy: (N, 2) array，世界坐标轨迹
        w2e: 世界 → ego frame 的转换函数
        color: 主轨迹颜色
        highlight_color: 每隔 interval 个点的高亮色
        base_alpha: 默认圆点透明度
    """
    import numpy as np

    traj_xy = np.array(traj_xy)
    if traj_xy.ndim != 2 or traj_xy.shape[1] < 2:
        logging.warning(
            f"[draw_trajectory_with_interval_dots] 异常 shape: {traj_xy.shape}"
        )
        return

    # === 转换坐标系 ===
    xy_local = w2e(traj_xy[:, :2])

    # === 连线 ===
    ax.plot(
        xy_local[:, 0],
        xy_local[:, 1],
        linestyle="--",
        color=color,
        linewidth=linewidth,
        alpha=base_alpha,
        label=label,
        zorder=zorder,
    )

    # === 糖葫芦点 ===
    ax.scatter(
        xy_local[:, 0],
        xy_local[:, 1],
        c=color,
        s=scatter_size,
        edgecolors="gray",
        linewidths=0.5,
        alpha=base_alpha,
        zorder=zorder,
    )

    # === 每隔 interval 的点高亮显示 ===
    highlight_xy = xy_local[::dot_interval]
    ax.scatter(
        highlight_xy[:, 0],
        highlight_xy[:, 1],
        c=highlight_color,
        s=scatter_size * 1.2,
        edgecolors="black",
        linewidths=0.6,
        alpha=0.9,
        zorder=zorder + 1,
    )
    from matplotlib.lines import Line2D

    ax.legend(handles=custom_legend, loc="upper right")
