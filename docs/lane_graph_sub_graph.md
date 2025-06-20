Lane Graph 构建与水流式子图筛选逻辑

核心目标

构建一张以 SDC 车辆所在车道为起点的子图，模拟交通规则下信息流动的拓扑结构，以供下游 GNN 模型学习 lane-level 的路径结构、轨迹预测或行为决策。

⸻

1️⃣ SDC 车道 ID 确定算法

输入
	•	sdc_xy: SDC 当前帧的 (x, y) 坐标点
	•	lane_graph['lanes']: 每个 lane_id → 含有 xyz_polyline 的 dict

步骤
	1.	遍历所有 lane_id：提取该车道的中心线点序列
	2.	计算 SDC 点到该中心线的最近距离（使用点到折线的最短距离）
	3.	选择最近的 lane_id 作为 ego_lane_id

输出
	•	ego_lane_id: 当前帧 SDC 所在的车道编号

⸻

2️⃣ 水流式子图构造策略

图结构
	•	使用 networkx.DiGraph 构建有向图 G
	•	节点为 lane_id
	•	边为 lane 拓扑对（含方向信息）

子图构建逻辑：
	1.	反推上游：
	•	找到所有 (from_id → ego_lane_id) 的前驱关系：即 suc_pairs[from_id] 中包含 ego_lane_id
	•	仅保留这些 from_id → ego_lane_id 的边，标注为 pre_to_sdc
	•	不引入这些上游车道的左/右/suc，避免图结构膨胀或混淆
	2.	正向展开：
	•	从 ego_lane_id 开始：
	•	添加 ego → suc，边类型 sdc_to_suc
	•	添加 ego → left/right，边类型 sdc_to_left / sdc_to_right
	•	可选：再对 suc 做 1-2 层扩展（控制 max_hops）
	3.	最终子图结构特点：
	•	类似水流从 ego 向下游扩展
	•	只包含明确方向的边
	•	避免“回头”或“并列车道跳转”

⸻

3️⃣ 节点与边的特征设计

节点（lane_id）
	•	坐标: 中心线多段折线 polyline
	•	是否为 SDC 所在车道: binary
	•	lane type（可选）

边（from → to）
	•	类型编码: 0=pre_to_sdc, 1=sdc_to_suc, 2=sdc_to_left, 3=sdc_to_right
	•	是否为主路径（如 sdc → suc）

⸻

4️⃣ 模型训练接口准备

未来 GNN 输入格式可采用：
	•	edge_index: 图边结构
	•	x: 节点特征张量
	•	edge_attr: 边的类型
	•	y: 可为目标轨迹、lane 分类、下一跳预测等

⸻

附注：可视化与验证
	•	我们使用 spring layout 可视化子图结构（清晰表达拓扑）
	•	可在图中标注边类型、方向与节点类别，用于 debug
	•	可扩展为动态图展示“水流推进”过程（逐帧冒泡）

⸻

总结

本构图方案遵循车辆在 lane 网络中的行为逻辑，以“前驱 → ego → 下游”构建方向图，同时排除与任务无关的横向分支与反向干扰。为后续 GNN 模型提供清晰、结构化的 lane-level 输入。
