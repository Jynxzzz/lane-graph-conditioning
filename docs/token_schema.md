当然哥马上帮你整理！以下是你四个 Token 结构的完整 Markdown 文档版本，适用于写论文附录、项目文档、或者结构注释。

⸻

Token Schema 设计文档

本文档定义了用于轨迹预测与图结构建模中的四种核心 Token 类型：SDC、Agent、Lane、Traffic Light。每个 Token 都具备结构化字段、语义注释与坐标定义，便于后续训练与论文描述。

⸻

📦 1. TrafficLightToken

用于表示交通信号灯的语义 token，具备空间位置、当前状态与控制信息。

字段名	类型	含义
id	int	Token 编号
frame_idx	int	出现帧编号
x, y	float	灯在 ego 车坐标系下的平面位置
state	int	灯色：0=红，1=黄，2=绿
controlled_lane	Optional[int]	控制的车道 token ID（可选）
dx, dy	float	用于绘图偏移方向的单位向量，默认向上


⸻

🚗 2. LaneToken

表示地图中一个 Lane Segment 的语义结构，用于图建模、路径规划与拓扑连接建图。

字段名	类型	含义
id	int	Token 编号
lane_id	str	原始地图中的 lane 标识
centerline	np.ndarray (N,2)	中心线坐标点集合
heading_token	int	离散化后的朝向 token（用于方向预测）

🧩 拓扑结构字段

字段名	类型	含义
pred_id, succ_id	List[str]	前驱 / 后继 lane_id
left_id, right_id	Optional[str]	相邻左右车道 ID

🏷️ 语义标签字段

字段名	类型	含义
is_start	bool	是否为起点 lane
is_goal	bool	是否为目标 lane
on_GT_path	bool	是否在 GT 路径上
has_traffic_light	bool	是否受控于交通灯
has_stop_sign	bool	是否有停止标志

🧭 坐标辅助字段

字段名	类型	含义
ego_xy	Optional[np.ndarray]	在 ego 坐标系下的参考点
w2e	Optional[np.ndarray]	世界坐标 → ego 坐标的变换矩阵


⸻

🦅 3. SDCToken

用于描述自车（Self Driving Car）在当前帧的状态信息。

字段名	类型	含义
id	int	Token 编号
x, y	float	自车在场景中的位置（通常为局部坐标）
heading	float	朝向角（单位：弧度）
speed	float	当前速度（单位：m/s）
accel	float	当前加速度（单位：m/s²）


⸻

👥 4. AgentToken

表示场景中其他车辆或动态对象的状态信息。

字段名	类型	含义
id	int	Token 编号
x, y	float	位置坐标
vx, vy	float	当前速度向量（x/y 方向，单位：m/s）
relative_to_sdc	bool	是否为相对于 SDC 的局部坐标系（True 为默认）


⸻

🧠 总结说明
	•	所有 Token 均可投射至统一坐标系（如 ego frame），支持时空建模。
	•	LaneToken 和 TrafficLightToken 可用于构建图结构；AgentToken 和 SDCToken 用于轨迹建模。
	•	可选字段如 w2e 和 relative_to_sdc 可灵活控制输入格式，便于训练统一性。
	•	可视化时建议使用 dx/dy 控制文字偏移，避免 label 重叠。

⸻

如需我同时生成 .py 代码文件或 .md 文件，请告诉我！也可以一键打包给你下载使用。
