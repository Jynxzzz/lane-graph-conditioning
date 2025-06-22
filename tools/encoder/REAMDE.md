# 默认 simple_token 编码器
python main.py

# 切换到 latent_path_token 编码器
python main.py encoder=latent_path_token

# 开启 lane+vehicle 邻居提取，调整半径
python main.py encoder=neighbor_token extractor.sdc_radius=30


idea:必须实现的， 但还没做
我理解你现在这个状态：你在一口气看到两个未来方向的关键点——

⸻

🚗 问题 A：“轨迹 → goal → 最短路 → Lane ID 映射”

主线任务，支撑结构约束和 trajectory 监督。

	•	🔎 目标：从 ground truth 轨迹推断出 goal lane。
	•	🧩 意义：补全训练目标、监督 loss、规划约束。
	•	✅ 一旦闭环，就能从每一帧中找出是否偏离最优路径。

⸻

🧠 问题 B：“以虫为镜”构建 [[局部 lane 图结构]]

输入设计任务，提升 token 表达能力。

	•	🔎 目标：像虫语一样，把 lane token 编码为“局部结构张量”。
	•	🧩 意义：让模型具备感知上下左右、转弯、红灯等语义。
	•	✅ 一旦编码好，模型的输入会结构化、鲁棒性提升。

⸻

🧭 建议的解决顺序：

优先级	路线	理由
✅第一步	问题 A：轨迹 → goal → lane_id	明确 ground truth 和结构约束来源，闭环“我往哪走”这个核心
✅第二步	问题 B：局部结构输入设计	之后可以基于 goal 路径增强输入 token 的结构性和方向性


⸻

🧠 更重要的是这句话：

“有了 Goal 和最短路，我们才能知道，哪一帧模型是不是跑歪了。”

这句话是全局监督的锚点，特别适用于：
	•	policy training
	•	imitation learning
	•	reward shaping

⸻

📌 你现在可以这样分工自己的任务：

模块	子任务	是否记得住？
Goal & Traj	logging → 解析轨迹 → 找最后一帧 → 找最近的 lane → 最短路 → lane_path	✅ 马上能做
Lane Grid	设计 build_local_lane_tensor() → 定义上下左右关系 → 构造结构化张量	✅ 之后专门搞

我建议现在先完成“Goal + 最短路径 + lane_id 映射”，这是闭环的根。

接下来你可以继续在当前上下文里操作，我会记住你刚才的 [[局部结构化 lane 编码]] 想法，等你完成主线后，我们就直接回头动手设计它。

需要我现在帮你写一个 goal → lane_path → logging 的函数骨架吗？


对！！你这个问题问得非常关键，说明你已经把“虫语地图”与“lane graph 编码”这两个世界联通了。我们来逐步对齐一下你说的两个系统的本质区别与相通点：

⸻

🐛 虫子世界的 Grid Encoding 回顾：

在虫子游戏中我们送给模型的输入是：

一个以 虫子为中心的 5×5 grid，每个格子是一个 离散 token，可以是“黑墙”“空地”“目标点”等。

	•	它天然是以虫为中心的、局部视角。
	•	是一个 绝对结构编码（grid 固定，token 会动）。
	•	输入是整个邻居图谱，而非单个 token。

⸻

🚗 Waymo Lane Encoding 的现实挑战：

目前我们给到模型的是：

一个个 [[LaneToken]]，每个是一个 lane + 属性打包的 token。我们目前依赖的是一个 水流图 G 来组织它们。

⚠️问题在于：
	1.	没有明确编码出“当前位置的局部图结构”，比如我左边、右边是什么 lane。
	2.	heading 与邻接 lane 的相对方向信息缺失或过于稀疏。
	3.	模型面对的是一个 flat list of tokens，而不是局部结构张量（如 grid）。

⸻

✅ 如果想向虫子世界靠拢，我们需要：

✅ 1. 给出以 ego 为中心的局部图（当前 + 前/后/左右）

可以把每一帧的 token 编成一个结构张量：

		
left2	front-left	right2
left	ego	right
back	back2	

你可以只选取附近的 N 个 lane 构建一个 3x3 局部结构图，或者更灵活地直接构造一个 neighbor_dict。

⸻

✅ 2. 每个位置上的 token 要包含方向（heading）、交通信号等语义信息

就像虫语里，“前面是目标点”可以让模型预测向前跳，这里你需要：
	•	ego lane 的 heading
	•	左右/前后 lane 的 heading 差值（Δθ）
	•	是否有 traffic light / stop sign
	•	是否是目标路径
	•	lane embedding

⸻

✅ 3. optional：把这些 token 转成“grid-like tensor”

可以用类似虫语输入的方式：

lane_tensor = np.zeros((3, 3, D))  # D 是每个 token 的向量维度
lane_tensor[1, 1] = encode_lane(ego_lane)
lane_tensor[1, 0] = encode_lane(back_lane)
lane_tensor[1, 2] = encode_lane(front_lane)
lane_tensor[0, 1] = encode_lane(left_lane)
lane_tensor[2, 1] = encode_lane(right_lane)
# 其他格子为空或 padding


⸻

🧠 总结

✅ 虫子系统强调的是 “结构 + 局部 + token grid”，
✅ 当前 lane token 系统是 “全局 token list + weak graph”，
🧩 所以：你说得没错，我们 没有明确构造“sdc视角下的局部结构输入”！

你可以设计一个新的 build_local_lane_tensor() 方法，模仿虫子 grid，把 lane 编码成局部结构，这将是模型输入的超级加分项。

你愿意试试我们一起设计这个函数吗？🧩



当然哥，我们今天达成的这个框架是你整个项目的基石，我这就给你更新一个稳固、清晰、结构化的 anchor，标志我们从“token 零散编码”进入到“统一轨迹 × 地图 × 信号语义对齐建模”的新阶段。

⸻

🧷 ANCHOR: 场景结构感知轨迹语言建模系统（ScenarioDreamer v1.0）

🎯 核心目标：

构建一个结构感知的轨迹语言模型系统，将交通场景中的三类核心元素 —— trajectory、lane、traffic_light —— 统一编码为 token，送入模型进行对齐学习与预测。模型具备对 token 序列与结构 memory 同步感知的能力，实现：
	•	多车轨迹建模（trajectory token）
	•	路网结构理解（lane token + graph memory）
	•	交通信号引导（traffic light token）

⸻

📦 模块结构设计：

┌────────────────────────────┐
│        Scenario Input       │
│ objects, lane_graph, lights│
└────────────────────────────┘
             │
             ▼
┌────────────────────────────┐
│     ✨ SimpleEncoder         │
│ ┌────────────────────────┐ │
│ │encode_lanes(scene)     │ │ → lane_tokens: [LaneToken]
│ │encode_traffic_lights() │ │ → traffic_tokens: [TLToken]
│ │encode_agents()         │ │ → traj_tokens: [int]
│ └────────────────────────┘ │
└────────────────────────────┘
             │
             ▼
    ◉ lane_token_map（每帧场景唯一）
    ◉ lane_memory_graph（结构图：succ, pred）
    ◉ traj_token + traj_lane_align（可选）

---

### 🧠 模型输入设计

- `input_tokens`: `traj_tokens`（主序列，行为表达）
- `memory_bank`: `lane_tokens + traffic_tokens` → embedding 后形成结构 memory
- `attention`: decoder / transformer 可访问 memory，支持结构引导

---

### 📌 Embedding 对齐机制（统一 vocab 空间）

| 类型             | 编码方式         | 目标                     |
|------------------|------------------|--------------------------|
| `traj_token`     | `(dx, dy)` 离散  | 表达运动行为             |
| `lane_token`     | heading, stop, id嵌入 | 提供结构上下文           |
| `light_token`    | 信号状态,位置     | 表达环境控制条件         |
| `token_embed()`  | 多类型统一映射   | 送入 transformer 训练使用 |

---

### 🏗 当前状态 Checkpoint：

| 模块名                     | 状态     |
|----------------------------|----------|
| `extract_sdc_and_neighbors()` | ✅ 已完成 |
| `encode_traj_to_tokens()`     | ✅ 初版完成，支持 dx/dy 离散 |
| `encode_lanes()`              | ✅ 已完成，含 lane graph 结构 |
| `encode_traffic_lights()`     | ✅ 可调通 |
| `lane_token_map` + `lane_token_graph` | ✅ 构建中 |
| `token/embedding 对齐系统`    | 🔜 正在设计，支持结构化表示 |

---

### 🧱 下一步建议：

1. ✅ 实现 `LaneMemoryBuilder(scene)`，构造结构图用于 attention。
2. ✅ 实现 `TrajectoryToLaneAligner()`：给每个 traj_token 附带对应的 lane_token_id。
3. 🔜 封装 `token_embed()`：将三类 token 映射入统一嵌入空间。
4. 🔜 搭建 Minimal Transformer Decoder，支持 attention over structure。
5. 🔜 数据批处理器（collator）：将 traj_seq + structure_graph 打包送入模型。

---

哥，咱们现在不只是做轨迹预测，而是亲手搭建一个“交通轨迹语言模型 + 图结构感知记忆”的融合系统。如果以后你要写论文/产品/demo，这将是你的“**架构之锚**”。

你说一句，我就立马开始写 `LaneMemoryBuilder` + `Aligner` 代码块！我们今天稳稳地确立了主干。
