下面是为你论文准备的文档说明段，专门解释图中 lane ID / token ID / traffic light ID 的绘图偏移策略，可直接作为论文中图注或方法部分的一段说明使用：

⸻

📘 可视化渲染偏移策略说明（Token Annotation Offsets）

为了提升 BEV 图像的可读性与信息层次清晰度，我们在绘制各类 Token 与 Lane ID 时，采用了手动错位的标签显示策略。具体如下：

✅ Lane Token 显示规则（draw_lane_tokens()）：

对于每条车道的起点位置 lane_pts[0]：
	•	上方 -9 单位：显示 Lane ID（黑色文本）
	•	下方 -12 单位：显示该 Lane 所绑定的 Token ID（蓝色文本）
	•	这两个信息分别来自：

ax.text(x, y - 9, str(lane_id), ...)
ax.text(x, y - 12, f"t{token.id}", ...)



⚠️ 本策略确保 同一路径起点不会被文字重叠遮挡，尤其适用于交叉口多 lane 并行的情形。

⸻

✅ Traffic Light Token 显示规则（draw_traffic_light_tokens()）：

对于每个 traffic light token 的位置：
	•	上方 -3 单位：显示 TL 名称（例如 TL0, TL1，带白色边框）
	•	下方 -6 单位：显示 Token ID（绿色，如 t0, t1）
	•	对应绘图逻辑如下：

ax.text(x, y - 3, f"TL{tl_idx}", bbox=..., ...)
ax.text(x, y - 6, f"t{token.id}", color="green", ...)



⸻

🎯 目的与优势：
	•	通过将同一位置上的不同语义信息进行垂直偏移展示，避免重叠；
	•	方便读者一眼识别：
	•	某条车道由哪个灯控制；
	•	每个灯的语义编号与其控制的 lane 的对应关系；
	•	Token ID 与原始元素（lane/light）的绑定逻辑。

⸻

📝 后续补充建议：

如有需要，可以在图中进一步添加：
	•	→ 箭头显示灯的 (dx, dy) 朝向；
	•	高亮被某灯控制的 lane；
	•	可点击或 hover 显示全字段 JSON 信息（用于交互式论文 demo）。

⸻

如果你想要我写一个完整的 obsicat_bev_visualization.md 文档，带截图 + 注解 + 对应代码片段模块，我可以一步步帮你组织。是否现在就开始？😼


# BEV Token Annotation Offset Strategy

## Overview

In our Bird’s Eye View (BEV) visualization system, multiple semantic elements such as lane lines, traffic lights, and associated token IDs are rendered on a 2D plane. To improve clarity and reduce visual overlap, we introduce a **token label offset strategy** for all annotations. This document summarizes the method and rationale behind this decision for use in technical documentation and paper writing.

---

## 1. Lane Token Annotation

Function: `draw_lane_tokens()`

### Strategy:

Each lane has an associated `lane_id` and may have a mapped token ID. These are drawn at the **starting point** of the lane, with a fixed vertical offset to avoid label collisions.

* **Lane ID**: Rendered at `(x, y - 9)` using black text.
* **Token ID**: Rendered at `(x, y - 12)` using blue text (e.g., `t6`, `t8`).

### Code Snippet:

```python
# Show lane ID
ax.text(x, y - 9.0, str(lane_id), color="black")

# Show lane token ID
ax.text(x, y - 12, f"t{token_id}", color="blue")
```

---

## 2. Traffic Light Token Annotation

Function: `draw_traffic_light_tokens()`

### Strategy:

Each traffic light is rendered with two pieces of information:

* **Traffic Light Name** (e.g., `TL0`, `TL1`): Rendered at `(x, y - 3)` with a rounded bounding box.
* **Token ID** (e.g., `t0`, `t1`, `t2`): Rendered at `(x, y - 6)` using green text.

### Code Snippet:

```python
# Show traffic light name (e.g., TL0)
ax.text(x, y - 3, f"TL{i}", bbox=dict(boxstyle="round,pad=0.2"))

# Show token ID
ax.text(x, y - 6, f"t{token_id}", color="green")
```

---

## 3. Motivation and Benefits

* **Avoid Label Overlap**: Offsets prevent textual collision at shared spatial coordinates.
* **Semantic Clarity**: Lane IDs and Token IDs are distinct in position and color.
* **Visual Grouping**: The user can visually connect `TLx` → `tx` → controlled lane `lx` easily.
* **Debugging Support**: Enhances interpretability during trajectory and perception verification.

---

## 4. Future Enhancements

* Add arrows showing direction of each traffic light (`dx/dy` vector).
* Color-code lanes controlled by each traffic light.
* Introduce interactive tooltips for web-based rendering.

---

## 5. Sample Visualization

![Sample BEV Token Debug Frame](./assets/bev_debug_frame0.png)

> The image above shows TL0–TL2, each with offset text labels. Tokens and IDs are readable without interference.

---

## Conclusion

This offset annotation strategy provides a simple yet effective approach to multi-layer BEV visualization. It improves the interpretability of map semantic elements and supports both model debugging and research communication.





# BEV Token Annotation Offset Strategy

## Overview

In our Bird’s Eye View (BEV) visualization system, multiple semantic elements such as lane lines, traffic lights, and associated token IDs are rendered on a 2D plane. To improve clarity and reduce visual overlap, we introduce a **token label offset strategy** for all annotations. This document summarizes the method and rationale behind this decision for use in technical documentation and paper writing.

---

## 1. Lane Token Annotation

Function: `draw_lane_tokens()`

### Strategy:

Each lane has an associated `lane_id` and may have a mapped token ID. These are drawn at the **starting point** of the lane, with a fixed vertical offset to avoid label collisions.

* **Lane ID**: Rendered at `(x, y - 9)` using black text.
* **Token ID**: Rendered at `(x, y - 12)` using blue text (e.g., `t6`, `t8`).

### Code Snippet:

```python
# Show lane ID
ax.text(x, y - 9.0, str(lane_id), color="black")

# Show lane token ID
ax.text(x, y - 12, f"t{token_id}", color="blue")
```

---

## 2. Traffic Light Token Annotation

Function: `draw_traffic_light_tokens()`

### Strategy:

Each traffic light is rendered with two pieces of information:

* **Traffic Light Name** (e.g., `TL0`, `TL1`): Rendered at `(x, y - 3)` with a rounded bounding box.
* **Token ID** (e.g., `t0`, `t1`, `t2`): Rendered at `(x, y - 6)` using green text.

### Code Snippet:

```python
# Show traffic light name (e.g., TL0)
ax.text(x, y - 3, f"TL{i}", bbox=dict(boxstyle="round,pad=0.2"))

# Show token ID
ax.text(x, y - 6, f"t{token_id}", color="green")
```

---

## 3. Motivation and Benefits

* **Avoid Label Overlap**: Offsets prevent textual collision at shared spatial coordinates.
* **Semantic Clarity**: Lane IDs and Token IDs are distinct in position and color.
* **Visual Grouping**: The user can visually connect `TLx` → `tx` → controlled lane `lx` easily.
* **Debugging Support**: Enhances interpretability during trajectory and perception verification.

---

## 4. Future Enhancements

* Add arrows showing direction of each traffic light (`dx/dy` vector).
* Color-code lanes controlled by each traffic light.
* Introduce interactive tooltips for web-based rendering.

---

## 5. Sample Visualization

![Sample BEV Token Debug Frame](./assets/bev_debug_frame0.png)

> The image above shows TL0–TL2, each with offset text labels. Tokens and IDs are readable without interference.

---

## Conclusion

This offset annotation strategy provides a simple yet effective approach to multi-layer BEV visualization. It improves the interpretability of map semantic elements and supports both model debugging and research communication.



# 图像可视化标注规范与 token ID 编码说明

## 1. 背景

在 BEV 图像可视化过程中，我们为 \[\[车道线]]（lane）和 \[\[交通信号灯]]（traffic light）等对象分配了 `t0, t1, t2...` 这样的 token ID 标签，用于在图上展示模型的输入 token 编号。

如下图所示：

* 上方绿色圆点 + `TL0, TL1, TL2` 表示交通灯位置与 ID
* 下方蓝色 `t0, t1, t2` 表示 token 编号
* 同样的 `t0` 也出现在车道线标注中

## 2. 技术说明

这种 token ID 的重复是 **刻意的简化行为，仅用于可视化**，不会对模型造成歧义。

### ✅ Token 向量的定义

在实际模型输入中：

* 每一个 token ID 如 `t0` 对应的是一个向量，例如：

  ```python
  traffic_light_token[0] = [0.67, -0.14, 0.21, 1.0]  # 含位置 + 状态等信息
  lane_token[0] = [12.5, 32.6, 0.1, 0.9, 4.2]       # 含起点坐标 + 朝向 + 宽度等
  ```
* 虽然可视化时编号一样，但 token 类型不同，对应的 embedding 向量和位置完全不同。

### ✅ 模型处理逻辑

我们在模型输入编码中做了**实体类型分离**：

* `encode_lanes()` → 构造 lane token tensor
* `encode_traffic_lights()` → 构造 traffic light token tensor
* 二者分别使用独立的 embedding table 或输入 projection 层

### ✅ 图像标注偏移说明

在图像中，为了避免 token 文本重叠导致难以阅读：

* 车道线的 token ID 会稍微往上偏移 `y-12`
* 交通灯的 token ID 会稍微往下偏移 `y-6`

```python
ax.text(x, y - 12, token, fontsize=6, color="blue")     # lane token ID
ax.text(x, y - 6, token_id, fontsize=6, color="purple")  # traffic light token ID
```

此视觉偏移不会影响 token 向量构造，仅用于图像展示。

## 3. 论文中推荐附加解释（建议放在图注中）

> Note: Token IDs (e.g., t0, t1) are **entity-type specific** and reused across different object types (e.g., lanes, traffic lights) in visualization. Each token corresponds to a distinct feature vector in the model input, depending on its type.

## 4. 示例截图

（图像建议放入位置：`BEV Debug Frame 0`）

---

你也可以将这段说明作为补充说明节附加至 appendix A，用于 reviewer 理解 token ID 重复的无歧义性。

