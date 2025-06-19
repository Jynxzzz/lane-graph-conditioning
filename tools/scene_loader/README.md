# 🧠 scene_loader 模块说明

本模块负责加载 Waymo 场景数据，支持以下两类数据入口：

- ✅ 从默认目录随机选择 `.pkl` 场景 → 用于 debug 或全量训练
- ✅ 从已筛选好的列表（如 green-only / stop-sign-only）中随机选择场景 → 用于 curriculum 构建、对照实验等

---

## 🧩 函数说明

### `load_random_scene(scene_dir: str) -> dict`
- 从指定目录中随机挑选一个 `.pkl` 场景加载
- 默认目录为 `DEFAULT_SCENE_DIR`
- 用于 baseline 或快速查看数据结构

---

### `load_selected_scene_list(list_path: str) -> List[str]`
- 从 txt / jsonl 文件中读取场景路径列表
- 每行为一个相对路径（相对于数据根目录）
- 用于绿色场景筛选、策略测试等

---

### `load_random_scene_from_list(scene_list: List[str], base_dir: str) -> dict`
- 从场景路径列表中随机选择一个 `.pkl` 文件加载
- 支持与任意筛选器（例如标签筛选器）组合使用
- 用于结构训练前的数据采样入口

---

### `load_scene_by_index(scene_list: List[str], index: int, base_dir: str) -> dict`
- 从 list 中指定 index 加载某个场景
- 用于 debug 固定 case、可视化等

---

### `load_scene_data(list_path=None, base_dir=None) -> dict`
- 统一入口：如果提供 list_path → 从列表随机加载，否则从目录随机加载
- 适合用于策略训练 / eval 等通用脚本调用

---

## 📦 用法示例

```python
from tools.scene_loader import *

scene_list = load_selected_scene_list("metadata/green_only_list.txt")
scene = load_random_scene_from_list(scene_list, base_dir=DEFAULT_SCENE_DIR)
