import json

index_path = "metadata/scenario_index.jsonl"
output_path = "green_only_list.txt"

matched = []

with open(index_path, "r") as f:
    for line in f:
        try:
            data = json.loads(line)
            if data.get("scene_summary", {}).get("has_only_green") is True:
                matched.append(data["path"])
        except json.JSONDecodeError:
            print("❌ Error decoding line")

with open(output_path, "w") as f:
    for path in matched:
        f.write(path + "\n")

print(f"✅ 筛选完成，共 {len(matched)} 条 green-only 场景，已保存到 {output_path}")
