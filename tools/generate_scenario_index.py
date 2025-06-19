import argparse
import json
import os

from tqdm import tqdm

from tools.scenario_indexer import analyze_scenario


def generate_index(scene_dir, output_path):
    """
    遍历 scene_dir 下所有 .pkl 文件，调用 analyze_scenario 获取信息，并保存为 .jsonl 格式。
    每行一个 JSON 对象，更适合大规模筛选。
    """
    corrupted_list = []

    fnames = [f for f in os.listdir(scene_dir) if f.endswith(".pkl")]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as fout:
        for fname in tqdm(fnames, desc="Indexing scenarios"):
            fpath = os.path.join(scene_dir, fname)

            try:
                info = analyze_scenario(fpath)
                fout.write(json.dumps(info) + "\n")

                if info.get("corrupted", False):
                    corrupted_list.append(fpath)

            except Exception as e:
                print(f"[!] Error parsing {fpath}: {e}")
                corrupted_list.append(fpath)

    # 保存错误列表（可选）
    if corrupted_list:
        errlog_path = os.path.join(os.path.dirname(output_path), "corrupted_files.txt")
        with open(errlog_path, "w") as ferr:
            for path in corrupted_list:
                ferr.write(path + "\n")
        print(f"[!] 共 {len(corrupted_list)} 个损坏文件已记录：{errlog_path}")

    print(f"[✓] JSONL 索引已保存至 {output_path}，共 {len(fnames)} 条场景")


# def generate_index(scene_dir, output_path):
#     index_list = []
#     corrupted_list = []
#
#     fnames = [f for f in os.listdir(scene_dir) if f.endswith(".pkl")]
#
#     for fname in tqdm(fnames, desc="Processing scenarios"):
#         fpath = os.path.join(scene_dir, fname)
#
#         info = analyze_scenario(fpath)
#         index_list.append(info)
#
#         if info.get("corrupted", False):
#             corrupted_list.append(fpath)
#
#     # 保存主索引文件
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     with open(output_path, "w") as f:
#         json.dump(index_list, f, indent=2)
#
#     # 保存错误列表（可选）
#     if corrupted_list:
#         error_log = os.path.join(os.path.dirname(output_path), "corrupted_files.txt")
#         with open(error_log, "w") as f:
#             for path in corrupted_list:
#                 f.write(path + "\n")
#         print(f"[!] {len(corrupted_list)} corrupted files logged to {error_log}")
#
#     print(f"[✓] Scenario index saved to {output_path} | Total: {len(index_list)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene_dir",
        type=str,
        default="/home/xingnan/VideoDataInbox/scenario_dreamer_waymo/train",
        help="路径：包含 .pkl 场景文件的文件夹",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metadata/scenario_index.jsonl",
        help="输出的索引 json 文件路径",
    )
    args = parser.parse_args()

    generate_index(args.scene_dir, args.output)
