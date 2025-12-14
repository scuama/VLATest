import os
import json
import sys

def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return None

def main(root):
    print(f"🔍 输入路径: {root}")

    # 第一层: root/openvla-7b_2024
    lv2 = os.path.join(root, "openvla-7b_2024")
    if not os.path.isdir(lv2):
        print("❌ 找不到 openvla-7b_2024")
        return

    # 第二层: root/openvla-7b_2024/<task_dir>
    dirs = [d for d in os.listdir(lv2)]
    if len(dirs) != 1:
        print("❌ 第二层目录不唯一:", dirs)
        return

    task_dir = os.path.join(lv2, dirs[0])
    print(f"📁 发现任务目录: {task_dir}")

    # 第三层: task_dir/openvla-7b_2024
    lv4 = os.path.join(task_dir, "openvla-7b_2024")
    if not os.path.isdir(lv4):
        print("❌ 找不到第三层 openvla-7b_2024:", lv4)
        return

    # 最终 episode 列表
    episode_dirs = sorted([
        os.path.join(lv4, d)
        for d in os.listdir(lv4)
        if d.isdigit()
    ])

    print(f"📂 找到 {len(episode_dirs)} 个 episode")

    total = len(episode_dirs)
    success = 0

    results = {}

    for ep_path in episode_dirs:
        ep = os.path.basename(ep_path)

        log_path = os.path.join(ep_path, "log.json")
        log = load_json(log_path)

        if log is None:
            results[ep] = {"success": False, "error": "missing log.json"}
            continue

        succ = log.get("success", False)
        if succ:
            success += 1

        results[ep] = {
            "success": succ,
            "info": log.get("info", "")
        }

    rate = success / total if total > 0 else 0

    print("\n===== 📊 Summary =====")
    print(f"Total:   {total}")
    print(f"Success: {success}")
    print(f"Rate:    {rate:.2%}")

    # 保存 JSON
    out_path = os.path.join(root, "summary.json")
    with open(out_path, "w") as f:
        json.dump({
            "total": total,
            "success": success,
            "rate": rate,
            "details": results
        }, f, indent=2)

    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 summary.py <result_dir>")
        sys.exit(1)

    main(sys.argv[1])
