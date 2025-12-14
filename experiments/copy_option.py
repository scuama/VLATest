#!/usr/bin/env python3
# mk_single_and_merge_v2.py
import json, pathlib, numpy as np

src_root = pathlib.Path(
    "/mnt/disk1/decom/VLATest/newresult/"
    "t-grasp_n-100_o-0_s-170912623-40/t-grasp_n-100_o-0_s-170912623-40/"
    "openvla-7b_2024/t-grasp_n-100_o-0_s-170912623-40/openvla-7b_2024"
)

# 1. 创建目标目录
group_name = "t-grasp_n-100_o-0_s-170912623-40"
out_dir = pathlib.Path("/mnt/disk1/decom/VLATest/newdata") / group_name
out_dir.mkdir(parents=True, exist_ok=True)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

merged = {}
idx = 0
while True:
    src = src_root / str(idx) / "options.json"
    if not src.exists():
        break
    with src.open() as f:
        data = json.load(f)

    # 2. 单文件：t-grasp_n-100_o-0_s-170912623-40-0.json ...
    single = {"0": data, "seed": 170912623, "num": 1}
    single_path = out_dir / f"{group_name}-{idx}.json"
    with single_path.open("w") as f:
        json.dump(single, f, indent=2, ensure_ascii=False, cls=NpEncoder)

    merged[str(idx)] = data
    print(f"Saved  {single_path}")
    idx += 1

# 3. 合并文件：t-grasp_n-100_o-0_s-170912623-40.json
merged["seed"] = 170912623
merged["num"] = idx
merge_path = out_dir / f"{group_name}.json"
with merge_path.open("w") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False, cls=NpEncoder)

print(f"全部完成！共 {idx} 个任务 → {out_dir}")