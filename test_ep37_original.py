#!/usr/bin/env python3
"""只测试37号场景原始配置的重放"""
import subprocess
import json

episode_dir = "/mnt/disk1/decom/VLATest/results/t-move_n-100_o-0_s-3225323079/openvla-7b_2024/37"

print("=" * 70)
print("🧪 测试37号场景原始配置重放")
print("=" * 70)

# 读取原始配置
with open(f"{episode_dir}/options.json", 'r') as f:
    options = json.load(f)

print(f"\n📋 配置信息:")
print(f"   model_ids: {options.get('model_ids')}")
print(f"   source_obj: {options['model_ids'][options.get('source_obj_id')]}")
if 'robot_init_options' in options:
    print(f"   robot_init_xy: {options['robot_init_options']['init_xy']}")
else:
    print(f"   robot_init_xy: ❌ 不存在（将使用默认值 [0.35, 0.21]）")

print(f"\n⏳ 开始重放...\n")

cmd = [
    "python3", "/mnt/disk1/decom/VLATest/experiments/replay_openvla_actions.py",
    "--episode_dir", episode_dir,
    "--task", "google_robot_move_near_customizable",
    "--render_every", "999999"
]

result = subprocess.run(
    cmd,
    cwd="/mnt/disk1/decom/VLATest",
    timeout=60
)

print(f"\n" + "=" * 70)
if result.returncode == 0:
    print("✅ 重放成功")
elif result.returncode == -11:
    print("💥 重放失败：段错误 (SIGSEGV)")
else:
    print(f"❌ 重放失败：返回码 {result.returncode}")
print("=" * 70)
