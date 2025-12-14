#!/usr/bin/env python3
"""测试37号场景的重放"""
import json
import shutil
import os

episode_dir = "/mnt/disk1/decom/VLATest/results/t-move_n-100_o-0_s-3225323079/openvla-7b_2024/37"
backup_dir = "/mnt/disk1/decom/VLATest/results/t-move_n-100_o-0_s-3225323079/openvla-7b_2024/37_backup"

print("=" * 70)
print("🧪 测试37号场景重放")
print("=" * 70)

# 备份原始数据
if not os.path.exists(backup_dir):
    print("\n📦 备份原始数据...")
    shutil.copytree(episode_dir, backup_dir)
    print(f"   ✅ 备份完成: {backup_dir}")
else:
    print(f"\n✅ 备份已存在: {backup_dir}")

# 读取原始配置
with open(f"{episode_dir}/options.json", 'r') as f:
    original_options = json.load(f)

print(f"\n📋 原始配置:")
print(f"   model_ids: {original_options.get('model_ids')}")
print(f"   source_obj_id: {original_options.get('source_obj_id')}")
if 'robot_init_options' in original_options:
    print(f"   robot_init_options: {original_options['robot_init_options']}")
else:
    print(f"   robot_init_options: ❌ 不存在（将使用prepackaged默认值）")

# 测试1: 原始配置重放
print(f"\n" + "=" * 70)
print("🔬 测试1: 使用原始配置重放")
print("=" * 70)

import subprocess
cmd = [
    "python3", "/mnt/disk1/decom/VLATest/experiments/replay_openvla_actions.py",
    "--episode_dir", episode_dir,
    "--task", "google_robot_move_near_customizable",
    "--render_every", "999999"
]

result = subprocess.run(
    cmd,
    cwd="/mnt/disk1/decom/VLATest",
    capture_output=True,
    text=True,
    timeout=60
)

if result.returncode == 0:
    print("✅ 重放成功")
elif result.returncode == -11:
    print("💥 重放失败：段错误 (SIGSEGV)")
else:
    print(f"❌ 重放失败：返回码 {result.returncode}")
    errors = [line for line in result.stderr.split('\n') 
             if line and 'GLFW' not in line and 'svulkan2' not in line]
    if errors:
        print(f"   错误信息: {errors[-1][:100]}")

# 测试2: 添加自定义机械臂位置
print(f"\n" + "=" * 70)
print("🔬 测试2: 添加自定义机械臂位置后重放")
print("=" * 70)

modified_options = original_options.copy()
custom_position = [-0.05, 0.10]  # 朝向物体方向移动
modified_options["robot_init_options"] = {
    "init_xy": custom_position,
    "init_rot_quat": [1.0, 0.0, 0.0, 0.0]
}

print(f"   设置机械臂位置: {custom_position}")

with open(f"{episode_dir}/options.json", 'w') as f:
    json.dump(modified_options, f, indent=2)

result = subprocess.run(
    cmd,
    cwd="/mnt/disk1/decom/VLATest",
    capture_output=True,
    text=True,
    timeout=60
)

if result.returncode == 0:
    print("✅ 重放成功")
elif result.returncode == -11:
    print("💥 重放失败：段错误 (SIGSEGV)")
else:
    print(f"❌ 重放失败：返回码 {result.returncode}")
    errors = [line for line in result.stderr.split('\n') 
             if line and 'GLFW' not in line and 'svulkan2' not in line]
    if errors:
        print(f"   错误信息: {errors[-1][:100]}")

# 恢复原始配置
print(f"\n📋 恢复原始配置...")
shutil.copy(f"{backup_dir}/options.json", f"{episode_dir}/options.json")
print("   ✅ 已恢复")

print("\n" + "=" * 70)
print("✅ 测试完成")
print("=" * 70)
