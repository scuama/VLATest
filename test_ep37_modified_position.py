#!/usr/bin/env python3
"""测试37号场景修改机械臂位置后的重放"""
import subprocess
import json
import shutil
import os

episode_dir = "/mnt/disk1/decom/VLATest/results/t-move_n-100_o-0_s-3225323079/openvla-7b_2024/37"
backup_dir = "/mnt/disk1/decom/VLATest/results/t-move_n-100_o-0_s-3225323079/openvla-7b_2024/37_backup"
test_dir = "/mnt/disk1/decom/VLATest/test_ep37_modified"

print("=" * 70)
print("🧪 测试37号场景修改机械臂位置后的重放")
print("=" * 70)

# 备份原始数据（如果还没备份）
if not os.path.exists(backup_dir):
    print(f"\n📦 备份原始数据...")
    shutil.copytree(episode_dir, backup_dir)
    print(f"   ✅ 备份完成: {backup_dir}")

# 创建测试目录
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)
print(f"\n📁 创建测试目录...")
shutil.copytree(backup_dir, test_dir)
print(f"   ✅ 测试目录: {test_dir}")

# 读取原始配置
with open(f"{test_dir}/options.json", 'r') as f:
    options = json.load(f)

# 物体位置
obj_name = options['model_ids'][options['source_obj_id']]
obj_xy = options['obj_init_options'][obj_name]['init_xy']

# 原始机械臂位置（prepackaged默认值）
original_robot_xy = [0.35, 0.21]

# 计算朝向物体的偏移
dx = obj_xy[0] - original_robot_xy[0]
dy = obj_xy[1] - original_robot_xy[1]

# 设置新位置：朝物体方向移动一小段距离
move_ratio = 0.15  # 移动15%的距离
new_robot_xy = [
    original_robot_xy[0] + dx * move_ratio,
    original_robot_xy[1] + dy * move_ratio
]

print(f"\n📋 位置信息:")
print(f"   物体位置 ({obj_name}): [{obj_xy[0]:.4f}, {obj_xy[1]:.4f}]")
print(f"   原始机械臂位置: [{original_robot_xy[0]:.4f}, {original_robot_xy[1]:.4f}]")
print(f"   新机械臂位置: [{new_robot_xy[0]:.4f}, {new_robot_xy[1]:.4f}]")
print(f"   偏移量: Δx={new_robot_xy[0]-original_robot_xy[0]:+.4f}, Δy={new_robot_xy[1]-original_robot_xy[1]:+.4f}")

# 添加自定义机械臂位置到配置
# 注意：不指定 init_rot_quat，让系统使用默认值（带-0.09弧度Z轴旋转）
# 如果指定 [1.0, 0.0, 0.0, 0.0]（无旋转），会导致相机姿态翻转！
options["robot_init_options"] = {
    "init_xy": new_robot_xy,
    # "init_rot_quat": [1.0, 0.0, 0.0, 0.0]  # ← 删除这行，使用默认值
}

with open(f"{test_dir}/options.json", 'w') as f:
    json.dump(options, f, indent=2)

print(f"\n✅ 配置已更新")

# 执行重放
print(f"\n⏳ 开始重放（会生成50张图片）...\n")

cmd = [
    "python3", "/mnt/disk1/decom/VLATest/experiments/replay_openvla_actions.py",
    "--episode_dir", test_dir,
    "--task", "google_robot_move_near_customizable"
]

result = subprocess.run(
    cmd,
    cwd="/mnt/disk1/decom/VLATest"
)

print(f"\n" + "=" * 70)
if result.returncode == 0:
    print("✅ 重放成功")
    
    # 检查生成的图片数量
    import glob
    images = glob.glob(f"{test_dir}/replay_images/*.jpg")
    print(f"📸 生成图片数量: {len(images)}")
    
    # 检查抓取状态
    with open(f"{test_dir}/replay_log.json", 'r') as f:
        replay_log = json.load(f)
    
    grasp_count = sum(1 for step in replay_log.values() if step.get('is_src_obj_grasped', False))
    print(f"🤏 抓取成功步数: {grasp_count}/{len(replay_log)}")
    
    print(f"\n📂 查看结果: {test_dir}")
    
elif result.returncode == -11:
    print("💥 重放失败：段错误 (SIGSEGV)")
else:
    print(f"❌ 重放失败：返回码 {result.returncode}")

print("=" * 70)
