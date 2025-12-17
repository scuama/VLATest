#!/usr/bin/env python3
"""
简化版：将物品移近机械臂
功能：仅修改配置，将物品向机械臂方向移动指定距离
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path


# ==================== 默认配置参数 ====================
DEFAULT_BASE_DIR = "results/t-move_n-100_o-0_s-3225323079/openvla-7b_2024"

# 机械臂中心位置（估计值）
ROBOT_CENTER = [0.0, 0.1]

# 默认移动比例（0-1），0.5表示移动到中点
DEFAULT_MOVE_RATIO = 0.5


# ==================== 工具函数 ====================

def load_options(episode_dir):
    """加载 options.json"""
    options_path = os.path.join(episode_dir, "options.json")
    with open(options_path, 'r') as f:
        return json.load(f)


def save_options(episode_dir, options):
    """保存 options.json"""
    options_path = os.path.join(episode_dir, "options.json")
    with open(options_path, 'w') as f:
        json.dump(options, f, indent=2)


def backup_original_options(episode_dir):
    """备份原始 options.json"""
    options_path = os.path.join(episode_dir, "options.json")
    backup_path = os.path.join(episode_dir, "origin.json")
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy(options_path, backup_path)
        print(f"✅ 已备份原始配置: {backup_path}")


def calculate_distance(pos1, pos2):
    """计算两点之间的欧氏距离"""
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    return np.sqrt(dx**2 + dy**2)


def move_closer_to_robot(original_xy, move_ratio=0.5):
    """将物体向机械臂方向移动
    
    Args:
        original_xy: 原始位置 [x, y]
        move_ratio: 移动比例 (0-1)，0.5表示移动到中点
    
    Returns:
        新位置 [x, y]
    """
    # 计算朝向机械臂的方向向量
    dx = ROBOT_CENTER[0] - original_xy[0]
    dy = ROBOT_CENTER[1] - original_xy[1]
    
    # 按比例移动
    new_x = original_xy[0] + dx * move_ratio
    new_y = original_xy[1] + dy * move_ratio
    
    # 确保在合理的桌面范围内
    new_x = np.clip(new_x, -0.4, 0.3)
    new_y = np.clip(new_y, -0.2, 0.5)
    
    return [float(new_x), float(new_y)]


def main():
    parser = argparse.ArgumentParser(description="将物品移近机械臂（仅修改配置）")
    
    parser.add_argument('episode_dir', type=str, help="Episode工作目录（包含options.json）")
    parser.add_argument('--move_ratio', type=float, default=DEFAULT_MOVE_RATIO, 
                       help=f"移动比例 (0-1)，默认: {DEFAULT_MOVE_RATIO}")
    
    args = parser.parse_args()
    
    episode_dir = args.episode_dir
    
    if not os.path.exists(episode_dir):
        print(f"❌ Episode目录不存在: {episode_dir}")
        return 1
    
    print("=" * 70)
    print("🎯 将物品移近机械臂")
    print("=" * 70)
    print(f"📍 目录: {episode_dir}")
    print(f"📊 移动比例: {args.move_ratio:.1%}")
    print("=" * 70)
    
    # 备份原始配置
    backup_original_options(episode_dir)
    
    # 加载原始备份配置（用于获取真正的原始位置）
    backup_path = os.path.join(episode_dir, "origin.json")
    if os.path.exists(backup_path):
        with open(backup_path, 'r') as f:
            options = json.load(f)
        print(f"✅ 从备份加载原始配置: origin.json")
    else:
        options = load_options(episode_dir)
        print(f"⚠️  备份不存在，使用当前配置")
    
    # 获取源物体（model_ids 是列表）
    source_obj_id = options["source_obj_id"]
    if isinstance(options["model_ids"], list):
        source_obj = options["model_ids"][source_obj_id]
    else:
        source_obj = options["model_ids"][source_obj_id]
    
    original_xy = options['obj_init_options'][source_obj]['init_xy']
    
    original_dist = calculate_distance(original_xy, ROBOT_CENTER)
    
    print(f"\n🎯 源物体: {source_obj}")
    print(f"📍 原始位置: [{original_xy[0]:.4f}, {original_xy[1]:.4f}]")
    print(f"📐 到机械臂距离: {original_dist:.3f}m")
    
    # 计算新位置
    new_xy = move_closer_to_robot(original_xy, args.move_ratio)
    new_dist = calculate_distance(new_xy, ROBOT_CENTER)
    
    print(f"\n📍 新位置: [{new_xy[0]:.4f}, {new_xy[1]:.4f}]")
    print(f"📏 偏移: Δx={new_xy[0]-original_xy[0]:+.4f}m, Δy={new_xy[1]-original_xy[1]:+.4f}m")
    print(f"📐 新距离: {new_dist:.3f}m (靠近 {original_dist-new_dist:.3f}m)")
    
    # 更新配置
    options["obj_init_options"][source_obj]["init_xy"] = new_xy
    save_options(episode_dir, options)
    
    print("\n" + "=" * 70)
    print("✅ 配置已更新！")
    print("=" * 70)
    print(f"💾 配置文件: {os.path.join(episode_dir, 'options.json')}")
    print(f"📦 原始备份: {os.path.join(episode_dir, 'origin.json')}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
