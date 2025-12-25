#!/usr/bin/env python3
"""
收集 grasp 和 move 任务结果，按成功/失败分类整理
"""

import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

# 配置
BASE_DIR = Path("/mnt/disk1/decom/VLATest")
OUTPUT_DIR = BASE_DIR / "result"

# 源目录配置
GRASP_SOURCE = BASE_DIR / "newresult/t-grasp_n-100_o-0_s-170912623/openvla-7b_2024/t-grasp_n-100_o-0_s-170912623/openvla-7b_2024"
GRASP_IMAGES_SOURCE = BASE_DIR / "newresult/t-grasp_n-100_o-0_s-170912623/openvla-7b_2024/t-grasp_n-100_o-0_s-170912623/images/openvla-7b_2024"

MOVE_SOURCE = BASE_DIR / "results/t-move_n-100_o-0_s-3225323079/openvla-7b_2024"
MOVE_IMAGES_SOURCE = BASE_DIR / "results/t-move_n-100_o-0_s-3225323079/images/openvla-7b_2024"

# 需要复制的文件
REQUIRED_FILES = ["log.json", "actions.json", "actions.npy", "options.json"]

MODEL_NAME = "openvla-7b_2024"


def check_success_from_log(log_file):
    """从log.json判断任务是否成功"""
    try:
        with open(log_file, 'r') as f:
            data = json.load(f)
        
        # 找到最后一步
        if not data:
            return False
        
        last_key = max(data.keys(), key=int)
        last_step = data[last_key]
        
        # 获取success字段
        success_value = last_step.get("success", False)
        
        # 处理字符串和布尔值两种情况
        if isinstance(success_value, str):
            return success_value.lower() == "true"
        return bool(success_value)
    
    except Exception as e:
        print(f"  错误读取 {log_file}: {e}")
        return False


def copy_scene_data(source_episode_dir, images_source_dir, target_dir, scene_id, task_type):
    """
    复制单个场景的数据
    
    Args:
        source_episode_dir: 源场景目录（包含json和npy文件）
        images_source_dir: 图片源目录
        target_dir: 目标目录
        scene_id: 场景ID
        task_type: 任务类型 (grasp/move)
    """
    try:
        # 创建目标场景目录
        scene_dir = target_dir / f"scene_{scene_id}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制必需的文件
        copied_files = []
        for filename in REQUIRED_FILES:
            src_file = source_episode_dir / filename
            if src_file.exists():
                dst_file = scene_dir / filename
                shutil.copy2(src_file, dst_file)
                copied_files.append(filename)
        
        # 复制图片目录
        images_dir = scene_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        if images_source_dir and images_source_dir.exists():
            image_count = 0
            for img_file in images_source_dir.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                    shutil.copy2(img_file, images_dir / img_file.name)
                    image_count += 1
            
            return True, len(copied_files), image_count
        else:
            print(f"  警告: 图片目录不存在: {images_source_dir}")
            return True, len(copied_files), 0
    
    except Exception as e:
        print(f"  错误复制场景 {scene_id}: {e}")
        return False, 0, 0


def process_grasp_task():
    """处理 grasp 任务"""
    print("\n" + "="*70)
    print("处理 GRASP 任务")
    print("="*70)
    
    if not GRASP_SOURCE.exists():
        print(f"错误: 源目录不存在: {GRASP_SOURCE}")
        return
    
    # 创建输出目录
    success_dir = OUTPUT_DIR / MODEL_NAME / "grasp" / "success"
    failure_dir = OUTPUT_DIR / MODEL_NAME / "grasp" / "failure"
    success_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有场景目录
    scene_dirs = sorted([d for d in GRASP_SOURCE.iterdir() if d.is_dir() and d.name.isdigit()],
                       key=lambda x: int(x.name))
    
    print(f"找到 {len(scene_dirs)} 个场景")
    
    stats = {"success": 0, "failure": 0, "error": 0}
    
    for scene_dir in tqdm(scene_dirs, desc="处理场景"):
        scene_id = scene_dir.name
        log_file = scene_dir / "log.json"
        
        if not log_file.exists():
            print(f"  警告: 场景 {scene_id} 缺少 log.json")
            stats["error"] += 1
            continue
        
        # 判断成功/失败
        is_success = check_success_from_log(log_file)
        
        # 确定目标目录
        target_dir = success_dir if is_success else failure_dir
        
        # 获取图片源目录 (在统一的images目录下)
        images_source = GRASP_IMAGES_SOURCE / scene_id
        
        # 复制数据
        success_copy, file_count, img_count = copy_scene_data(
            scene_dir, images_source, target_dir, scene_id, "grasp"
        )
        
        if success_copy:
            if is_success:
                stats["success"] += 1
            else:
                stats["failure"] += 1
        else:
            stats["error"] += 1
    
    # 打印统计
    print(f"\nGRASP 任务统计:")
    print(f"  成功案例: {stats['success']}")
    print(f"  失败案例: {stats['failure']}")
    print(f"  错误: {stats['error']}")
    print(f"  总计: {len(scene_dirs)}")


def process_move_task():
    """处理 move 任务"""
    print("\n" + "="*70)
    print("处理 MOVE 任务")
    print("="*70)
    
    if not MOVE_SOURCE.exists():
        print(f"错误: 源目录不存在: {MOVE_SOURCE}")
        return
    
    # 创建输出目录
    success_dir = OUTPUT_DIR / MODEL_NAME / "move" / "success"
    failure_dir = OUTPUT_DIR / MODEL_NAME / "move" / "failure"
    success_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有场景目录
    scene_dirs = sorted([d for d in MOVE_SOURCE.iterdir() if d.is_dir() and d.name.isdigit()],
                       key=lambda x: int(x.name))
    
    print(f"找到 {len(scene_dirs)} 个场景")
    
    stats = {"success": 0, "failure": 0, "error": 0}
    
    for scene_dir in tqdm(scene_dirs, desc="处理场景"):
        scene_id = scene_dir.name
        log_file = scene_dir / "log.json"
        
        if not log_file.exists():
            print(f"  警告: 场景 {scene_id} 缺少 log.json")
            stats["error"] += 1
            continue
        
        # 判断成功/失败
        is_success = check_success_from_log(log_file)
        
        # 确定目标目录
        target_dir = success_dir if is_success else failure_dir
        
        # 获取图片源目录 (在不同的位置)
        images_source = MOVE_IMAGES_SOURCE / scene_id
        
        # 复制数据
        success_copy, file_count, img_count = copy_scene_data(
            scene_dir, images_source, target_dir, scene_id, "move"
        )
        
        if success_copy:
            if is_success:
                stats["success"] += 1
            else:
                stats["failure"] += 1
        else:
            stats["error"] += 1
    
    # 打印统计
    print(f"\nMOVE 任务统计:")
    print(f"  成功案例: {stats['success']}")
    print(f"  失败案例: {stats['failure']}")
    print(f"  错误: {stats['error']}")
    print(f"  总计: {len(scene_dirs)}")


def main():
    print("="*70)
    print("任务结果收集脚本")
    print("="*70)
    print(f"输出目录: {OUTPUT_DIR}")
    
    # 创建输出根目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 处理两个任务
    process_grasp_task()
    process_move_task()
    
    print("\n" + "="*70)
    print("完成！")
    print("="*70)
    print(f"\n结果已保存到: {OUTPUT_DIR}")
    print(f"\n目录结构:")
    print(f"  {OUTPUT_DIR / MODEL_NAME / 'grasp' / 'success'}")
    print(f"  {OUTPUT_DIR / MODEL_NAME / 'grasp' / 'failure'}")
    print(f"  {OUTPUT_DIR / MODEL_NAME / 'move' / 'success'}")
    print(f"  {OUTPUT_DIR / MODEL_NAME / 'move' / 'failure'}")


if __name__ == "__main__":
    main()
