#!/usr/bin/env python3
"""
从优化后的options.json生成批量模拟配置（任务类型修复版）
关键修复：添加 task 字段，使 openVLA.py 能正确识别任务类型
"""

import os
import json
import argparse
import subprocess
from datetime import datetime


# ==================== 配置参数 ====================
DEFAULT_BASE_DIR = "/mnt/disk1/decom/VLATest/newresult/t-grasp_n-100_o-0_s-170912623-2-0/openvla-7b_2024/t-grasp_n-100_o-0_s-170912623-2/openvla-7b_2024"
DEFAULT_OUTPUT_DIR = "/mnt/disk1/decom/VLATest/data"
DEFAULT_SEED = 2498586606
DEFAULT_TASK_TYPE = "grasp"  # 关键：默认任务类型
MODEL_NAME = "openvla-7b"


# ==================== 工具函数 ====================

def load_json(file_path):
    """安全加载JSON文件"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 加载失败 {file_path}: {e}")
        return None


def save_json(data, file_path, indent=2):
    """保存JSON文件"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
        print(f"✅ 已保存: {file_path}")
        return True
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return False


def scan_optimized_episodes(base_dir):
    """扫描所有包含优化后 options.json 的episode目录"""
    print("=" + "=" * 69)
    print("🤖 开始扫描优化后的episode配置...")
    print("=" + "=" * 69)
    
    if not os.path.exists(base_dir):
        print(f"❌ 目录不存在: {base_dir}")
        return {}
    
    optimized_episodes = {}
    episode_dirs = sorted([d for d in os.listdir(base_dir) 
                          if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()])
    
    print(f"📂 检查 {len(episode_dirs)} 个episode...")
    
    for ep_id in episode_dirs:
        episode_path = os.path.join(base_dir, ep_id)
        options_path = os.path.join(episode_path, "options.json")
        
        if not os.path.exists(options_path):
            print(f"   ⏭️  跳过 {ep_id}: 缺少 options.json")
            continue
        
        options = load_json(options_path)
        if options is None:
            print(f"   ❌ 跳过 {ep_id}: 无法加载配置")
            continue
        
        if "robot_init_options" not in options:
            print(f"   ⚠️  警告 {ep_id}: 未找到 robot_init_options（可能未优化）")
            continue
        
        optimized_episodes[ep_id] = options
        print(f"   ✅ {ep_id}: 物体={options.get('model_id', 'unknown')}, "
              f"机械臂位置={options['robot_init_options']['init_xy']}")
    
    print(f"✅ 共找到 {len(optimized_episodes)} 个优化后的配置")
    return optimized_episodes


def build_simulation_config(episode_data, seed=None, task_type=DEFAULT_TASK_TYPE):
    """
    构建模拟需要的配置格式
    关键修复：添加 task 字段，使 openVLA.py 能正确识别任务类型
    """
    config = {}
    sorted_episodes = sorted(episode_data.items(), key=lambda x: int(x[0]))
    
    for idx, (ep_id, options) in enumerate(sorted_episodes):
        model_id = options.get("model_id", "unknown")
        obj_init_options = options.get("obj_init_options", {})
        robot_init_options = options.get("robot_init_options", {})
        
        config[str(idx)] = {
            "model_id": model_id,
            "obj_init_options": obj_init_options,
            "robot_init_options": robot_init_options
        }
    
    # 关键：添加任务信息
    config["seed"] = seed if seed is not None else DEFAULT_SEED
    config["num"] = len(sorted_episodes)
    config["task"] = f"google_robot_{task_type}_customizable"  # 任务类型
    
    return config


def detect_run_script():
    """自动检测运行脚本"""
    possible_paths = [
        "/mnt/disk1/decom/VLATest/run_openVLA.sh",
        "/mnt/disk1/decom/VLATest/run_openvla.sh",
        "/mnt/disk1/decom/VLATest/experiments/run_openVLA.sh",
        os.path.join(os.path.dirname(__file__), "..", "run_openVLA.sh"),
        os.path.join(os.path.dirname(__file__), "..", "run_openvla.sh")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def run_simulation(config_path, model_name=MODEL_NAME, result_dir=None):
    """运行OpenVLA模拟"""
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    # 自动检测脚本
    run_script = detect_run_script()
    if not run_script:
        print(f"❌ 无法找到运行脚本 run_openVLA.sh")
        print(f"   请在以下位置查找或手动指定：")
        for path in possible_paths:
            print(f"      - {path}")
        return False
    
    cmd = ["bash", run_script, model_name, config_path, result_dir if result_dir else "../optimized_results"]
    
    print("\n" + "=" + "=" * 69)
    print("🚀 启动模拟...")
    print(f"📋 配置: {config_path}")
    print(f"🤖 模型: {model_name}")
    print(f"📜 脚本: {run_script}")
    print(f"📂 结果目录: {cmd[-1]}")
    print("=" + "=" * 69)
    
    try:
        result = subprocess.run(cmd, cwd="/mnt/disk1/decom/VLATest", capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"\n❌ 执行模拟时出错: {e}")
        return False


def print_config_summary(config):
    """打印配置摘要"""
    print("\n" + "=" + "=" * 69)
    print("📊 生成的模拟配置摘要")
    print("=" + "=" * 69)
    print(f"🔢 Episode数量: {config['num']}")
    print(f"🌱 Seed: {config['seed']}")
    print(f"🎯 任务类型: {config.get('task', 'unknown')}")  # 显示任务类型
    print(f"\n📋 前3个episode预览:")
    
    for i in range(min(3, config['num'])):
        ep_key = str(i)
        ep_data = config[ep_key]
        print(f"\n【Episode {i}】")
        print(f"   物体: {ep_data['model_id']}")
        print(f"   物体位置: {ep_data['obj_init_options']['init_xy']}")
        print(f"   物体朝向: {ep_data['obj_init_options']['orientation'][:2]}...")
        print(f"   机械臂位置: {ep_data['robot_init_options']['init_xy']}")
    
    print("\n" + "=" + "=" * 69)


def main():
    parser = argparse.ArgumentParser(
        description="从优化后的options.json生成批量模拟配置（任务类型修复版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 生成并运行抓取任务配置
  python3 generate_optimized_config.py \
    --base_dir /path/to/results \
    --output data/grasp_batch.json \
    --task grasp \
    --run_simulation
  
  # 生成并运行搬运任务配置
  python3 generate_optimized_config.py \
    --base_dir /path/to/results \
    --output data/move_batch.json \
    --task move \
    --run_simulation
        """
    )
    
    parser.add_argument(
        '--base_dir',
        type=str,
        default=DEFAULT_BASE_DIR,
        help="包含优化后options.json的基础目录"
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="输出配置文件路径（默认: data/optimized_config_<timestamp>.json）"
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help="模拟seed值（默认: 2498586606）"
    )
    
    parser.add_argument(
        '--task',
        type=str,
        choices=["grasp", "move"],
        default=DEFAULT_TASK_TYPE,
        help="任务类型（默认: grasp）"
    )
    
    parser.add_argument(
        '--run_simulation',
        action='store_true',
        help="生成配置后立即运行模拟"
    )
    
    parser.add_argument(
        '--result_dir',
        type=str,
        default="../optimized_results",
        help="模拟结果输出目录（默认: ../optimized_results）"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=MODEL_NAME,
        help=f"模型名称（默认: {MODEL_NAME}）"
    )
    
    args = parser.parse_args()
    
    # 1. 扫描优化配置
    episodes = scan_optimized_episodes(args.base_dir)
    
    if not episodes:
        print("\n❌ 未找到任何优化后的配置")
        return 1
    
    # 2. 构建配置（关键：传递 task_type）
    print("\n" + "=" + "=" * 69)
    print("🔧 构建模拟配置文件...")
    print("=" + "=" * 69)
    
    sim_config = build_simulation_config(episodes, seed=args.seed, task_type=args.task)
    print_config_summary(sim_config)
    
    # 3. 保存配置
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(DEFAULT_OUTPUT_DIR, f"optimized_{args.task}_{timestamp}.json")
    else:
        output_path = args.output
    
    if not save_json(sim_config, output_path):
        return 1
    
    print(f"\n💾 配置文件已保存: {os.path.abspath(output_path)}")
    
    # 4. 运行模拟
    if args.run_simulation:
        success = run_simulation(output_path, model_name=args.model, result_dir=args.result_dir)
        return 0 if success else 1
    
    print("\n✅ 配置生成完成！")
    print(f"💡 手动运行命令:")
    print(f"   cd /mnt/disk1/decom/VLATest")
    print(f"   bash run_openVLA.sh {args.model} {output_path} ../optimized_results")
    return 0


if __name__ == "__main__":
    exit(main())