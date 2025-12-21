#!/usr/bin/env python3
"""
优化流程统一入口
功能：
1. 读取任务配置（如 move/batch_config.json）
2. 跳过已成功的案例
3. 复制原始配置到工作目录（如 move/episodes/）
4. 应用修复策略
5. 运行推理，结果保存到 move/results/
6. 收集成功案例到 move/success/
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


# ==================== 配置 ====================
# 项目根目录
PROJECT_ROOT = "/mnt/disk1/decom/VLATest"

# 虚拟环境 Python（如果存在）
VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv/bin/python3")
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = "python3"  # 回退到系统 Python

OPENVLA_SCRIPT = "/mnt/disk1/decom/VLATest/experiments/openVLA.py"

STRATEGY_SCRIPTS = {
    "optimize_grasp": "/mnt/disk1/decom/VLATest/optimization/fix_strategy_optimize_grasp.py",
    "move_closer": "/mnt/disk1/decom/VLATest/optimization/fix_strategy_move_closer.py",
    "replace_object": "/mnt/disk1/decom/VLATest/optimization/fix_strategy_replace_object.py",
}


# ==================== 工具函数 ====================

def load_config(task_type):
    """加载任务配置"""
    config_file = Path(f"optimization/{task_type}/batch_config.json")
    
    if not config_file.exists():
        print(f"❌ 配置文件不存在: {config_file}")
        return None
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # 设置默认值
    config.setdefault('task', task_type)
    config.setdefault('model', 'openvla-7b')
    config.setdefault('lora_path', None)
    
    return config


def get_successful_episodes(success_dir):
    """获取已成功的 episode 列表"""
    success_dir = Path(success_dir)
    if not success_dir.exists():
        return set()
    
    # 读取所有已成功的 episode ID
    # 结构: success/<episode_id>/
    successful = set()
    for episode_dir in success_dir.iterdir():
        if episode_dir.is_dir() and episode_dir.name.isdigit():
            successful.add(episode_dir.name)  # 添加 episode ID（如 "2", "7", "44"）
    
    return successful


def check_success_from_log(log_file):
    """检查推理是否成功"""
    if not Path(log_file).exists():
        return False
    
    try:
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        
        # 获取最后一步
        step_keys = [k for k in log_data.keys() if k.isdigit()]
        if not step_keys:
            return False
        
        last_step_key = str(max(int(k) for k in step_keys))
        last_step = log_data[last_step_key]
        
        success_value = last_step.get("success", False)
        if isinstance(success_value, str):
            return success_value.lower() == "true"
        return bool(success_value)
        
    except Exception as e:
        print(f"⚠️  读取日志失败: {e}")
        return False


def copy_episode_config(source_dir, target_dir):
    """复制 episode 配置到工作目录"""
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # 创建目标目录
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制 options.json
    options_file = source_dir / "options.json"
    if not options_file.exists():
        print(f"❌ 源配置不存在: {options_file}")
        return False
    
    shutil.copy2(options_file, target_dir / "options.json")
    
    # 复制其他必要文件（如果存在）
    for fname in ["actions.npy", "scene_config.json"]:
        src_file = source_dir / fname
        if src_file.exists():
            shutil.copy2(src_file, target_dir / fname)
    
    print(f"   ✅ 配置已复制: {source_dir.name} -> {target_dir.name}")
    return True


def apply_strategy(episode_id, strategy, case_params, work_dir, base_dir, task_type):
    """应用修复策略"""
    strategy_script = STRATEGY_SCRIPTS.get(strategy)
    if not strategy_script:
        print(f"❌ 未知策略: {strategy}")
        return False
    
    print(f"\n   🔧 应用策略: {strategy}")
    
    # 构建命令 - 所有策略都只需要 episode_dir
    cmd = ["python3", strategy_script, work_dir]
    
    # 策略特定参数
    if strategy == "optimize_grasp":
        direction = case_params.get('direction', 'right-up')
        attempts = case_params.get('attempts', 10)
        cmd.extend([direction, "--attempts", str(attempts)])
    elif strategy == "move_closer":
        if 'move_ratio' in case_params:
            cmd.extend(["--move_ratio", str(case_params['move_ratio'])])
    elif strategy == "replace_object":
        new_object = case_params.get('new_object', 'coke_can')
        cmd.extend(["--new_object", new_object])
    
    try:
        result = subprocess.run(
            cmd,
            cwd="/mnt/disk1/decom/VLATest",
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"   ✅ 策略应用成功")
            return True
        else:
            print(f"   ⚠️  策略应用失败")
            if result.stderr:
                print(f"   错误: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"   ⏱️  策略执行超时")
        return False
    except Exception as e:
        print(f"   ❌ 策略执行异常: {e}")
        return False


def run_inference(episode_dir, result_dir, episode_id, task_type, model="openvla-7b", lora_path=None):
    """运行推理
    
    将优化后的options.json转换为数据集格式并运行推理
    """
    episode_dir = Path(episode_dir).absolute()
    result_dir = Path(result_dir).absolute()
    result_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n   🚀 运行推理...")
    
    # 1. 读取优化后的 options.json
    options_file = episode_dir / "options.json"
    if not options_file.exists():
        print(f"   ❌ 找不到 options.json: {options_file}")
        return False, None
    
    try:
        with open(options_file, 'r') as f:
            options = json.load(f)
    except Exception as e:
        print(f"   ❌ 读取 options.json 失败: {e}")
        return False, None
    
    # 2. 从 options.json 提取源物体信息
    try:
        model_ids = options.get("model_ids", [])
        source_obj_id = options.get("source_obj_id", 0)
        
        if isinstance(model_ids, list) and 0 <= source_obj_id < len(model_ids):
            source_model = model_ids[source_obj_id]
        else:
            print(f"   ⚠️  无法确定源物体，使用第一个物体")
            source_model = model_ids[0] if model_ids else "unknown"
        
        obj_init_options = options.get("obj_init_options", {})
        
        # 获取源物体的初始化选项
        if source_model in obj_init_options:
            source_init_opts = obj_init_options[source_model]
        else:
            print(f"   ⚠️  源物体 {source_model} 没有初始化选项")
            source_init_opts = {}
        
        print(f"   📍 源物体: {source_model}")
        
    except Exception as e:
        print(f"   ❌ 解析 options.json 失败: {e}")
        return False, None
    
    # 3. 创建临时数据集 JSON
    # 格式: {"0": {"model_id": "...", "obj_init_options": {...}}, "seed": ..., "num": 1}
    dataset = {
        "0": {
            "model_id": source_model,
            "obj_init_options": source_init_opts
        },
        "seed": options.get("seed", 0),
        "num": 1
    }
    
    # 创建临时数据集文件
    temp_dataset_file = episode_dir / f"temp_dataset_{episode_id}.json"
    try:
        with open(temp_dataset_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"   📄 创建临时数据集: {temp_dataset_file.name}")
    except Exception as e:
        print(f"   ❌ 创建临时数据集失败: {e}")
        return False, None
    
    # 4. 构建推理命令
    cmd = [
        "python3", OPENVLA_SCRIPT,
        "--data", str(temp_dataset_file),
        "--output", str(result_dir),
        "--model", model
    ]
    
    if lora_path:
        cmd.extend(["--lora_path", lora_path])
    
    try:
        result = subprocess.run(
            cmd,
            cwd="/mnt/disk1/decom/VLATest",
            capture_output=True,
            text=True
        )
        
        # 清理临时文件
        if temp_dataset_file.exists():
            temp_dataset_file.unlink()
        
        if result.returncode != 0:
            print(f"   ❌ 推理失败")
            if result.stderr:
                print(f"   错误: {result.stderr[:500]}")
            return False, None
        
        # 5. 查找生成的 log.json
        # openVLA.py 会创建类似 result_dir/temp_dataset_ep0/openvla-7b_<seed>/0/log.json 的结构
        log_file = None
        for subdir in result_dir.rglob("log.json"):
            log_file = subdir
            break
        
        if not log_file or not log_file.exists():
            print(f"   ⚠️  推理完成但未找到 log.json")
            return False, None
        
        print(f"   ✅ 推理完成，生成日志: {log_file}")
        return True, log_file
        
    except subprocess.TimeoutExpired:
        # 清理临时文件
        if temp_dataset_file.exists():
            temp_dataset_file.unlink()
        print(f"   ⏱️  推理超时")
        return False, None
    except Exception as e:
        # 清理临时文件
        if temp_dataset_file.exists():
            temp_dataset_file.unlink()
        print(f"   ❌ 推理异常: {e}")
        return False, None


def collect_success(source_file_dir, success_base_dir, episode_id):
    """
    收集成功案例到简洁结构
    
    结构: success_base_dir/<episode_id>/
          ├── log.json
          ├── actions.npy
          ├── actions.json
          ├── options.json
          └── images/
              ├── 0.jpg
              └── ...
    
    Args:
        source_file_dir: 源文件目录（包含log.json等的目录）
        success_base_dir: 成功案例基础目录 (如 optimization/move/success)
        episode_id: Episode ID（如 "2", "7", "44"）
    """
    source_file_dir = Path(source_file_dir)
    success_base_dir = Path(success_base_dir)
    
    # 创建简洁结构目录: success/<episode_id>/
    target_dir = success_base_dir / episode_id
    
    # 如果目录已存在，先删除
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"   📦 收集成功案例到: {target_dir}")
    
    # 复制结果文件 (log.json, actions.npy, actions.json, options.json)
    files_copied = 0
    for filename in ["log.json", "actions.npy", "actions.json", "options.json"]:
        src_file = source_file_dir / filename
        if src_file.exists():
            shutil.copy2(src_file, target_dir / filename)
            files_copied += 1
    
    # 复制图片目录到 episode 目录下
    source_images_dir = source_file_dir / "images"
    target_images_dir = target_dir / "images"
    
    if source_images_dir.exists() and source_images_dir.is_dir():
        target_images_dir.mkdir(exist_ok=True)
        for img_file in source_images_dir.iterdir():
            if img_file.is_file():
                shutil.copy2(img_file, target_images_dir / img_file.name)
                files_copied += 1
    
    print(f"   ✅ 已收集 {files_copied} 个文件")
    return True


# ==================== 主流程 ====================

def process_case(case, config, task_type, skip_successful=True):
    """处理单个案例 - 仅应用策略"""
    episode_id = case['episode_id']
    strategy = case.get('strategy', 'none')
    skip_optimization = case.get('skip_optimization', False)
    
    print(f"\n{'='*70}")
    print(f"📋 处理案例: Episode {episode_id}")
    print(f"   策略: {strategy}")
    if skip_optimization:
        print(f"   🔄 跳过优化，使用已有配置")
    print(f"{'='*70}")
    
    # 目录结构
    task_dir = Path(f"optimization/{task_type}")
    episodes_dir = task_dir / "episodes"
    success_dir = task_dir / "success"
    
    work_episode_dir = episodes_dir / episode_id
    
    # 1. 检查是否已成功
    if skip_successful:
        # 检查 success 目录下是否有该 episode
        if (success_dir / episode_id).exists():
            print(f"⏭️  已成功，跳过")
            return {"status": "skipped", "reason": "already_successful"}
    
    # 2. 如果设置了 skip_optimization，检查已有配置是否存在
    if skip_optimization:
        options_file = work_episode_dir / "options.json"
        if not options_file.exists():
            print(f"   ❌ 设置了 skip_optimization 但配置文件不存在: {options_file}")
            print(f"   💡 提示: 需要先运行一次优化生成配置，或手动创建配置文件")
            return {"status": "failed", "reason": "config_not_found"}
        
        print(f"   ✅ 使用已有配置: {options_file}")
        return {"status": "optimized", "episode_dir": str(work_episode_dir)}
    
    # 3. 复制原始配置到工作目录
    source_episode_dir = Path(config['base_dir']) / episode_id
    if not copy_episode_config(source_episode_dir, work_episode_dir):
        return {"status": "failed", "reason": "copy_config_failed"}
    
    # 4. 应用策略
    if not apply_strategy(episode_id, strategy, case, work_episode_dir, 
                         config['base_dir'], task_type):
        return {"status": "failed", "reason": "strategy_failed"}
    
    print(f"   ✅ 策略应用完成")
    return {"status": "optimized", "episode_dir": str(work_episode_dir)}


def run_single_inference(episode_id, work_episode_dir, results_dir, success_dir, config):
    """
    运行单个案例的推理，并实时保存结果
    
    Args:
        episode_id: Episode ID
        work_episode_dir: 工作目录（包含优化后的 options.json）
        results_dir: 结果保存目录
        success_dir: 成功案例目录
        config: 配置信息
    
    Returns:
        dict: 推理结果 {'status': 'success'/'failed', 'episode_id': ..., ...}
    """
    work_episode_dir = Path(work_episode_dir)
    results_dir = Path(results_dir)
    success_dir = Path(success_dir)
    
    # 结果保存到 results/<episode_id>/
    result_episode_dir = results_dir / episode_id
    
    # 检查是否已有结果（支持中断恢复）
    if result_episode_dir.exists() and (result_episode_dir / "log.json").exists():
        print(f"   ℹ️  已有推理结果，检查是否成功...")
        if check_success_from_log(result_episode_dir / "log.json"):
            print(f"   ✅ 已成功，跳过推理")
            # 确保成功案例已收集
            if not (success_dir / episode_id).exists():
                collect_success(result_episode_dir, success_dir, episode_id)
            return {
                'episode_id': episode_id,
                'status': 'success',
                'result_dir': str(result_episode_dir)
            }
        else:
            print(f"   ⚠️  之前推理失败，重新运行...")
            shutil.rmtree(result_episode_dir)
    
    result_episode_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n   🚀 运行推理...")
    
    # 1. 读取优化后的 options.json
    options_file = work_episode_dir / "options.json"
    if not options_file.exists():
        print(f"   ❌ 找不到 options.json: {options_file}")
        return {'episode_id': episode_id, 'status': 'failed', 'reason': 'config_not_found'}
    
    try:
        with open(options_file, 'r') as f:
            options = json.load(f)
    except Exception as e:
        print(f"   ❌ 读取 options.json 失败: {e}")
        return {'episode_id': episode_id, 'status': 'failed', 'reason': 'config_read_error'}
    
    # 2. 创建单个案例的数据集
    dataset = {
        episode_id: options,
        "seed": options.get("seed", 0),
        "num": 1
    }
    
    # 创建临时数据集文件
    task_type = config.get('task', 'unknown')
    temp_dataset_file = work_episode_dir / f"temp_{task_type}_{episode_id}.json"
    try:
        with open(temp_dataset_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"   📄 创建临时数据集: {temp_dataset_file.name}")
    except Exception as e:
        print(f"   ❌ 创建临时数据集失败: {e}")
        return {'episode_id': episode_id, 'status': 'failed', 'reason': 'dataset_create_error'}
    
    # 3. 运行推理
    model = config.get('model', 'openvla-7b')
    cmd = [
        VENV_PYTHON, OPENVLA_SCRIPT,
        "--data", str(temp_dataset_file),
        "--output", str(result_episode_dir.parent) + "/",  # 输出到 results/
        "--model", model
    ]
    
    if config.get('lora_path'):
        cmd.extend(["--lora_path", config['lora_path']])
    
    try:
        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONPATH'] = PROJECT_ROOT
        env['PYOPENGL_PLATFORM'] = 'egl'
        env['MUJOCO_GL'] = 'egl'
        env.pop('DISPLAY', None)
        env.pop('WAYLAND_DISPLAY', None)
        
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            env=env,
            timeout=600  # 10分钟超时
        )
        
        # 清理临时文件
        if temp_dataset_file.exists():
            temp_dataset_file.unlink()
        
        if result.returncode != 0:
            print(f"   ❌ 推理失败")
            if result.stderr:
                print(f"   错误: {result.stderr[:500]}")
            return {'episode_id': episode_id, 'status': 'failed', 'reason': 'inference_error'}
        
        # 4. 查找生成的 log.json
        # openVLA.py 会创建类似 results/temp_{task}_{episode_id}/model_seed/episode_id/log.json
        log_file = None
        
        # 先在直接路径查找
        direct_log = result_episode_dir / "log.json"
        if direct_log.exists():
            log_file = direct_log
        else:
            # 在临时目录中查找
            for subdir in result_episode_dir.parent.rglob(f"*/{episode_id}/log.json"):
                log_file = subdir
                # 将结果移动到正确位置
                source_dir = log_file.parent
                for item in source_dir.iterdir():
                    if item.is_file():
                        shutil.copy2(item, result_episode_dir / item.name)
                    elif item.is_dir() and item.name == "images":
                        target_images = result_episode_dir / "images"
                        if target_images.exists():
                            shutil.rmtree(target_images)
                        shutil.copytree(item, target_images)
                log_file = result_episode_dir / "log.json"
                break
        
        if not log_file or not log_file.exists():
            print(f"   ⚠️  推理完成但未找到 log.json")
            return {'episode_id': episode_id, 'status': 'failed', 'reason': 'log_not_found'}
        
        # 5. 检查是否成功
        is_successful = check_success_from_log(log_file)
        
        if is_successful:
            print(f"   ✅ 推理成功！")
            # 收集到成功目录
            collect_success(result_episode_dir, success_dir, episode_id)
            return {
                'episode_id': episode_id,
                'status': 'success',
                'result_dir': str(result_episode_dir)
            }
        else:
            print(f"   ❌ 推理失败（任务未成功）")
            return {'episode_id': episode_id, 'status': 'failed', 'reason': 'task_not_successful'}
        
    except subprocess.TimeoutExpired:
        if temp_dataset_file.exists():
            temp_dataset_file.unlink()
        print(f"   ⏱️  推理超时")
        return {'episode_id': episode_id, 'status': 'failed', 'reason': 'timeout'}
    except Exception as e:
        if temp_dataset_file.exists():
            temp_dataset_file.unlink()
        print(f"   ❌ 推理异常: {e}")
        return {'episode_id': episode_id, 'status': 'failed', 'reason': str(e)}


def run_all_inference(optimized_cases, config, task_type):
    """逐个运行所有优化案例的推理，支持中断恢复"""
    print(f"\n\n{'#'*70}")
    print(f"# 开始推理阶段（逐个执行，支持中断恢复）")
    print(f"{'#'*70}\n")
    
    task_dir = Path(f"optimization/{task_type}")
    results_dir = task_dir / "results"
    success_dir = task_dir / "success"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    success_dir.mkdir(parents=True, exist_ok=True)
    
    inference_results = []
    
    for idx, case_info in enumerate(optimized_cases, 1):
        episode_id = case_info['episode_id']
        work_episode_dir = Path(case_info['episode_dir'])
        
        print(f"\n{'='*70}")
        print(f"📋 推理案例 [{idx}/{len(optimized_cases)}]: Episode {episode_id}")
        print(f"{'='*70}")
        
        result = run_single_inference(
            episode_id=episode_id,
            work_episode_dir=work_episode_dir,
            results_dir=results_dir,
            success_dir=success_dir,
            config=config
        )
        
        inference_results.append(result)
        
        # 实时显示进度
        success_count = sum(1 for r in inference_results if r['status'] == 'success')
        print(f"\n   📊 当前进度: {idx}/{len(optimized_cases)}, 成功: {success_count}")
    
    # 清理临时目录
    for temp_dir in results_dir.glob("temp_*"):
        if temp_dir.is_dir():
            try:
                shutil.rmtree(temp_dir)
                print(f"   🗑️  清理临时目录: {temp_dir.name}")
            except:
                pass
    
    return inference_results


def diagnose_inference_stuck(task_type):
    """诊断推理是否卡住"""
    task_dir = Path(f"optimization/{task_type}")
    batch_dataset_file = task_dir / f"batch_{task_type}_dataset.json"
    
    print(f"\n🔍 诊断推理状态:")
    
    # 检查数据集文件
    if not batch_dataset_file.exists():
        print(f"   ❌ 数据集文件不存在: {batch_dataset_file}")
        return False
    
    # 检查日志文件
    log_dir = task_dir / "log"
    if log_dir.exists():
        log_files = list(log_dir.glob("batch_inference_*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            print(f"   📄 最新日志: {latest_log}")
            
            # 读取最后几行
            try:
                with open(latest_log, 'r') as f:
                    lines = f.readlines()
                    print(f"   日志行数: {len(lines)}")
                    
                    if lines:
                        print(f"\n   最后10行内容:")
                        for line in lines[-10:]:
                            print(f"      {line.strip()}")
            except Exception as e:
                print(f"   读取日志失败: {e}")
    
    # 检查结果目录
    results_dir = task_dir / "results" / "batch_inference"
    if results_dir.exists():
        subdirs = list(results_dir.iterdir())
        print(f"\n   结果目录内容 ({len(subdirs)} 项):")
        for subdir in subdirs:
            print(f"      {subdir.name}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="优化流程统一入口")
    parser.add_argument("task", type=str, 
                       help="任务类型 (如 move, grasp)")
    parser.add_argument("--no_skip", action="store_true",
                       help="不跳过已成功的案例")
    parser.add_argument("--episode", type=str,
                       help="只处理指定的 episode ID")
    parser.add_argument("--skip-optimization", action="store_true",
                       help="跳过优化阶段，直接运行推理")
    parser.add_argument("--diagnose", action="store_true",
                       help="诊断推理状态")
    
    args = parser.parse_args()
    
    if args.diagnose:
        diagnose_inference_stuck(args.task)
        return 0
    
    print(f"\n{'#'*70}")
    print(f"# 优化流程 - {args.task.upper()} 任务")
    print(f"{'#'*70}\n")
    
    # 1. 加载配置
    config = load_config(args.task)
    if not config:
        return 1
    
    print(f"📋 配置信息:")
    print(f"   基础目录: {config['base_dir']}")
    print(f"   任务类型: {config['task']}")
    print(f"   模型: {config.get('model', 'openvla-7b')}")
    print(f"   案例数量: {len(config['cases'])}")
    
    # 2. 获取已成功的案例
    success_dir = Path(f"optimization/{args.task}/success")
    successful_episodes = get_successful_episodes(success_dir)
    if successful_episodes:
        print(f"\n✅ 已成功案例: {len(successful_episodes)} 个")
        print(f"   {', '.join(sorted(successful_episodes))}")
    
    # 3. 过滤案例
    cases_to_process = config['cases']
    if args.episode:
        # 只处理指定的 episode
        cases_to_process = [c for c in cases_to_process if c['episode_id'] == args.episode]
        if not cases_to_process:
            print(f"\n❌ 未找到 episode {args.episode}")
            return 1
    
    # 4. 阶段1：应用所有策略（或跳过）
    optimized_cases = []
    optimization_results = {
        'total': len(cases_to_process),
        'optimized': 0,
        'failed': 0,
        'skipped': 0,
        'details': []
    }
    
    if args.skip_optimization:
        # 跳过优化，直接使用已有配置
        print(f"\n\n{'#'*70}")
        print(f"# 跳过优化阶段，使用已有配置")
        print(f"{'#'*70}\n")
        
        episodes_dir = Path(f"optimization/{args.task}/episodes")
        
        for idx, case in enumerate(cases_to_process, 1):
            episode_id = case['episode_id']
            work_episode_dir = episodes_dir / episode_id
            options_file = work_episode_dir / "options.json"
            
            print(f"\n[{idx}/{len(cases_to_process)}] Episode {episode_id}")
            
            # 检查是否已成功（如果不是 no_skip 模式）
            if not args.no_skip and (success_dir / episode_id).exists():
                print(f"   ⏭️  已成功，跳过")
                optimization_results['skipped'] += 1
                optimization_results['details'].append({
                    'episode_id': episode_id,
                    'status': 'skipped',
                    'reason': 'already_successful'
                })
                continue
            
            if not options_file.exists():
                print(f"   ❌ 配置文件不存在: {options_file}")
                print(f"   💡 提示: 需要先运行一次优化生成配置")
                optimization_results['failed'] += 1
                optimization_results['details'].append({
                    'episode_id': episode_id,
                    'status': 'failed',
                    'reason': 'config_not_found'
                })
                continue
            
            print(f"   ✅ 使用已有配置")
            optimization_results['optimized'] += 1
            result = {
                'episode_id': episode_id,
                'status': 'optimized',
                'episode_dir': str(work_episode_dir)
            }
            optimization_results['details'].append(result)
            optimized_cases.append(result)
    else:
        # 正常优化流程
        print(f"\n\n{'#'*70}")
        print(f"# 阶段1：应用优化策略")
        print(f"{'#'*70}\n")
        
        for idx, case in enumerate(cases_to_process, 1):
            print(f"\n\n{'#'*70}")
            print(f"# 优化案例 [{idx}/{len(cases_to_process)}]")
            print(f"{'#'*70}")
            
            result = process_case(case, config, args.task, skip_successful=not args.no_skip)
            
            result['episode_id'] = case['episode_id']
            optimization_results['details'].append(result)
            
            if result['status'] == 'optimized':
                optimization_results['optimized'] += 1
                optimized_cases.append(result)
            elif result['status'] == 'skipped':
                optimization_results['skipped'] += 1
            else:
                optimization_results['failed'] += 1
    
    # 打印优化阶段总结
    print(f"\n\n{'='*70}")
    print(f"📊 {'配置检查' if args.skip_optimization else '优化阶段'}完成")
    print(f"{'='*70}")
    print(f"总案例数: {optimization_results['total']}")
    print(f"✅ {'可推理' if args.skip_optimization else '已优化'}: {optimization_results['optimized']}")
    print(f"❌ 失败: {optimization_results['failed']}")
    print(f"⏭️  已跳过: {optimization_results['skipped']}")
    print(f"{'='*70}")
    
    # 5. 阶段2：统一推理
    if not optimized_cases:
        print(f"\n⚠️  没有需要推理的案例")
        
        # 保存报告
        report_file = Path(f"optimization/{args.task}/batch_report.json")
        optimization_results['timestamp'] = datetime.now().isoformat()
        optimization_results['run_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(report_file, 'w') as f:
            json.dump(optimization_results, f, indent=2)
        print(f"\n📄 报告已保存: {report_file}\n")
        return 1
    
    print(f"\n⏸️  准备开始推理，共 {len(optimized_cases)} 个案例")
    
    inference_results = run_all_inference(optimized_cases, config, args.task)
    
    # 6. 合并结果
    results = {
        'total': len(cases_to_process),
        'success': 0,
        'failed': 0,
        'skipped': optimization_results['skipped'],
        'details': []
    }
    
    # 合并优化和推理结果
    for opt_result in optimization_results['details']:
        episode_id = opt_result['episode_id']
        
        if opt_result['status'] == 'skipped':
            results['details'].append(opt_result)
            continue
        elif opt_result['status'] == 'failed':
            results['failed'] += 1
            results['details'].append(opt_result)
            continue
        
        # 查找对应的推理结果
        inf_result = next((r for r in inference_results if r['episode_id'] == episode_id), None)
        
        if inf_result and inf_result['status'] == 'success':
            results['success'] += 1
            results['details'].append(inf_result)
        else:
            results['failed'] += 1
            results['details'].append(inf_result or opt_result)
    
    # 5. 总结报告
    print(f"\n\n{'='*70}")
    print(f"📊 处理完成")
    print(f"{'='*70}")
    print(f"总案例数: {results['total']}")
    print(f"✅ 成功: {results['success']}")
    print(f"❌ 失败: {results['failed']}")
    print(f"⏭️  跳过: {results['skipped']}")
    print(f"{'='*70}")
    
    # 保存报告（固定文件名，在内容中添加时间戳）
    report_file = Path(f"optimization/{args.task}/batch_report.json")
    results['timestamp'] = datetime.now().isoformat()
    results['run_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 报告已保存: {report_file}")
    print(f"⏰ 运行时间: {results['run_time']}\n")
    
    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    exit(main())
