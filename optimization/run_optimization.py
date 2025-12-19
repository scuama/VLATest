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
    # 结构: success/<model_tag>_<seed>/<episode_id>/
    successful = set()
    for model_dir in success_dir.iterdir():
        if model_dir.is_dir():
            # 遍历该模型目录下的所有 episode
            for episode_dir in model_dir.iterdir():
                if episode_dir.is_dir():
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


def collect_success(source_file_dir, success_base_dir, model_tag, episode_id, seed):
    """
    收集成功案例到标准结构
    
    结构: success_base_dir/openvla-7b_<seed>/<episode_id>/
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
        model_tag: 模型标签 (如 openvla-7b)
        episode_id: Episode ID（保持原始编号，如 "2", "7", "44"）
        seed: 随机种子
    """
    source_file_dir = Path(source_file_dir)
    success_base_dir = Path(success_base_dir)
    
    # 创建标准结构目录: success/openvla-7b_<seed>/<episode_id>/
    model_dir_name = f"{model_tag}_{seed}"
    target_dir = success_base_dir / model_dir_name / episode_id
    
    # 如果目录已存在，先删除
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n   📦 收集成功案例 Episode {episode_id}...")
    
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
    
    print(f"   ✅ 成功案例已收集: {target_dir}")
    print(f"      文件数: {files_copied}")
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
        # 检查所有模型目录下是否有该 episode
        for model_dir in success_dir.iterdir():
            if model_dir.is_dir() and (model_dir / episode_id).exists():
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


def run_all_inference(optimized_cases, config, task_type):
    """统一运行所有优化案例的推理 - 整合成一个大数据集"""
    print(f"\n\n{'#'*70}")
    print(f"# 开始统一推理阶段")
    print(f"{'#'*70}\n")
    
    task_dir = Path(f"optimization/{task_type}")
    results_dir = task_dir / "results"
    success_dir = task_dir / "success"
    
    # 获取 model 和 seed 信息（稍后用于收集成功案例）
    model_tag = config.get('model', 'openvla-7b')
    
    # 1. 整合所有场景到一个数据集
    print(f"📦 整合数据集...")
    
    batch_dataset = {
        "num": len(optimized_cases)
    }
    
    episode_list = []  # 记录处理顺序，用于后续映射
    
    for case_info in optimized_cases:
        episode_id = case_info['episode_id']
        work_episode_dir = Path(case_info['episode_dir'])
        
        # 读取优化后的 options.json
        options_file = work_episode_dir / "options.json"
        if not options_file.exists():
            print(f"   ⚠️  Episode {episode_id}: 找不到 options.json")
            continue
        
        try:
            with open(options_file, 'r') as f:
                options = json.load(f)
            
            # 应用 max_episode_steps（如果在配置中指定了）
            if 'max_episode_steps' in case_info:
                options['max_episode_steps'] = case_info['max_episode_steps']
                print(f"   🔧 Episode {episode_id}: 设置最大步数 = {case_info['max_episode_steps']}")
            
            # 使用原始 episode_id 作为键（保持一致性）
            batch_dataset[episode_id] = options
            
            # 如果有 seed，使用第一个场景的 seed
            if "seed" not in batch_dataset and "seed" in options:
                batch_dataset["seed"] = options["seed"]
            
            episode_list.append(episode_id)
            
            # 显示源物体信息（仅用于日志）
            model_ids = options.get("model_ids", [])
            source_obj_id = options.get("source_obj_id", 0)
            if isinstance(model_ids, list) and 0 <= source_obj_id < len(model_ids):
                source_model = model_ids[source_obj_id]
            else:
                source_model = "unknown"
            print(f"   ✅ Episode {episode_id}: {source_model}")
            
        except Exception as e:
            print(f"   ❌ Episode {episode_id}: 解析失败 - {e}")
            continue
    
    # 更新实际数量
    batch_dataset["num"] = len(episode_list)
    
    # 设置默认 seed
    if "seed" not in batch_dataset:
        batch_dataset["seed"] = 0
    
    # 2. 保存批量数据集（文件名必须包含任务类型关键词以便 openVLA.py 推断任务）
    batch_dataset_file = task_dir / f"batch_{task_type}_dataset.json"
    try:
        with open(batch_dataset_file, 'w') as f:
            json.dump(batch_dataset, f, indent=2)
        print(f"\n📄 批量数据集已保存: {batch_dataset_file}")
        print(f"   包含 {len(episode_list)} 个场景: {', '.join(episode_list)}")
    except Exception as e:
        print(f"❌ 保存批量数据集失败: {e}")
        return []
    
    # 3. 运行批量推理
    print(f"\n🚀 开始批量推理...")
    
    batch_result_dir = results_dir / "batch_inference"
    batch_result_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建批量推理日志文件
    batch_inference_log = task_dir / "log" / f"batch_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    batch_inference_log.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        VENV_PYTHON, OPENVLA_SCRIPT,
        "--data", str(batch_dataset_file),
        "--output", str(batch_result_dir) + "/",  # 确保以 / 结尾
        "--model", config.get('model', 'openvla-7b')
    ]
    
    # 不使用 resume 模式，确保重新运行
    # (默认 action='store_true' 不传参数就是 False)
    
    if config.get('lora_path'):
        cmd.extend(["--lora_path", config['lora_path']])
    
    try:
        # 设置环境变量，确保能找到 simpler_env 模块
        env = os.environ.copy()
        env['PYTHONPATH'] = PROJECT_ROOT
        
        # 打开日志文件用于实时写入
        log_file = open(batch_inference_log, 'w', buffering=1)  # 行缓冲
        
        # 写入头部信息
        log_file.write(f"批量推理命令: {' '.join(cmd)}\n")
        log_file.write(f"Python 解释器: {VENV_PYTHON}\n")
        log_file.write(f"环境变量 PYTHONPATH: {env['PYTHONPATH']}\n")
        log_file.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("="*70 + "\n\n")
        log_file.flush()
        
        # 使用 Popen 实现实时输出
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 合并 stderr 到 stdout
            text=True,
            bufsize=1,  # 行缓冲
            env=env
        )
        
        # 实时读取并写入日志
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
        
        # 等待进程结束
        process.wait()
        
        # 写入尾部信息
        log_file.write("\n" + "="*70 + "\n")
        log_file.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"返回码: {process.returncode}\n")
        log_file.close()
        
        print(f"   📄 推理日志已保存: {batch_inference_log}")
        
        if process.returncode != 0:
            print(f"❌ 批量推理失败（返回码: {process.returncode}）")
            print(f"   详细日志: {batch_inference_log}")
            return []
        
        print(f"✅ 批量推理完成")
        
    except Exception as e:
        print(f"❌ 批量推理异常: {e}")
        # 保存异常信息
        try:
            with open(batch_inference_log, 'a') as f:
                f.write(f"\n异常信息: {str(e)}\n")
        except:
            pass
        return []
    
    # 4. 处理每个场景的推理结果
    print(f"\n📊 处理推理结果...")
    
    inference_results = []
    
    for episode_id in episode_list:
        print(f"\n{'='*70}")
        print(f"📋 Episode {episode_id}")
        print(f"{'='*70}")
        
        # 查找对应的 log.json
        # openVLA.py 生成的结构: batch_result_dir/batch_{task_type}_dataset/openvla-7b_<seed>/{episode_id}/log.json
        log_file = None
        
        # 直接查找包含该 episode_id 的路径
        for subdir in batch_result_dir.rglob(f"*/{episode_id}/log.json"):
            log_file = subdir
            break
        
        # 如果失败，尝试更宽松的查找
        if not log_file:
            for subdir in batch_result_dir.rglob("log.json"):
                # 检查路径中是否包含正确的 episode_id
                if f"/{episode_id}/" in str(subdir):
                    log_file = subdir
                    break
        
        if not log_file or not log_file.exists():
            print(f"   ⚠️  未找到 log.json")
            inference_results.append({
                'episode_id': episode_id,
                'status': 'failed',
                'reason': 'log_not_found'
            })
            continue
        
        print(f"   📄 日志文件: {log_file}")
        
        # 检查是否成功
        is_successful = check_success_from_log(log_file)
        
        if is_successful:
            print(f"   ✅ 推理成功！")
            
            # 复制结果到对应 episode 目录
            result_episode_dir = results_dir / episode_id
            result_episode_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制所有结果文件
            import shutil
            source_result_dir = log_file.parent
            
            # 复制 log.json, actions.npy, actions.json, options.json
            for filename in ["log.json", "actions.npy", "actions.json", "options.json"]:
                src_file = source_result_dir / filename
                if src_file.exists():
                    shutil.copy2(src_file, result_episode_dir / filename)
            
            # 复制 images 目录（如果存在）
            images_dir = source_result_dir / "images"
            if images_dir.exists():
                target_images_dir = result_episode_dir / "images"
                if target_images_dir.exists():
                    shutil.rmtree(target_images_dir)
                shutil.copytree(images_dir, target_images_dir)
            
            # 收集成功案例到标准结构
            seed = batch_dataset.get("seed", 0)
            collect_success(result_episode_dir, success_dir, model_tag, episode_id, seed)
            
            inference_results.append({
                'episode_id': episode_id,
                'status': 'success',
                'result_dir': str(result_episode_dir)
            })
        else:
            print(f"   ❌ 推理失败")
            inference_results.append({
                'episode_id': episode_id,
                'status': 'failed',
                'reason': 'inference_not_successful'
            })
    
    return inference_results


def main():
    parser = argparse.ArgumentParser(description="优化流程统一入口")
    parser.add_argument("task", type=str, 
                       help="任务类型 (如 move, grasp)")
    parser.add_argument("--no_skip", action="store_true",
                       help="不跳过已成功的案例")
    parser.add_argument("--episode", type=str,
                       help="只处理指定的 episode ID")
    
    args = parser.parse_args()
    
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
    
    # 4. 阶段1：应用所有策略
    print(f"\n\n{'#'*70}")
    print(f"# 阶段1：应用优化策略")
    print(f"{'#'*70}\n")
    
    optimized_cases = []
    optimization_results = {
        'total': len(cases_to_process),
        'optimized': 0,
        'failed': 0,
        'skipped': 0,
        'details': []
    }
    
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
    print(f"📊 优化阶段完成")
    print(f"{'='*70}")
    print(f"总案例数: {optimization_results['total']}")
    print(f"✅ 已优化: {optimization_results['optimized']}")
    print(f"❌ 优化失败: {optimization_results['failed']}")
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
