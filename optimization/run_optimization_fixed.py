# optimization/run_optimization_fixed.py
#!/usr/bin/env python3
"""
优化流程统一入口 - 修复版
修复推理卡住问题，支持多种渲染器
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
import threading
import time
import signal
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

# 🎯 修复：渲染器配置（按优先级尝试）
RENDERER_CONFIGS = [
    {
        "name": "OSMesa",
        "env": {"PYOPENGL_PLATFORM": "osmesa", "MUJOCO_GL": "osmesa"},
        "description": "软件渲染，不需要GPU，最可靠"
    },
    {
        "name": "默认",
        "env": {},
        "description": "系统默认渲染器（如果有显示器）"
    },
    {
        "name": "EGL",
        "env": {"PYOPENGL_PLATFORM": "egl", "MUJOCO_GL": "egl"},
        "description": "GPU渲染，需要NVIDIA EGL支持"
    }
]

# 🎯 修复：推理超时设置（秒）
INFERENCE_TIMEOUT = 600  # 10分钟
MAX_RETRIES = 2  # 最大重试次数


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
    
    # 🎯 修复：验证任务名
    valid_tasks = {
        'grasp': ['google_robot_pick_customizable', 'google_robot_pick_customizable_ycb'],
        'move': ['google_robot_move_near_customizable', 'google_robot_move_near_customizable_ycb']
    }
    
    current_task = config.get('task', '')
    expected_tasks = valid_tasks.get(task_type, [])
    
    if expected_tasks and current_task not in expected_tasks:
        print(f"⚠️  警告: 任务名 '{current_task}' 可能不正确")
        if expected_tasks:
            print(f"   对于 {task_type} 任务，建议使用: {expected_tasks[0]}")
            print(f"   当前配置将继续使用: {current_task}")
    
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


# 🎯 修复：改进的推理运行函数
def run_inference_with_monitoring(cmd, env, log_file_path, timeout, renderer_name="未知"):
    """运行推理进程，带监控和超时控制"""
    
    print(f"   渲染器: {renderer_name}")
    print(f"   超时: {timeout}秒")
    print(f"   日志: {log_file_path}")
    
    # 打开日志文件
    log_file = open(log_file_path, 'w', buffering=1)
    
    # 写入头部信息
    log_file.write(f"推理命令: {' '.join(cmd)}\n")
    log_file.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"渲染器: {renderer_name}\n")
    log_file.write(f"超时: {timeout}秒\n")
    log_file.write("="*70 + "\n\n")
    log_file.flush()
    
    process_start_time = time.time()
    
    try:
        # 启动进程
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        
        print(f"   进程PID: {process.pid}")
        
        # 创建监控线程
        stop_monitoring = threading.Event()
        
        def monitor_process():
            """监控进程状态"""
            last_activity = time.time()
            consecutive_no_output = 0
            
            while not stop_monitoring.is_set():
                if process.poll() is not None:
                    break
                    
                elapsed = time.time() - process_start_time
                
                # 检查超时
                if elapsed > timeout:
                    timeout_msg = f"[超时] 运行时间: {elapsed:.0f}秒 > {timeout}秒"
                    print(f"   ⏰ {timeout_msg}")
                    log_file.write(f"\n{timeout_msg}\n")
                    log_file.flush()
                    
                    try:
                        print(f"   终止进程 {process.pid}...")
                        process.terminate()
                        time.sleep(2)
                        
                        if process.poll() is None:
                            print(f"   强制终止进程 {process.pid}...")
                            process.kill()
                    except Exception as e:
                        print(f"   终止进程失败: {e}")
                    
                    break
                
                # 每30秒输出状态
                if int(elapsed) % 30 == 0:
                    status = f"[运行中] {int(elapsed)}秒 | PID: {process.pid}"
                    print(f"   ⏳ {status}")
                    log_file.write(f"\n[状态] {status}\n")
                    log_file.flush()
                
                time.sleep(5)  # 每5秒检查一次
        
        # 启动监控线程
        monitor_thread = threading.Thread(target=monitor_process, daemon=True)
        monitor_thread.start()
        
        # 读取输出
        line_count = 0
        last_progress_time = time.time()
        
        print(f"   开始读取输出...")
        
        for line in process.stdout:
            line = line.rstrip()
            line_count += 1
            
            # 写入日志
            log_file.write(line + "\n")
            log_file.flush()
            
            # 显示关键信息
            elapsed = time.time() - process_start_time
            
            # 实时显示重要输出
            if any(keyword in line.lower() for keyword in 
                  ['error', 'exception', 'traceback', 'failed']):
                print(f"   🔴 [{elapsed:6.1f}s] {line[:100]}")
            elif any(keyword in line.lower() for keyword in 
                    ['start', 'go', 'step', 'success']):
                print(f"   🟢 [{elapsed:6.1f}s] {line[:100]}")
            elif 'loading' in line.lower() or 'progress' in line.lower():
                print(f"   🔵 [{elapsed:6.1f}s] {line[:100]}")
            elif line_count % 20 == 0:
                print(f"   📝 [{elapsed:6.1f}s] {line[:80]}...")
            
            # 每30秒显示统计
            current_time = time.time()
            if current_time - last_progress_time > 30:
                print(f"   📊 已处理 {line_count} 行 | 运行: {int(elapsed)}秒")
                last_progress_time = current_time
        
        # 停止监控
        stop_monitoring.set()
        monitor_thread.join(timeout=5)
        
        # 等待进程结束
        return_code = process.wait()
        elapsed_total = time.time() - process_start_time
        
        # 写入尾部信息
        log_file.write("\n" + "="*70 + "\n")
        log_file.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"总运行时间: {elapsed_total:.1f}秒\n")
        log_file.write(f"总输出行数: {line_count}\n")
        log_file.write(f"返回码: {return_code}\n")
        log_file.close()
        
        print(f"\n   📊 推理结束:")
        print(f"     运行时间: {elapsed_total:.1f}秒")
        print(f"     输出行数: {line_count}")
        print(f"     返回码: {return_code}")
        
        # 分析结果
        if line_count < 50:
            print(f"   ⚠️  输出过少 ({line_count}行)，可能环境初始化失败")
        
        return return_code, line_count, elapsed_total
        
    except Exception as e:
        print(f"❌ 推理执行异常: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            log_file.write(f"\n异常信息: {str(e)}\n")
            log_file.write(traceback.format_exc())
            log_file.close()
        except:
            pass
        
        return 1, 0, 0


# 🎯 修复：改进的统一推理函数
def run_all_inference(optimized_cases, config, task_type, inference_timeout=INFERENCE_TIMEOUT):
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
    
    # 3. 运行批量推理（带重试和多种渲染器）
    print(f"\n🚀 开始批量推理...")
    
    batch_result_dir = results_dir / "batch_inference"
    batch_result_dir.mkdir(parents=True, exist_ok=True)
    
    # 尝试不同的渲染器配置
    successful = False
    inference_log = None
    
    for retry in range(MAX_RETRIES + 1):
        renderer_config = RENDERER_CONFIGS[min(retry, len(RENDERER_CONFIGS)-1)]
        
        print(f"\n{'='*70}")
        print(f"🔄 尝试 {retry + 1}/{MAX_RETRIES + 1}: {renderer_config['name']}")
        print(f"   描述: {renderer_config['description']}")
        print(f"{'='*70}")
        
        # 创建日志文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_inference_log = task_dir / "log" / f"batch_inference_{timestamp}_{renderer_config['name'].lower()}.log"
        batch_inference_log.parent.mkdir(parents=True, exist_ok=True)
        inference_log = batch_inference_log
        
        # 构建命令
        cmd = [
            VENV_PYTHON, OPENVLA_SCRIPT,
            "--data", str(batch_dataset_file),
            "--output", str(batch_result_dir) + "/",
            "--model", config.get('model', 'openvla-7b')
        ]
        
        if config.get('lora_path'):
            cmd.extend(["--lora_path", config['lora_path']])
        
        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONPATH'] = PROJECT_ROOT
        env.update(renderer_config['env'])
        
        # 清除可能干扰的变量
        env.pop('DISPLAY', None)
        env.pop('WAYLAND_DISPLAY', None)
        
        # 运行推理
        return_code, line_count, elapsed = run_inference_with_monitoring(
            cmd, env, batch_inference_log, inference_timeout, renderer_config['name']
        )
        
        if return_code == 0:
            print(f"   ✅ {renderer_config['name']} 渲染器成功！")
            successful = True
            break
        else:
            print(f"   ❌ {renderer_config['name']} 渲染器失败")
            
            if retry < MAX_RETRIES:
                print(f"   🔄 准备尝试下一个渲染器...")
                time.sleep(3)  # 等待3秒再重试
            else:
                print(f"   ⛔ 所有渲染器都失败")
                
                # 提供诊断建议
                print(f"\n🔍 诊断建议:")
                print(f"   1. 检查GPU内存: nvidia-smi")
                print(f"   2. 检查渲染器支持:")
                print(f"      python3 -c \"from OpenGL import OSMesa; print('OSMesa支持')\"")
                print(f"   3. 检查数据集: {batch_dataset_file}")
                print(f"   4. 查看详细日志: {inference_log}")
    
    if not successful:
        print(f"\n❌ 所有推理尝试都失败")
        return []
    
    print(f"✅ 批量推理完成")
    
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


def collect_success(source_file_dir, success_base_dir, model_tag, episode_id, seed):
    """
    收集成功案例到标准结构
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
    parser = argparse.ArgumentParser(description="优化流程统一入口 - 修复版")
    parser.add_argument("task", type=str, 
                       help="任务类型 (如 move, grasp)")
    parser.add_argument("--no_skip", action="store_true",
                       help="不跳过已成功的案例")
    parser.add_argument("--episode", type=str,
                       help="只处理指定的 episode ID")
    parser.add_argument("--diagnose", action="store_true",
                       help="诊断推理状态")
    parser.add_argument("--force_osmesa", action="store_true",
                       help="强制使用OSMesa渲染器")
    parser.add_argument("--timeout", type=int, default=INFERENCE_TIMEOUT,
                       help=f"推理超时时间(秒)，默认: {INFERENCE_TIMEOUT}")
    
    args = parser.parse_args()
    
    # 如果强制使用OSMesa，调整渲染器配置
    current_renderer_configs = RENDERER_CONFIGS.copy()
    if args.force_osmesa:
        current_renderer_configs = [RENDERER_CONFIGS[0]]  # 只保留OSMesa
        print(f"🔧 强制使用OSMesa渲染器")
    
    if args.diagnose:
        diagnose_inference_stuck(args.task)
        return 0
    
    print(f"\n{'#'*70}")
    print(f"# 优化流程 - {args.task.upper()} 任务 (修复版)")
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
    print(f"   推理超时: {args.timeout}秒")
    
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
        print(f"{'='*70}")
        
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
    
    # 🎯 修复：传递当前渲染器配置
    global RENDERER_CONFIGS
    original_renderer_configs = RENDERER_CONFIGS
    RENDERER_CONFIGS = current_renderer_configs
    
    try:
        inference_results = run_all_inference(optimized_cases, config, args.task, args.timeout)
    finally:
        RENDERER_CONFIGS = original_renderer_configs
    
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
    
    # 7. 总结报告
    print(f"\n\n{'='*70}")
    print(f"📊 处理完成")
    print(f"{'='*70}")
    print(f"总案例数: {results['total']}")
    print(f"✅ 成功: {results['success']}")
    print(f"❌ 失败: {results['failed']}")
    print(f"⏭️  跳过: {results['skipped']}")
    print(f"{'='*70}")
    
    # 保存报告
    report_file = Path(f"optimization/{args.task}/batch_report.json")
    results['timestamp'] = datetime.now().isoformat()
    results['run_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results['inference_timeout'] = args.timeout
    results['renderer_used'] = current_renderer_configs[0]['name'] if args.force_osmesa else "自动选择"
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 报告已保存: {report_file}")
    print(f"⏰ 运行时间: {results['run_time']}\n")
    
    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    exit(main())