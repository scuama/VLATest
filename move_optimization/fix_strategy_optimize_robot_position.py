#!/usr/bin/env python3
"""
通过优化机械臂初始位置实现抓取成功（支持批量处理）
适配简化版 options.json 结构（直接包含 obj_init_options.init_xy）
"""

import os
import json
import random
import subprocess
import argparse
import numpy as np
from transforms3d.euler import euler2quat
import sapien.core as sapien


# ==================== 默认配置参数 ====================
# 根据你的实际路径修改
DEFAULT_BASE_DIR = "/mnt/disk1/decom/VLATest/results/t-move_n-100_o-0_s-3225323079/openvla-7b_2024"
DEFAULT_TASK_TYPE = "move"
REPLAY_SCRIPT = "/mnt/disk1/decom/VLATest/experiments/replay_openvla_actions.py"
OPTIMIZED_CONFIGS_DIR = "/mnt/disk1/decom/VLATest/move_optimization/optimized_configs"

# 成功判断标准
MIN_CONSECUTIVE_GRASP_STEPS = 5

# 搜索参数
DEFAULT_TOTAL_ATTEMPTS = 20
COARSE_FINE_RATIO = 0.5

# 确保优化配置目录存在
os.makedirs(OPTIMIZED_CONFIGS_DIR, exist_ok=True)

# 机械臂方向定义（坐标系：+y向上，-y向下，-x向左，+x向右）
# 粗搜索范围：x和y轴统一控制在±0.05米以内
DIRECTION_OFFSETS = {
    "left": {"x": (-0.05, -0.01), "y": (-0.05, 0.05)},
    "right": {"x": (0.01, 0.05), "y": (-0.05, 0.05)},
    "up": {"x": (-0.05, 0.05), "y": (0.01, 0.05)},
    "down": {"x": (-0.05, 0.05), "y": (-0.05, -0.01)},
    "left-up": {"x": (-0.05, -0.01), "y": (0.01, 0.05)},
    "left-down": {"x": (-0.05, -0.01), "y": (-0.05, -0.01)},
    "right-up": {"x": (0.01, 0.05), "y": (0.01, 0.05)},
    "right-down": {"x": (0.01, 0.05), "y": (-0.05, -0.01)},
}

# 精细搜索范围（也略微增大）
FINE_SEARCH_RANGE = {"x": (-0.01, 0.01), "y": (-0.01, 0.01)}
ROBOT_X_RANGE = (-0.3, 0.3)
ROBOT_Y_RANGE = (-0.3, 0.3)


# ==================== 工具函数 ====================

def get_safe_default_rot_quat():
    """
    获取安全的默认旋转四元数
    与 ManiSkill2 prepackaged_config 完全一致
    确保相机姿态正常（朝向前下方）
    """
    quat = (sapien.Pose(q=euler2quat(0, 0, -0.09)) * sapien.Pose(q=[0, 0, 0, 1])).q
    return quat


def copy_episode_to_optimized_dir(episode_id, original_base_dir):
    """将episode配置复制到优化目录"""
    import shutil
    
    original_episode_dir = os.path.join(original_base_dir, episode_id)
    optimized_episode_dir = os.path.join(OPTIMIZED_CONFIGS_DIR, episode_id)
    
    # 如果优化目录已存在，先删除
    if os.path.exists(optimized_episode_dir):
        shutil.rmtree(optimized_episode_dir)
    
    # 创建目录
    os.makedirs(optimized_episode_dir, exist_ok=True)
    
    # 复制必要文件（支持多种格式）
    files_to_copy = ["options.json", "log.json"]
    action_files = ["actions.pkl", "actions.npy", "actions.json"]
    
    for filename in files_to_copy:
        src = os.path.join(original_episode_dir, filename)
        dst = os.path.join(optimized_episode_dir, filename)
        if os.path.exists(src):
            shutil.copy(src, dst)
    
    # 复制 actions 文件（可能有不同格式）
    for action_file in action_files:
        src = os.path.join(original_episode_dir, action_file)
        if os.path.exists(src):
            dst = os.path.join(optimized_episode_dir, action_file)
            shutil.copy(src, dst)
    
    print(f"✅ 复制episode配置: {original_episode_dir} -> {optimized_episode_dir}")
    return optimized_episode_dir


def load_options(episode_dir):
    """加载 options.json"""
    options_path = os.path.join(episode_dir, "options.json")
    try:
        with open(options_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 加载配置失败 {options_path}: {e}")
        return None


def save_options(episode_dir, options):
    """保存 options.json"""
    options_path = os.path.join(episode_dir, "options.json")
    with open(options_path, 'w') as f:
        json.dump(options, f, indent=2)


def backup_original_options(episode_dir):
    """备份原始 options.json"""
    options_path = os.path.join(episode_dir, "options.json")
    backup_path = os.path.join(episode_dir, "options.json.original_backup")
    if not os.path.exists(backup_path):
        try:
            import shutil
            shutil.copy(options_path, backup_path)
            return True
        except Exception as e:
            print(f"⚠️  备份失败: {e}")
    return False


def initialize_robot_options(options, default_xy=[0.0, 0.0], force_safe_quat=True):
    """
    初始化机械臂位置配置（安全版本）
    
    Args:
        options: 配置字典
        default_xy: 默认XY坐标
        force_safe_quat: 是否强制使用安全的旋转四元数（默认True，确保相机姿态正常）
        
    Returns:
        更新后的配置
    """
    if "robot_init_options" not in options:
        options["robot_init_options"] = {}
    
    # 设置XY位置
    if "init_xy" not in options["robot_init_options"]:
        options["robot_init_options"]["init_xy"] = default_xy
    
    # 🔑 关键改进：强制使用安全的旋转四元数，确保相机姿态正常
    if force_safe_quat:
        safe_quat = get_safe_default_rot_quat()
        options["robot_init_options"]["init_rot_quat"] = safe_quat.tolist()
        # 检测是否覆盖了错误的单位四元数
        if "init_rot_quat" in options.get("robot_init_options", {}):
            existing_quat = options["robot_init_options"]["init_rot_quat"]
            if np.allclose(existing_quat, [1, 0, 0, 0], atol=1e-3):
                print(f"   ⚠️  检测到错误的单位四元数，已自动修正为安全值")
    elif "init_rot_quat" not in options["robot_init_options"]:
        # 如果不强制但四元数不存在，也使用安全值
        safe_quat = get_safe_default_rot_quat()
        options["robot_init_options"]["init_rot_quat"] = safe_quat.tolist()
    
    return options


def adjust_robot_position(original_xy, x_range, y_range):
    """调整机械臂位置"""
    x_offset = random.uniform(x_range[0], x_range[1])
    y_offset = random.uniform(y_range[0], y_range[1])
    
    new_x = original_xy[0] + x_offset
    new_y = original_xy[1] + y_offset
    
    # 确保在机械臂安全范围内
    new_x = np.clip(new_x, ROBOT_X_RANGE[0], ROBOT_X_RANGE[1])
    new_y = np.clip(new_y, ROBOT_Y_RANGE[0], ROBOT_Y_RANGE[1])
    
    return [float(new_x), float(new_y)]


def modify_robot_position(options, x_range, y_range):
    """修改机械臂初始位置"""
    options = initialize_robot_options(options, force_safe_quat=True)  # 确保使用安全四元数
    original_xy = options["robot_init_options"]["init_xy"].copy()
    new_xy = adjust_robot_position(original_xy, x_range, y_range)
    options["robot_init_options"]["init_xy"] = new_xy
    return new_xy, original_xy


def run_replay_subprocess(episode_dir, task_name):
    """使用子进程重放（避免环境崩溃影响主进程）"""
    try:
        cmd = [
            'python3', REPLAY_SCRIPT,
            '--episode_dir', episode_dir,
            '--task', task_name
            # 不指定 render_every，使用默认值 1（每步都渲染）
        ]
        
        env_vars = os.environ.copy()
        env_vars['MUJOCO_GL'] = 'egl'  # 强制使用EGL渲染
        env_vars['PYOPENGL_PLATFORM'] = 'egl'  # 确保使用EGL
        # 不设置DISPLAY，避免X11相关错误
        if 'DISPLAY' in env_vars:
            del env_vars['DISPLAY']
        
        result = subprocess.run(
            cmd,
            env=env_vars,
            capture_output=True,
            text=True,
            timeout=60  # 60秒超时
        )
        
        # 检查是否生成了replay_log.json
        replay_log_path = os.path.join(episode_dir, 'replay_log.json')
        if os.path.exists(replay_log_path):
            return True
        
        # 输出详细错误信息
        if result.returncode != 0:
            print(f"\n   ⚠️  重放脚本返回错误码: {result.returncode}")
            if result.stderr:
                print(f"   错误输出: {result.stderr[:300]}")
            if result.stdout:
                print(f"   标准输出: {result.stdout[:200]}")
        
        return False
        
    except subprocess.TimeoutExpired:
        print("⏱️ 超时")
        return False
    except Exception as e:
        print(f"❌ 异常: {str(e)[:100]}")
        return False


def run_replay_direct(episode_dir, task_name):
    """直接在当前进程中重放（避免子进程段错误）"""
    try:
        import simpler_env
        
        # 加载 actions
        actions_npy = os.path.join(episode_dir, 'actions.npy')
        actions_json = os.path.join(episode_dir, 'actions.json')
        
        if os.path.exists(actions_npy):
            actions = np.load(actions_npy, allow_pickle=True)
            actions = [None if a is None else np.array(a, dtype=np.float64) for a in actions]
        elif os.path.exists(actions_json):
            with open(actions_json, 'r') as f:
                actions = json.load(f)
            actions = [None if a is None else np.array(a, dtype=np.float64) for a in actions]
        else:
            return False
        
        # 加载 options
        options_path = os.path.join(episode_dir, 'options.json')
        if os.path.exists(options_path):
            with open(options_path, 'r') as f:
                options = json.load(f)
        else:
            options = {}
        
        # 创建环境并重放
        env = simpler_env.make(task_name)
        obs, info = env.reset(seed=options.get('seed'), options=options)
        
        replay_log = {}
        for t, action in enumerate(actions):
            if action is None:
                continue
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 检查抓取状态（move 任务）
            if hasattr(env.unwrapped, 'episode_source_obj'):
                source_obj = env.unwrapped.episode_source_obj
                is_grasped = env.unwrapped.agent.check_grasp(source_obj)
                info['is_src_obj_grasped'] = is_grasped
            
            replay_log[t] = info
            
            if terminated:
                break
        
        # 保存 replay_log
        replay_log_path = os.path.join(episode_dir, 'replay_log.json')
        with open(replay_log_path, 'w') as f:
            # 转换 numpy 类型为 Python 原生类型
            def convert_value(v):
                if isinstance(v, np.ndarray):
                    return v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    return v.item()
                elif isinstance(v, np.bool_):
                    return bool(v)
                elif isinstance(v, dict):
                    return {k: convert_value(val) for k, val in v.items()}
                elif isinstance(v, list):
                    return [convert_value(item) for item in v]
                return v
            
            clean_log = {k: convert_value(v) for k, v in replay_log.items()}
            json.dump(clean_log, f, indent=2)
        
        env.close()
        return True
        
    except Exception as e:
        print(f"   重放异常: {str(e)[:60]}")
        return False


def check_grasp_success(episode_dir, task_type, min_steps=MIN_CONSECUTIVE_GRASP_STEPS):
    """检查是否成功抓取"""
    replay_log_path = os.path.join(episode_dir, "replay_log.json")
    log_path = os.path.join(episode_dir, "log.json")
    
    log_to_read = replay_log_path if os.path.exists(replay_log_path) else log_path
    
    if not os.path.exists(log_to_read):
        return False, 0, {}
    
    try:
        with open(log_to_read, 'r') as f:
            log_data = json.load(f)
        
        def to_bool(value):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() == "true"
            return False
        
        if task_type == "grasp":
            for step_key, step_info in log_data.items():
                if isinstance(step_info, dict):
                    is_grasped = to_bool(step_info.get("is_grasped", False))
                    lifted = to_bool(step_info.get("lifted_object", False))
                    if is_grasped or lifted:
                        return True, 1, {
                            "step": step_key,
                            "is_grasped": is_grasped,
                            "lifted_object": lifted,
                        }
            return False, 0, {}
        
        elif task_type == "move":
            grasp_steps = []
            sorted_steps = sorted([(int(k), v) for k, v in log_data.items() 
                                  if isinstance(v, dict)], key=lambda x: x[0])
            
            for step_num, step_info in sorted_steps:
                is_grasped = step_info.get("is_src_obj_grasped")
                if is_grasped is True:
                    grasp_steps.append(step_num)
            
            if not grasp_steps:
                return False, 0, {}
            
            consecutive_sequences = []
            current_seq = [grasp_steps[0]]
            
            for i in range(1, len(grasp_steps)):
                if grasp_steps[i] == grasp_steps[i-1] + 1:
                    current_seq.append(grasp_steps[i])
                else:
                    consecutive_sequences.append(current_seq)
                    current_seq = [grasp_steps[i]]
            consecutive_sequences.append(current_seq)
            
            longest_seq = max(consecutive_sequences, key=len)
            consecutive_steps = len(longest_seq)
            success = consecutive_steps >= min_steps
            
            details = {
                "consecutive_grasp_steps": consecutive_steps,
                "grasp_step_range": f"{longest_seq[0]}-{longest_seq[-1]}",
                "is_src_obj_grasped": True,
            }
            
            return success, consecutive_steps, details
        
        return False, 0, {}
    except Exception as e:
        print(f"⚠️  读取日志失败: {e}")
        return False, 0, {}


def run_single_attempt(attempt, total_attempts, stage_name, episode_dir, task_name, 
                      task_type, original_options, x_range, y_range, original_robot_xy):
    """执行单次尝试"""
    print(f"\n🔄 [{stage_name}] 尝试 {attempt}/{total_attempts}")
    
    options = json.loads(json.dumps(original_options))
    new_xy, _ = modify_robot_position(options, x_range, y_range)
    
    print(f"   🤖 机械臂位置: [{new_xy[0]:.4f}, {new_xy[1]:.4f}]")
    print(f"   📏 偏移: Δx={new_xy[0]-original_robot_xy[0]:+.4f}m, Δy={new_xy[1]-original_robot_xy[1]:+.4f}m")
    
    save_options(episode_dir, options)
    
    print("   ⏳ 执行重放...", end=" ", flush=True)
    replay_success = run_replay_subprocess(episode_dir, task_name)
    
    if not replay_success:
        print("❌ 重放失败")
        return None
    
    print("✅")
    
    print("   🔍 检查抓取...", end=" ", flush=True)
    is_success, grasp_steps, details = check_grasp_success(episode_dir, task_type)
    
    if is_success:
        print(f"✅ 成功！连续抓取 {grasp_steps} 步")
    else:
        print(f"❌ 失败（连续抓取 {grasp_steps} 步）")
    
    return {
        'attempt': attempt,
        'stage': stage_name,
        'robot_position': new_xy,
        'original_robot_position': original_robot_xy,
        'options': options,
        'success': is_success,
        'grasp_steps': grasp_steps,
        'details': details,
    }


def process_single_episode(episode_id, base_dir, direction, task, attempts, min_steps, task_name):
    """处理单个episode（适配简化结构）"""
    
    # 首先复制到优化目录
    print(f"\n📋 准备处理 Episode: {episode_id}")
    print(f"📂 原始目录: {base_dir}")
    
    optimized_episode_dir = copy_episode_to_optimized_dir(episode_id, base_dir)
    
    if not os.path.exists(optimized_episode_dir):
        print(f"\n❌ 错误: Episode目录不存在: {optimized_episode_dir}")
        return None
    
    # 从优化目录加载配置
    original_options = load_options(optimized_episode_dir)
    if original_options is None:
        return None
    
    # 从配置中获取物体位置
    # 支持两种结构:
    # 1. 简化结构: {"model_id": "...", "obj_init_options": {"init_xy": [...], "orientation": [...]}}
    # 2. 标准结构: {"model_ids": [...], "obj_init_options": {"obj1": {...}, "obj2": {...}}, "source_obj_id": 0}
    object_xy = None
    object_name = "unknown"
    
    if "obj_init_options" in original_options:
        obj_opts = original_options["obj_init_options"]
        
        # 检查是否为简化结构
        if isinstance(obj_opts, dict) and "init_xy" in obj_opts:
            # 简化结构
            object_xy = obj_opts["init_xy"]
            object_name = original_options.get("model_id", "unknown")
            print(f"ℹ️  检测到简化结构: {object_name}.init_xy = {object_xy}")
        elif isinstance(obj_opts, dict):
            # 标准结构: 需要从 source_obj_id 获取源物体
            if "source_obj_id" in original_options and "model_ids" in original_options:
                source_idx = original_options["source_obj_id"]
                model_ids = original_options["model_ids"]
                if 0 <= source_idx < len(model_ids):
                    object_name = model_ids[source_idx]
                    if object_name in obj_opts:
                        object_xy = obj_opts[object_name]["init_xy"]
                        print(f"ℹ️  检测到标准结构: {object_name}.init_xy = {object_xy}")
                    else:
                        print(f"❌ 错误: 在 obj_init_options 中找不到物体 '{object_name}'")
                        return None
                else:
                    print(f"❌ 错误: source_obj_id={source_idx} 超出范围")
                    return None
            else:
                print(f"❌ 错误: 标准结构缺少 source_obj_id 或 model_ids 字段")
                return None
        else:
            print(f"❌ 错误: obj_init_options 结构不支持")
            return None
    else:
        print(f"❌ 错误: 缺少 obj_init_options 字段")
        return None
    
    if object_xy is None:
        print(f"❌ 错误: 无法获取物体位置")
        return None
    
    # 计算机械臂初始位置（默认原点）
    robot_xy = [0.0, 0.0]
    if "robot_init_options" in original_options:
        robot_xy = original_options["robot_init_options"]["init_xy"]
    else:
        original_options = initialize_robot_options(original_options, default_xy=robot_xy, force_safe_quat=True)
        print(f"ℹ️  初始化机械臂位置: {robot_xy}")
    
    # 强制修正旋转四元数（即使robot_init_options已存在）
    original_options = initialize_robot_options(original_options, default_xy=robot_xy, force_safe_quat=True)
    
    # 获取方向偏移
    if direction == "closer-to-obj":
        # 计算从机械臂到物体的方向
        dx = object_xy[0] - robot_xy[0]
        dy = object_xy[1] - robot_xy[1]
        
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            print(f"⚠️  警告: 物体与机械臂位置重合，使用备用方向")
            direction_offset = DIRECTION_OFFSETS["up"]
        else:
            # 计算朝向物体的方向，搜索范围控制在±0.05米以内
            # 确保偏移量不超过0.05米
            scale = min(0.05 / max(abs(dx), abs(dy)), 1.0)
            direction_offset = {
                "x": (dx * scale * 0.2, dx * scale * 0.8),
                "y": (dy * scale * 0.2, dy * scale * 0.8)
            }
            print(f"ℹ️  计算朝向物体方向: Δx={dx:+.4f}m, Δy={dy:+.4f}m (scale={scale:.2f})")
    elif direction in DIRECTION_OFFSETS:
        direction_offset = DIRECTION_OFFSETS[direction]
    else:
        print(f"\n❌ 错误: 未知方向 '{direction}'")
        return None
    
    # 计算搜索次数
    coarse_attempts = int(attempts * COARSE_FINE_RATIO)
    fine_attempts = attempts - coarse_attempts
    
    print("\n" + "=" * 70)
    print(f"🎯 处理 Episode: {episode_id}")
    print(f"📍 优化配置目录: {optimized_episode_dir}")
    print(f"🌟 物体: {object_name} [{object_xy[0]:.4f}, {object_xy[1]:.4f}]")
    print(f"🤖 机械臂: [{robot_xy[0]:.4f}, {robot_xy[1]:.4f}]")
    print(f"🧭 方向: {direction}")
    print(f"🔢 搜索: {attempts}次（粗搜索{coarse_attempts} + 精细搜索{fine_attempts}）")
    print("=" * 70)
    
    all_attempts = []
    
    # 粗搜索
    if coarse_attempts > 0:
        print(f"\n🔍 阶段1: 粗搜索")
        for attempt in range(1, coarse_attempts + 1):
            result = run_single_attempt(
                attempt, coarse_attempts, "粗搜索",
                optimized_episode_dir, task_name, task,
                original_options,
                direction_offset['x'], direction_offset['y'],
                robot_xy
            )
            if result:
                all_attempts.append(result)
                if result['success']:
                    print(f"\n🎉 找到成功配置！提前进入精细搜索...")
                    break
    
    # 精细搜索
    if fine_attempts > 0 and all_attempts:
        all_attempts.sort(key=lambda x: x['grasp_steps'], reverse=True)
        best_coarse = all_attempts[0]
        best_position = best_coarse['robot_position']
        
        print(f"\n📊 阶段1完成！最佳: {best_coarse['grasp_steps']} 步")
        print(f"🔍 阶段2: 精细搜索")
        
        for attempt in range(1, fine_attempts + 1):
            options = json.loads(json.dumps(original_options))
            options = initialize_robot_options(options, default_xy=best_position, force_safe_quat=True)
            
            x_offset = random.uniform(FINE_SEARCH_RANGE['x'][0], FINE_SEARCH_RANGE['x'][1])
            y_offset = random.uniform(FINE_SEARCH_RANGE['y'][0], FINE_SEARCH_RANGE['y'][1])
            
            new_xy = [
                float(np.clip(best_position[0] + x_offset, ROBOT_X_RANGE[0], ROBOT_X_RANGE[1])),
                float(np.clip(best_position[1] + y_offset, ROBOT_Y_RANGE[0], ROBOT_Y_RANGE[1]))
            ]
            options["robot_init_options"]["init_xy"] = new_xy
            
            print(f"\n🔄 [精细搜索] 尝试 {attempt}/{fine_attempts}")
            print(f"   🤖 位置: [{new_xy[0]:.4f}, {new_xy[1]:.4f}]")
            
            save_options(optimized_episode_dir, options)
            
            print("   ⏳ 执行重放...", end=" ", flush=True)
            replay_success = run_replay_direct(optimized_episode_dir, task_name)
            
            if not replay_success:
                print("❌ 重放失败")
                continue
            
            print("✅")
            print("   🔍 检查抓取...", end=" ", flush=True)
            is_success, grasp_steps, details = check_grasp_success(optimized_episode_dir, task, min_steps)
            
            if is_success:
                print(f"✅ 成功！连续抓取 {grasp_steps} 步")
            else:
                print(f"❌ 失败（连续抓取 {grasp_steps} 步）")
            
            all_attempts.append({
                'attempt': coarse_attempts + attempt,
                'stage': '精细搜索',
                'robot_position': new_xy,
                'original_robot_position': robot_xy,
                'options': options,
                'success': is_success,
                'grasp_steps': grasp_steps,
                'details': details,
            })
    
    if not all_attempts:
        return None
    
    # 返回最佳结果
    all_attempts.sort(key=lambda x: x['grasp_steps'], reverse=True)
    best = all_attempts[0]
    
    return {
        'episode_id': episode_id,
        'optimized_dir': optimized_episode_dir,
        'best_result': best,
        'original_position': robot_xy,
        'all_attempts': all_attempts
    }


def get_all_episode_ids(base_dir):
    """获取目录下所有有效的episode ID"""
    episode_ids = []
    
    print(f"🔍 扫描目录: {base_dir}")
    print(f"   绝对路径: {os.path.abspath(base_dir)}")
    
    if not os.path.exists(base_dir):
        print(f"❌ 错误: 目录不存在！")
        return episode_ids
    
    try:
        all_items = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
        print(f"📂 找到 {len(all_items)} 个目录项")
        
        valid_count = 0
        for item in all_items:
            episode_path = os.path.join(base_dir, item)
            options_path = os.path.join(episode_path, "options.json")
            
            if os.path.exists(options_path):
                episode_ids.append(item)
                valid_count += 1
                print(f"   ✅ {item}")
            else:
                print(f"   ⏭️  跳过 '{item}': 缺少 options.json")
        
        print(f"✅ 找到 {valid_count} 个有效episode")
    except Exception as e:
        print(f"❌ 扫描目录失败: {e}")
    
    return episode_ids


def print_summary(all_results):
    """打印批量执行总结"""
    print("\n" + "=" * 70)
    print("📊 批量处理完成！总结报告")
    print("=" * 70)
    
    if not all_results:
        print("⚠️  没有成功处理任何episode")
        return
    
    # 统计信息
    total_episodes = len(all_results)
    successful_episodes = sum(1 for r in all_results if r['best_result']['success'])
    
    print(f"\n📈 统计信息:")
    print(f"   总处理场景数: {total_episodes}")
    print(f"   成功找到配置: {successful_episodes}")
    print(f"   成功率: {successful_episodes/total_episodes*100:.1f}%")
    
    # 详细结果
    print(f"\n📋 详细结果:")
    for i, result in enumerate(all_results, 1):
        ep = result['episode_id']
        best = result['best_result']
        opt_dir = result['optimized_dir']
        status = "✅ 成功" if best['success'] else "❌ 未达标"
        print(f"   {i:2d}. {ep:>4}: {status} | {best['grasp_steps']:2d}步 | "
              f"位置: [{best['robot_position'][0]:+.4f}, {best['robot_position'][1]:+.4f}]")
        print(f"       配置目录: {opt_dir}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="通过优化机械臂初始位置实现抓取成功（支持批量处理）",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 模式选择
    parser.add_argument(
        '--all',
        action='store_true',
        help="批量处理所有episode"
    )
    
    parser.add_argument(
        '--episode',
        type=str,
        help="单个episode ID（与--all互斥）"
    )
    
    parser.add_argument(
        '--direction',
        type=str,
        default="closer-to-obj",
        choices=list(DIRECTION_OFFSETS.keys()) + ['closer-to-obj'],
        help="搜索方向（默认: closer-to-obj）"
    )
    
    parser.add_argument(
        '--base_dir',
        type=str,
        default=DEFAULT_BASE_DIR,
        help=f"基础目录（默认: {DEFAULT_BASE_DIR}）"
    )
    
    parser.add_argument(
        '--task',
        type=str,
        choices=["move", "grasp"],
        default=DEFAULT_TASK_TYPE,
        help=f"任务类型（默认: {DEFAULT_TASK_TYPE}）"
    )
    
    parser.add_argument(
        '--attempts',
        type=int,
        default=DEFAULT_TOTAL_ATTEMPTS,
        help=f"总搜索次数（默认: {DEFAULT_TOTAL_ATTEMPTS}）"
    )
    
    parser.add_argument(
        '--min_steps',
        type=int,
        default=MIN_CONSECUTIVE_GRASP_STEPS,
        help=f"最小连续抓取步数（默认: {MIN_CONSECUTIVE_GRASP_STEPS}）"
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if not args.all and not args.episode:
        print("❌ 错误: 必须指定 --all 或 --episode")
        return 1
    
    # 任务名称映射
    task_map = {
        "move": "google_robot_move_near_customizable",
        "grasp": "google_robot_pick_customizable"
    }
    task_name = task_map[args.task]
    
    print(f"📂 当前工作目录: {os.getcwd()}")
    print(f"🔍 扫描基础目录: {os.path.abspath(args.base_dir)}")
    
    # 批量处理模式
    if args.all:
        print("=" * 70)
        print("🤖 批量模式：处理所有episode")
        print("=" * 70)
        
        episode_ids = get_all_episode_ids(args.base_dir)
        
        if not episode_ids:
            print(f"\n❌ 在 {args.base_dir} 下未找到任何有效的episode目录")
            print("💡 请检查 --base_dir 参数是否正确")
            return 1
        
        print(f"📋 处理列表: {', '.join(episode_ids[:5])}{'...' if len(episode_ids) > 5 else ''}")
        print("=" * 70)
        
        all_results = []
        for i, ep_id in enumerate(episode_ids, 1):
            print(f"\n\n【进度: {i}/{len(episode_ids)}】")
            try:
                result = process_single_episode(
                    ep_id, args.base_dir, args.direction,
                    args.task, args.attempts, args.min_steps, task_name
                )
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"\n❌ 处理 {ep_id} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 打印总结
        print_summary(all_results)
        
        successful = any(r['best_result']['success'] for r in all_results)
        return 0 if successful else 1
    
    # 单个episode处理模式
    else:
        result = process_single_episode(
            args.episode, args.base_dir, args.direction,
            args.task, args.attempts, args.min_steps, task_name
        )
        
        if result is None:
            return 1
        
        best = result['best_result']
        print("\n" + "=" * 70)
        if best['success']:
            print("✅ 成功！已达到目标")
            return 0
        else:
            print(f"⚠️  最优结果({best['grasp_steps']}步)仍未达到目标({args.min_steps}步)")
            return 1


if __name__ == "__main__":
    exit(main())


# python3 fix_strategy_optimize_robot_all.py --all --attempts 15