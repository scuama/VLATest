import os
import numpy as np
import simpler_env
import gym
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import sapien.core as sapien

from pathlib import Path

SEED = 2024

PACKAGE_DIR = Path(__file__).parent.resolve()

RT_1_CHECKPOINTS = {
    "rt_1_x": "rt_1_x_tf_trained_for_002272480_step",
    "rt_1_400k": "rt_1_tf_trained_for_000400120",
    "rt_1_58k": "rt_1_tf_trained_for_000058240",
    "rt_1_1k": "rt_1_tf_trained_for_000001120",
}

TASKS = [
    "google_robot_pick_customizable",
    "google_robot_pick_customizable_ycb",
    "google_robot_pick_customizable_no_overlay",
    "google_robot_move_near_customizable",
    "google_robot_move_near_customizable_ycb",
    "google_robot_move_near_customizable_no_overlay",
    "widowx_put_on_customizable",
    "widowx_put_on_customizable_ycb",
    "widowx_put_on_customizable_no_overlay",
    "widowx_put_in_customizable",
    "widowx_put_in_customizable_ycb",
    "widowx_put_in_customizable_no_overlay",
]

def create_customized_environment():
    options = {
        # 机械臂位置和朝向
        "robot_init_options": {
            "init_xy": [0.4, 0.1],                    # X,Y坐标
            "init_rot_quat": [0, 0, 0, 1]             # 四元数表示朝向
        },
         "camera_cfgs": {
            "add_segmentation": True,
            "width": 256,
            "height": 256
        }
    }
    
    return options

ckpt_dir = str(PACKAGE_DIR) + '/../checkpoints'

sapien.render_config.rt_use_denoiser = True

def unwrap_and_print_env_attributes(env):
    """解包环境并打印所有属性信息"""
    
    print("=" * 60)
    print("ENVIRONMENT UNWRAPPING AND ATTRIBUTE ANALYSIS")
    print("=" * 60)
    
    # 1. 打印包装器链
    print("\n1. ENVIRONMENT WRAPPER CHAIN:")
    print("-" * 30)
    current_env = env
    level = 0
    while hasattr(current_env, 'env'):
        print(f"  Level {level}: {type(current_env).__name__}")
        current_env = current_env.env
        level += 1
        if level > 10:  # 防止无限循环
            print("  ... (too many levels, stopping)")
            break
    print(f"  Core environment: {type(current_env).__name__}")
    
    # 2. 获取核心环境
    core_env = env
    while hasattr(core_env, 'env'):
        core_env = core_env.env
    
    print(f"\n2. CORE ENVIRONMENT TYPE:")
    print("-" * 30)
    print(f"  Class: {type(core_env).__module__}.{type(core_env).__name__}")
    
    # 3. 打印核心环境的所有属性
    print(f"\n3. CORE ENVIRONMENT ATTRIBUTES:")
    print("-" * 30)
    
    # 获取所有属性名（排除私有属性和方法）
    attrs = [attr for attr in dir(core_env) 
             if not attr.startswith('_') and not callable(getattr(core_env, attr, None))]
    
    # 分类属性
    basic_attrs = []
    complex_attrs = []
    none_attrs = []
    
    for attr in attrs:
        try:
            value = getattr(core_env, attr)
            if value is None:
                none_attrs.append(attr)
            elif isinstance(value, (str, int, float, bool, list, dict, tuple)):
                basic_attrs.append((attr, value))
            else:
                complex_attrs.append((attr, type(value).__name__))
        except Exception as e:
            complex_attrs.append((attr, f"<Error accessing: {e}>"))
    
    # 打印基本属性
    print("  Basic attributes:")
    for attr, value in basic_attrs:
        if isinstance(value, str) and len(value) > 100:
            print(f"    {attr}: {value[:100]}...")
        else:
            print(f"    {attr}: {value}")
    
    # 打印复杂属性
    print("  Complex attributes:")
    for attr, type_name in complex_attrs:
        print(f"    {attr}: {type_name}")
    
    # 打印None属性
    if none_attrs:
        print("  None attributes:")
        for attr in none_attrs:
            print(f"    {attr}: None")
    
    # 4. 打印特殊关注的属性
    print(f"\n4. SPECIAL ATTRIBUTES:")
    print("-" * 30)
    
    special_attrs = [
        'model_id', 'model_scale', 'obj', 'agent', 'observation_space', 
        'action_space', 'episode_stats', 'obj_init_options'
    ]
    
    for attr in special_attrs:
        if hasattr(core_env, attr):
            try:
                value = getattr(core_env, attr)
                if attr == 'obj' and value:
                    print(f"    {attr}: {type(value).__name__}")
                    print(f"      name: {getattr(value, 'name', 'N/A')}")
                    if hasattr(value, 'pose'):
                        print(f"      pose: {value.pose}")
                elif attr in ['observation_space', 'action_space'] and value:
                    print(f"    {attr}: {type(value).__name__}")
                    print(f"      shape: {getattr(value, 'shape', 'N/A')}")
                else:
                    print(f"    {attr}: {value}")
            except Exception as e:
                print(f"    {attr}: <Error accessing: {e}>")
        else:
            print(f"    {attr}: <Not found>")
    
    # 5. 打印方法信息
    print(f"\n5. AVAILABLE METHODS:")
    print("-" * 30)
    methods = [method for method in dir(core_env) 
               if not method.startswith('_') and callable(getattr(core_env, method, None))]
    for method in sorted(methods):
        print(f"    {method}")
    
    # 6. 打印环境配置信息
    print(f"\n6. ENVIRONMENT CONFIGURATION:")
    print("-" * 30)
    config_attrs = ['task', 'model_ids', 'asset_root', 'scene_name']
    for attr in config_attrs:
        if hasattr(core_env, attr):
            print(f"    {attr}: {getattr(core_env, attr)}")
    
    env.close()
    return env

def reset_robot_position_and_print(task_name, robot_xy=None, robot_rotation=None):
    """通过options参数重设机械臂位置并打印参数"""
    print("=" * 60)
    print("RESET ROBOT POSITION AND PRINT PARAMETERS")
    print("=" * 60)
    
    env = simpler_env.make(task_name)
    core_env=env
    while hasattr(core_env, 'env'):
        core_env = core_env.env
    core_env.prepackaged_config = False
    print("Disabled prepackaged config")
    # 准备options参数
    options = {}

    options = create_customized_environment()
    
    print(f"\nResetting environment with options: {options}")
    
    # 重置环境
    obs, reset_info = env.reset(seed = 1,options=options)
    
    # 获取核心环境
    core_env = env
    while hasattr(core_env, 'env'):
        core_env = core_env.env
    
    # 打印重置后的参数
    print("\n" + "-" * 40)
    print("AFTER RESET PARAMETERS:")
    print("-" * 40)
    print(f"Robot init options after reset: {getattr(core_env, 'robot_init_options', 'Not found')}")
    
    if hasattr(core_env, 'robot_init_options'):
        robot_opts = core_env.robot_init_options
        if 'init_xy' in robot_opts:
            print(f"Robot XY position: {robot_opts['init_xy']}")
        if 'init_rot_quat' in robot_opts:
            print(f"Robot rotation: {robot_opts['init_rot_quat']}")
    
    # 打印机器人信息
    if hasattr(core_env, 'agent') and core_env.agent:
        print(f"Agent type: {type(core_env.agent).__name__}")
        if hasattr(core_env.agent, 'robot') and core_env.agent.robot:
            print(f"Robot pose: {core_env.agent.robot.pose}")
    
    # 打印完整环境属性
    unwrap_and_print_env_attributes(env)
    
    # 不要在这里关闭环境，让它在主函数中关闭
    return env, obs, reset_info

if __name__ == '__main__':
    # 使用示例
    task_name = "google_robot_pick_customizable_ycb"
    env, obs, info = reset_robot_position_and_print(
        task_name, 
        robot_xy=[0.4, 0.1],  # 新的XY位置
        robot_rotation=[0, 0, 0, 1]  # 新的旋转四元数
    )
    env.close()  # 在主函数中关闭环境