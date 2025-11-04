#!/usr/bin/env python3
"""
直接复制notebook中的代码来测试OpenVLA
"""
import os
import numpy as np
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import sapien.core as sapien

# 设置
task_name = "google_robot_pick_customizable"
SEED = 2024

# 创建环境
sapien.render_config.rt_use_denoiser = True
env = simpler_env.make(task_name)

obs, reset_info = env.reset(seed=SEED)
# Handle potential wrapper
if hasattr(env, 'get_language_instruction'):
    instruction = env.get_language_instruction()
else:
    instruction = env.unwrapped.get_language_instruction()
print("Reset info", reset_info)
print("Instruction", instruction)

policy_setup = "google_robot"

# 加载模型
model_name = "openvla-7b"

print(f"\n加载模型: {model_name}")
from simpler_env.policies.openvla.openvla_model import OpenVLAInference
model = OpenVLAInference(model_type=model_name, policy_setup=policy_setup)
print("✓ 模型加载成功")

# 运行推理
print("\n开始推理...")
model.reset(instruction)

image = get_image_from_maniskill2_obs_dict(env, obs)
images = [image]
predicted_terminated, success, truncated = False, False, False
timestep = 0

try:
    while not (predicted_terminated or truncated) and timestep < 100:  # 限制最大步数
        print(f"Step {timestep}...")
        # step the model
        raw_action, action = model.step(image)
        predicted_terminated = bool(action["terminate_episode"][0] > 0)
        obs, reward, success, truncated, info = env.step(
            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
        )
        print(f"  {timestep}: {info}")
        # update image observation
        image = get_image_from_maniskill2_obs_dict(env, obs)
        images.append(image)
        timestep += 1
    
    print(f"\n✅ 推理完成！")
    print(f"成功: {success}")
    print(f"总步数: {timestep}")
    
except Exception as e:
    print(f"\n❌ 推理失败: {type(e).__name__}")
    print(f"错误信息: {str(e)[:200]}")
    import traceback
    traceback.print_exc()

finally:
    env.close()
