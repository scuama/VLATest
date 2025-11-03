"""
扩展的VLA接口 - 记录动作序列用于训练数据收集
"""
import os
import numpy as np
import simpler_env
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

# 修正路径：从项目根目录查找checkpoints
ckpt_dir = str(PACKAGE_DIR.parent.parent) + '/checkpoints'

sapien.render_config.rt_use_denoiser = True


class VLAInterfaceWithActions:
    """VLA接口，记录完整的动作序列"""
    
    def __init__(self, task, model_name):
        if task in TASKS:
            self.task = task
        else:
            raise ValueError(task)
        if "google" in self.task:
            self.policy_setup = "google_robot"
        else:
            self.policy_setup = "widowx_bridge"
        
        if "rt_1" in model_name:
            from simpler_env.policies.rt1.rt1_model import RT1Inference
            ckpt_path = os.path.join(ckpt_dir, RT_1_CHECKPOINTS[model_name])
            self.model = RT1Inference(saved_model_path=ckpt_path, policy_setup=self.policy_setup)
        elif "octo" in model_name:
            from simpler_env.policies.octo.octo_model import OctoInference
            self.model = OctoInference(model_type=model_name, policy_setup=self.policy_setup, init_rng=0)
        elif "openvla" in model_name:
            from simpler_env.policies.openvla.openvla_model import OpenVLAInference
            self.model = OpenVLAInference(model_type=model_name, policy_setup=self.policy_setup)
        else:
            raise ValueError(model_name)

    def run_interface(self, seed=None, options=None):
        """运行接口并记录完整的动作序列"""
        env = simpler_env.make(self.task)
        obs, reset_info = env.reset(seed=seed, options=options)
        
        # 获取语言指令
        if hasattr(env, 'get_language_instruction'):
            instruction = env.get_language_instruction()
        else:
            instruction = env.unwrapped.get_language_instruction()
        
        self.model.reset(instruction)
        print(instruction)
        print("Reset info", reset_info)

        image = get_image_from_maniskill2_obs_dict(env, obs)
        images = [image]
        actions = []  # 记录动作序列
        raw_actions = []  # 记录原始动作
        
        predicted_terminated, success, truncated = False, False, False
        timestep = 0
        episode_stats = {}
        
        while not (predicted_terminated or truncated):
            # 获取模型动作
            raw_action, action = self.model.step(image)
            predicted_terminated = bool(action["terminate_episode"][0] > 0)
            
            # 记录动作
            action_array = np.concatenate([
                action["world_vector"],
                action["rot_axangle"],
                action["gripper"]
            ])
            actions.append(action_array)
            raw_actions.append(raw_action)
            
            # 执行动作
            obs, reward, success, truncated, info = env.step(action_array)
            
            print(timestep, info)
            episode_stats[timestep] = info
            
            # 更新图像
            image = get_image_from_maniskill2_obs_dict(env, obs)
            images.append(image)
            timestep += 1

        print(f"Episode success: {success}")
        env.close()
        del env
        
        return images, actions, raw_actions, episode_stats
