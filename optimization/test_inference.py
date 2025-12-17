#!/usr/bin/env python3
"""
测试批量推理脚本
用于验证 batch_dataset.json 的推理功能
"""

import os
import sys
import json
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/mnt/disk1/decom/VLATest')

from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from experiments.openvla_utils import get_model, get_processor


def load_dataset(dataset_file):
    """加载数据集文件"""
    with open(dataset_file, 'r') as f:
        return json.load(f)


def create_env(task_name, model_id, obj_init_options, seed=0):
    """创建环境
    
    Args:
        task_name: 任务名称
        model_id: 物体模型ID
        obj_init_options: 物体初始化选项
        seed: 随机种子
    """
    from simpler_env.utils.env.env_builder import build_maniskill2_env
    
    env = build_maniskill2_env(
        env_name=task_name,
        obs_mode="rgbd",
        enable_raytracing="auto",
        robot_uid="google_robot_static"
    )
    
    # 重置环境
    env_reset_options = {
        "obj_init_options": {
            model_id: obj_init_options
        }
    }
    
    obs = env.reset(seed=seed, options=env_reset_options)
    
    return env, obs


def run_inference_single(env, obs, model, processor, episode_idx, output_dir):
    """运行单个场景的推理
    
    Args:
        env: 环境
        obs: 初始观察
        model: VLA模型
        processor: 处理器
        episode_idx: Episode索引
        output_dir: 输出目录
    """
    from PIL import Image
    import torch
    
    output_dir = Path(output_dir)
    episode_dir = output_dir / str(episode_idx)
    episode_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = episode_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    log_data = {}
    max_steps = 200
    
    print(f"\n{'='*70}")
    print(f"推理 Episode {episode_idx}")
    print(f"{'='*70}")
    
    for step in range(max_steps):
        # 获取图像
        image = get_image_from_maniskill2_obs_dict(env, obs)
        
        # 保存图像
        if step % 10 == 0:
            img_path = images_dir / f"step_{step:04d}.jpg"
            Image.fromarray(image).save(img_path)
        
        # 获取动作
        # 注意：这里需要根据实际的 openvla 推理接口调整
        # 简化版本，假设有 predict_action 函数
        try:
            # 处理图像
            inputs = processor(images=image, return_tensors="pt")
            
            # 预测动作
            with torch.no_grad():
                outputs = model(**inputs)
                action = outputs.logits[0].cpu().numpy()
            
            # 执行动作
            obs, reward, done, info = env.step(action)
            
            # 记录日志
            log_data[str(step)] = {
                "action": action.tolist(),
                "reward": float(reward),
                "done": bool(done),
                **{k: v for k, v in info.items() if isinstance(v, (int, float, bool, str))}
            }
            
            if step % 10 == 0:
                print(f"  Step {step}: reward={reward:.3f}, done={done}")
            
            if done:
                print(f"  ✅ 完成于 Step {step}")
                break
                
        except Exception as e:
            print(f"  ❌ Step {step} 失败: {e}")
            log_data[str(step)] = {
                "error": str(e)
            }
            break
    
    # 保存日志
    log_file = episode_dir / "log.json"
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"  📄 日志已保存: {log_file}")
    
    env.close()
    
    return log_data


def main():
    parser = argparse.ArgumentParser(description="测试批量推理")
    parser.add_argument("--dataset", type=str, required=True, help="数据集文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--task", type=str, default="google_robot_move_near_customizable", 
                       help="任务名称")
    parser.add_argument("--model", type=str, default="openvla-7b", help="模型名称")
    parser.add_argument("--episode", type=int, help="只推理指定的 episode index")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("测试批量推理")
    print("=" * 70)
    print(f"数据集: {args.dataset}")
    print(f"输出目录: {args.output}")
    print(f"任务: {args.task}")
    print(f"模型: {args.model}")
    print("=" * 70)
    
    # 1. 加载数据集
    print("\n📦 加载数据集...")
    dataset = load_dataset(args.dataset)
    num_episodes = dataset.get("num", 0)
    seed = dataset.get("seed", 0)
    
    print(f"  场景数量: {num_episodes}")
    print(f"  随机种子: {seed}")
    
    # 2. 加载模型
    print("\n🤖 加载模型...")
    try:
        model, processor = get_model(args.model)
        print(f"  ✅ 模型加载成功")
    except Exception as e:
        print(f"  ❌ 模型加载失败: {e}")
        return 1
    
    # 3. 运行推理
    episodes_to_run = [args.episode] if args.episode is not None else range(num_episodes)
    
    for idx in episodes_to_run:
        if str(idx) not in dataset:
            print(f"\n⚠️  Episode {idx} 不在数据集中")
            continue
        
        episode_data = dataset[str(idx)]
        model_id = episode_data["model_id"]
        obj_init_options = episode_data["obj_init_options"]
        
        print(f"\n{'#'*70}")
        print(f"# Episode {idx}: {model_id}")
        print(f"{'#'*70}")
        
        try:
            # 创建环境
            env, obs = create_env(args.task, model_id, obj_init_options, seed)
            
            # 运行推理
            run_inference_single(env, obs, model, processor, idx, args.output)
            
        except Exception as e:
            print(f"❌ Episode {idx} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print("✅ 推理完成")
    print(f"{'='*70}")
    print(f"结果保存在: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
