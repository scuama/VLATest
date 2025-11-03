"""
数据收集脚本 - 从成功案例中收集专家演示
使用RT-1模型在已知成功的场景上收集轨迹数据
"""
import argparse
import json
import os
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image

from training.scripts.model_interface_with_actions import VLAInterfaceWithActions

# 任务配置
TASK_CONFIG = {
    'grasp': {
        'task_name': 'google_robot_pick_customizable',
        'data_file': 't-grasp_n-1000_o-m3_s-2498586606.json',
        'dataset_name': 'fractal20220817_data'
    },
    'move': {
        'task_name': 'google_robot_move_near_customizable',
        'data_file': 't-move_n-1000_o-m3_s-2263834374.json',
        'dataset_name': 'fractal20220817_data'
    },
    'put-on': {
        'task_name': 'widowx_put_on_customizable',
        'data_file': 't-put-on_n-1000_o-m3_s-2593734741.json',
        'dataset_name': 'bridge_orig'
    },
    'put-in': {
        'task_name': 'widowx_put_in_customizable',
        'data_file': 't-put-in_n-1000_o-m3_s-2905191776.json',
        'dataset_name': 'bridge_orig'
    }
}

# 模型映射
MODEL_KEYS = {
    'rt_1_x': 'rt_1_x_2024',
    'rt_1_400k': 'rt_1_400k_2024',
    'rt_1_58k': 'rt_1_58k_2024'
}


def extract_action_from_step(env_step_info):
    """从环境step信息中提取动作（需要从模型输出获取）"""
    # 这个函数会在run_interface中被调用时记录动作
    # 暂时返回None，实际动作会在收集时记录
    return None


def collect_trajectory(vla_interface, task_config, scene_idx, seed, save_images=True):
    """收集单个场景的轨迹"""
    # 加载场景配置
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'data' / task_config['data_file']
    with open(data_path, 'r') as f:
        tasks = json.load(f)
    
    options = tasks[str(scene_idx)]
    
    # 运行模型收集轨迹
    # run_interface返回: images, actions, raw_actions, episode_stats
    images, actions, raw_actions, episode_stats = vla_interface.run_interface(seed=seed, options=options)
    
    # 检查是否成功
    last_step = max(episode_stats.keys())
    success = episode_stats[last_step].get('success', False)
    if isinstance(success, str):
        success = success.lower() == 'true'
    
    if not success:
        return None
    
    # 提取轨迹数据
    trajectory = {
        'scene_idx': scene_idx,
        'images': images,
        'actions': actions,
        'raw_actions': raw_actions,
        'num_steps': len(images),
        'success': success,
        'episode_stats': episode_stats,
        'task': task_config['task_name'],
        'dataset_name': task_config['dataset_name']
    }
    
    return trajectory


def collect_expert_data(args):
    """主数据收集函数"""
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载任务配置
    task_config = TASK_CONFIG[args.task]
    
    # 加载成功案例索引
    # 修正路径：从项目根目录查找
    project_root = Path(__file__).parent.parent.parent
    success_file = project_root / 'data' / 'all-correct-task.json'
    with open(success_file, 'r') as f:
        all_success = json.load(f)
    
    # 获取该任务的成功索引
    task_key = f"t-{args.task}_n-1000_o-m3_s-" + task_config['data_file'].split('_s-')[1].replace('.json', '')
    model_key = MODEL_KEYS[args.model]
    success_indices = all_success[task_key][model_key]
    
    print(f"任务: {args.task}")
    print(f"模型: {args.model}")
    print(f"成功案例数: {len(success_indices)}")
    
    # 限制收集数量
    if args.num_samples > 0:
        success_indices = success_indices[:args.num_samples]
        print(f"限制收集数量: {args.num_samples}")
    
    # 初始化模型
    print(f"\n初始化模型...")
    vla = VLAInterfaceWithActions(model_name=args.model, task=task_config['task_name'])
    
    # 收集轨迹
    trajectories = []
    failed_count = 0
    log_file = output_dir / 'collection.log'
    
    print(f"\n开始收集轨迹...")
    with open(log_file, 'w') as log:
        log.write(f"Task: {args.task}\n")
        log.write(f"Model: {args.model}\n")
        log.write(f"Total scenes: {len(success_indices)}\n\n")
        
        for idx in tqdm(success_indices, desc="收集轨迹"):
            try:
                trajectory = collect_trajectory(
                    vla, 
                    task_config, 
                    idx, 
                    seed=args.seed,
                    save_images=True
                )
                
                if trajectory is not None:
                    trajectories.append(trajectory)
                    log.write(f"✓ Scene {idx}: Success ({trajectory['num_steps']} steps)\n")
                    log.flush()
                else:
                    failed_count += 1
                    log.write(f"✗ Scene {idx}: Failed (not successful)\n")
                    log.flush()
                    
            except Exception as e:
                failed_count += 1
                log.write(f"✗ Scene {idx}: Error - {str(e)}\n")
                log.flush()
                print(f"\n警告: Scene {idx} 收集失败: {e}")
                continue
    
    print(f"\n收集完成!")
    print(f"成功: {len(trajectories)}/{len(success_indices)}")
    print(f"失败: {failed_count}")
    
    # 保存轨迹数据
    print(f"\n保存数据到 {output_dir}...")
    
    # 保存轨迹（不含图像，图像单独保存）
    trajectories_light = []
    for i, traj in enumerate(trajectories):
        # 保存图像
        img_dir = output_dir / f"scene_{traj['scene_idx']}"
        img_dir.mkdir(exist_ok=True)
        
        for step_idx, img in enumerate(traj['images']):
            img_pil = Image.fromarray(img)
            img_pil.save(img_dir / f"step_{step_idx:03d}.jpg", quality=95)
        
        # 保存动作序列
        actions_array = np.array(traj['actions'])  # shape: (num_steps, 7)
        np.save(img_dir / 'actions.npy', actions_array)
        
        # 保存轻量级轨迹信息
        traj_light = {
            'scene_idx': traj['scene_idx'],
            'num_steps': traj['num_steps'],
            'success': traj['success'],
            'episode_stats': traj['episode_stats'],
            'task': traj['task'],
            'dataset_name': traj['dataset_name'],
            'image_dir': str(img_dir.relative_to(output_dir)),
            'actions_shape': actions_array.shape
        }
        trajectories_light.append(traj_light)
    
    # 保存元数据
    with open(output_dir / 'trajectories.pkl', 'wb') as f:
        pickle.dump(trajectories_light, f)
    
    metadata = {
        'task': args.task,
        'model': args.model,
        'num_trajectories': len(trajectories),
        'total_steps': sum(t['num_steps'] for t in trajectories),
        'success_indices': [t['scene_idx'] for t in trajectories],
        'dataset_name': task_config['dataset_name']
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ 保存完成!")
    print(f"  - 轨迹数: {len(trajectories)}")
    print(f"  - 总步数: {metadata['total_steps']}")
    print(f"  - 数据目录: {output_dir}")
    
    return trajectories


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='收集专家演示数据')
    parser.add_argument('--task', type=str, required=True,
                        choices=['grasp', 'move', 'put-on', 'put-in'],
                        help='任务类型')
    parser.add_argument('--model', type=str, default='rt_1_400k',
                        choices=['rt_1_x', 'rt_1_400k', 'rt_1_58k'],
                        help='使用的模型')
    parser.add_argument('--num_samples', type=int, default=-1,
                        help='收集的样本数量，-1表示全部')
    parser.add_argument('--output', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--seed', type=int, default=2024,
                        help='随机种子')
    
    args = parser.parse_args()
    
    collect_expert_data(args)
