"""
数据预处理脚本 - 将收集的轨迹转换为训练格式
"""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm


class VLATrainingDataset:
    """OpenVLA训练数据集"""
    
    def __init__(self, data_dir, image_size=224):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        
        # 加载元数据
        with open(self.data_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        with open(self.data_dir / 'trajectories.pkl', 'rb') as f:
            self.trajectories = pickle.load(f)
        
        # 构建样本索引 (trajectory_idx, step_idx)
        self.samples = []
        for traj_idx, traj in enumerate(self.trajectories):
            for step_idx in range(traj['num_steps']):
                self.samples.append((traj_idx, step_idx))
        
        print(f"数据集加载完成:")
        print(f"  - 轨迹数: {len(self.trajectories)}")
        print(f"  - 样本数: {len(self.samples)}")
        print(f"  - 任务: {self.metadata['task']}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        traj_idx, step_idx = self.samples[idx]
        traj = self.trajectories[traj_idx]
        
        # 加载图像
        img_dir = self.data_dir / traj['image_dir']
        img_path = img_dir / f"step_{step_idx:03d}.jpg"
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        image = np.array(image, dtype=np.uint8)
        
        # 加载动作
        actions = np.load(img_dir / 'actions.npy')
        action = actions[step_idx].astype(np.float32)
        
        # 获取语言指令（从任务名生成）
        instruction = self._get_instruction(traj['task'])
        
        return {
            'image': image,
            'action': action,
            'instruction': instruction,
            'dataset_name': traj['dataset_name']
        }
    
    def _get_instruction(self, task_name):
        """根据任务名生成语言指令"""
        if 'pick' in task_name:
            return "pick up the object"
        elif 'move' in task_name:
            return "move the object near the target"
        elif 'put_on' in task_name:
            return "put the object on the target"
        elif 'put_in' in task_name:
            return "put the object in the container"
        else:
            return "complete the task"


def prepare_training_data(args):
    """准备训练数据"""
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    # 加载原始数据
    print("\n加载原始数据...")
    with open(input_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    with open(input_dir / 'trajectories.pkl', 'rb') as f:
        trajectories = pickle.load(f)
    
    print(f"轨迹数: {len(trajectories)}")
    print(f"总步数: {metadata['total_steps']}")
    
    # 构建训练样本
    print("\n构建训练样本...")
    samples = []
    
    for traj_idx, traj in enumerate(tqdm(trajectories, desc="处理轨迹")):
        img_dir = input_dir / traj['image_dir']
        actions = np.load(img_dir / 'actions.npy')
        
        # 注意：num_steps包括最后一张图像，但动作数=num_steps-1
        num_actions = len(actions)
        for step_idx in range(num_actions):
            sample = {
                'trajectory_idx': traj_idx,
                'step_idx': step_idx,
                'scene_idx': traj['scene_idx'],
                'image_path': str(img_dir / f"step_{step_idx:03d}.jpg"),
                'action': actions[step_idx].tolist(),
                'task': traj['task'],
                'dataset_name': traj['dataset_name']
            }
            samples.append(sample)
    
    print(f"总样本数: {len(samples)}")
    
    # 划分训练/验证集
    np.random.seed(42)
    indices = np.random.permutation(len(samples))
    split_idx = int(len(samples) * args.train_ratio)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    
    print(f"\n数据划分:")
    print(f"  - 训练集: {len(train_samples)} ({args.train_ratio*100:.0f}%)")
    print(f"  - 验证集: {len(val_samples)} ({(1-args.train_ratio)*100:.0f}%)")
    
    # 保存处理后的数据
    print("\n保存处理后的数据...")
    
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    with open(train_dir / 'samples.json', 'w') as f:
        json.dump(train_samples, f, indent=2)
    
    with open(val_dir / 'samples.json', 'w') as f:
        json.dump(val_samples, f, indent=2)
    
    # 保存元数据
    processed_metadata = {
        'original_metadata': metadata,
        'num_train_samples': len(train_samples),
        'num_val_samples': len(val_samples),
        'train_ratio': args.train_ratio,
        'image_size': 224,
        'action_dim': 7,
        'input_dir': str(input_dir)
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(processed_metadata, f, indent=2)
    
    print(f"\n✓ 数据预处理完成!")
    print(f"  - 输出目录: {output_dir}")
    print(f"  - 训练样本: {len(train_samples)}")
    print(f"  - 验证样本: {len(val_samples)}")
    
    # 数据统计
    print(f"\n数据统计:")
    action_stats = compute_action_statistics(train_samples)
    print(f"  - 动作均值: {action_stats['mean']}")
    print(f"  - 动作标准差: {action_stats['std']}")
    
    # 保存动作统计（用于归一化）
    with open(output_dir / 'action_stats.json', 'w') as f:
        json.dump({
            'mean': action_stats['mean'].tolist(),
            'std': action_stats['std'].tolist()
        }, f, indent=2)


def compute_action_statistics(samples):
    """计算动作统计信息"""
    actions = np.array([s['action'] for s in samples])
    return {
        'mean': actions.mean(axis=0),
        'std': actions.std(axis=0),
        'min': actions.min(axis=0),
        'max': actions.max(axis=0)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='预处理训练数据')
    parser.add_argument('--input', type=str, required=True,
                        help='输入目录（collect_expert_data.py的输出）')
    parser.add_argument('--output', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='训练集比例')
    
    args = parser.parse_args()
    
    prepare_training_data(args)
