"""
OpenVLA LoRA微调脚本 - 针对RTX 3090优化
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import wandb

# 设置HuggingFace镜像源（中国境内加速）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class OpenVLADataset(Dataset):
    """OpenVLA训练数据集"""
    
    def __init__(self, data_dir, split='train', processor=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.processor = processor
        
        # 加载样本
        samples_file = self.data_dir / split / 'samples.json'
        with open(samples_file, 'r') as f:
            self.samples = json.load(f)
        
        # 加载元数据
        with open(self.data_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # 加载动作统计（用于归一化）
        with open(self.data_dir / 'action_stats.json', 'r') as f:
            stats = json.load(f)
            self.action_mean = np.array(stats['mean'], dtype=np.float32)
            self.action_std = np.array(stats['std'], dtype=np.float32)
        
        print(f"{split}集: {len(self.samples)}个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像
        image = Image.open(sample['image_path']).convert('RGB')
        
        # 加载动作并归一化
        action = np.array(sample['action'], dtype=np.float32)
        action_normalized = (action - self.action_mean) / (self.action_std + 1e-8)
        
        # 生成指令
        instruction = self._get_instruction(sample['task'])
        
        return {
            'image': image,
            'action': torch.from_numpy(action_normalized),
            'action_raw': torch.from_numpy(action),
            'instruction': instruction,
            'dataset_name': sample['dataset_name']
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


def collate_fn(batch, processor):
    """数据批处理函数"""
    images = [item['image'] for item in batch]
    instructions = [item['instruction'] for item in batch]
    actions = torch.stack([item['action'] for item in batch])
    
    # 使用processor处理图像和文本
    inputs = processor(
        text=instructions,
        images=images,
        return_tensors="pt",
        padding=True
    )
    
    # 确保所有输入都是bfloat16类型（与模型一致）
    if 'pixel_values' in inputs:
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
    
    inputs['labels'] = actions
    return inputs


def train_epoch(model, dataloader, optimizer, device, epoch, args):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # 移动数据到设备
        input_ids = batch['input_ids'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            return_dict=True
        )
        
        # 计算损失（使用最后一层的输出预测动作）
        # 注意：这里需要根据OpenVLA的实际架构调整
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs.last_hidden_state
        
        # 简单的MSE损失
        loss = nn.functional.mse_loss(logits[:, -1, :7], labels)
        
        # 梯度累积
        loss = loss / args.gradient_accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # 更新参数
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * args.gradient_accumulation_steps
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss.item() * args.gradient_accumulation_steps:.4f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
        })
        
        # 记录到wandb
        if args.use_wandb and batch_idx % args.log_interval == 0:
            wandb.log({
                'train/loss': loss.item() * args.gradient_accumulation_steps,
                'train/epoch': epoch,
                'train/step': epoch * len(dataloader) + batch_idx
            })
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="验证"):
            input_ids = batch['input_ids'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                return_dict=True
            )
            
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs.last_hidden_state
            loss = nn.functional.mse_loss(logits[:, -1, :7], labels)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main(args):
    """主训练函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化wandb
    if args.use_wandb:
        wandb.init(
            project="openvla-finetuning",
            name=args.run_name,
            config=vars(args)
        )
    
    # 加载processor
    print("\n加载processor...")
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        local_files_only=False  # 优先使用本地，如果没有则下载
    )
    
    # 加载模型
    print("加载模型...")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device,
        attn_implementation="eager",  # 使用eager模式避免SDPA兼容性问题
        local_files_only=False  # 优先使用本地，如果没有则下载
    )
    
    # 配置LoRA
    print("\n配置LoRA...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 加载数据集
    print("\n加载数据集...")
    train_dataset = OpenVLADataset(args.data_dir, split='train', processor=processor)
    val_dataset = OpenVLADataset(args.data_dir, split='val', processor=processor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn(batch, processor)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn(batch, processor)
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 训练循环
    print(f"\n开始训练...")
    print(f"训练样本: {len(train_dataset)}")
    print(f"验证样本: {len(val_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"有效batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Epochs: {args.num_epochs}")
    
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"{'='*50}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, args)
        print(f"训练损失: {train_loss:.4f}")
        
        # 验证
        val_loss = validate(model, val_loader, device)
        print(f"验证损失: {val_loss:.4f}")
        
        # 记录到wandb
        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/epoch_loss': train_loss,
                'val/loss': val_loss
            })
        
        # 保存检查点
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"✓ 保存最佳模型 (val_loss: {val_loss:.4f})")
            model.save_pretrained(output_dir / 'best_model')
            processor.save_pretrained(output_dir / 'best_model')
        
        # 定期保存
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_dir = output_dir / f'checkpoint-epoch-{epoch+1}'
            model.save_pretrained(checkpoint_dir)
            processor.save_pretrained(checkpoint_dir)
            print(f"✓ 保存检查点: {checkpoint_dir}")
    
    # 保存最终模型
    print(f"\n保存最终模型...")
    model.save_pretrained(output_dir / 'final_model')
    processor.save_pretrained(output_dir / 'final_model')
    
    # 保存训练日志
    with open(output_dir / 'training.log', 'w') as f:
        f.write(f"Best validation loss: {best_val_loss:.4f}\n")
        f.write(f"Final training loss: {train_loss:.4f}\n")
    
    print(f"\n✓ 训练完成!")
    print(f"  - 最佳验证损失: {best_val_loss:.4f}")
    print(f"  - 模型保存在: {output_dir}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenVLA LoRA微调')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True,
                        help='预处理后的数据目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='openvla/openvla-7b',
                        help='预训练模型名称')
    
    # LoRA参数
    parser.add_argument('--lora_rank', type=int, default=32,
                        help='LoRA秩')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批大小')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='梯度累积步数')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='权重衰减')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='梯度裁剪')
    
    # 其他参数
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--save_interval', type=int, default=2,
                        help='保存检查点间隔')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='日志记录间隔')
    parser.add_argument('--use_wandb', action='store_true',
                        help='使用wandb记录')
    parser.add_argument('--run_name', type=str, default='openvla_grasp_lora',
                        help='运行名称')
    
    args = parser.parse_args()
    
    main(args)
