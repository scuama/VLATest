"""
使用微调后的OpenVLA模型运行评估
支持加载LoRA微调模型并在测试场景中运行
"""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from PIL import Image

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict


class FinetunedOpenVLAInference:
    """微调后的OpenVLA推理接口"""
    
    def __init__(self, base_model_path, lora_model_path, device='cuda'):
        """
        初始化微调模型
        
        Args:
            base_model_path: 基础模型路径（openvla/openvla-7b 或本地路径）
            lora_model_path: LoRA适配器路径
            device: 运行设备
        """
        self.device = device
        print(f"加载基础模型: {base_model_path}")
        print(f"加载LoRA适配器: {lora_model_path}")
        
        # 加载processor
        self.processor = AutoProcessor.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        
        # 加载基础模型
        base_model = AutoModelForVision2Seq.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device,
            attn_implementation="eager"
        )
        
        # 加载LoRA适配器
        self.model = PeftModel.from_pretrained(
            base_model,
            lora_model_path,
            torch_dtype=torch.bfloat16
        )
        
        self.model.eval()
        print("✓ 模型加载完成")
    
    def predict_action(self, image, instruction):
        """
        预测动作
        
        Args:
            image: PIL Image或numpy array
            instruction: 语言指令
            
        Returns:
            action: 7维动作向量 [x, y, z, roll, pitch, yaw, gripper]
        """
        # 转换图像格式
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 处理输入
        inputs = self.processor(
            text=[instruction],
            images=[image],
            return_tensors="pt"
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # 转换为bfloat16
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
        
        # 推理
        with torch.no_grad():
            # 使用模型的predict_action方法
            # 使用bridge_orig数据集的统计数据（OpenVLA常用的数据集）
            action = self.model.predict_action(
                input_ids=inputs['input_ids'],
                unnorm_key="bridge_orig"  # 指定归一化统计数据
            )
        
        return action


def run_evaluation(args):
    """运行评估"""
    
    # 加载微调模型
    model = FinetunedOpenVLAInference(
        base_model_path=args.base_model,
        lora_model_path=args.lora_model,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 加载测试数据
    print(f"\n加载测试数据: {args.test_data}")
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)
    
    # 限制测试样本数
    if args.num_samples > 0:
        test_data = test_data[:args.num_samples]
    
    print(f"测试样本数: {len(test_data)}")
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行评估
    results = []
    success_count = 0
    
    print("\n开始评估...")
    for i, sample in enumerate(tqdm(test_data)):
        try:
            # 这里需要根据实际的测试数据格式进行调整
            # 假设test_data包含场景信息
            
            # TODO: 实际运行仿真环境
            # 1. 初始化环境
            # 2. 获取观察
            # 3. 使用模型预测动作
            # 4. 执行动作
            # 5. 判断成功/失败
            
            result = {
                'sample_id': i,
                'success': False,  # 待实现
                'steps': 0,
                'error': None
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"\n样本 {i} 失败: {e}")
            results.append({
                'sample_id': i,
                'success': False,
                'error': str(e)
            })
    
    # 计算统计信息
    success_count = sum(1 for r in results if r['success'])
    success_rate = success_count / len(results) if results else 0
    
    # 保存结果
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'model': args.lora_model,
            'test_data': args.test_data,
            'num_samples': len(results),
            'success_count': success_count,
            'success_rate': success_rate,
            'results': results
        }, f, indent=2)
    
    print(f"\n✓ 评估完成!")
    print(f"  - 成功率: {success_rate:.2%} ({success_count}/{len(results)})")
    print(f"  - 结果保存在: {results_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='评估微调后的OpenVLA模型')
    
    parser.add_argument('--base_model', type=str, 
                        default='openvla/openvla-7b',
                        help='基础模型路径')
    parser.add_argument('--lora_model', type=str, required=True,
                        help='LoRA模型路径')
    parser.add_argument('--test_data', type=str, required=True,
                        help='测试数据JSON文件')
    parser.add_argument('--output', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--num_samples', type=int, default=-1,
                        help='测试样本数（-1表示全部）')
    
    args = parser.parse_args()
    
    run_evaluation(args)
