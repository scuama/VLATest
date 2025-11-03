"""
测试微调模型加载
验证LoRA模型是否可以正常加载和推理
"""
import sys
from pathlib import Path
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from PIL import Image
import numpy as np

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))


def test_model_loading():
    """测试模型加载"""
    
    # 配置
    base_model_path = "openvla/openvla-7b"
    lora_model_path = str(Path(__file__).parent.parent / "checkpoints" / "openvla_grasp_test" / "best_model")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("测试微调模型加载")
    print("=" * 60)
    
    # 1. 加载processor
    print("\n[1/4] 加载processor...")
    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    print("✓ Processor加载成功")
    
    # 2. 加载基础模型
    print("\n[2/4] 加载基础模型...")
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device,
        attn_implementation="eager"
    )
    print("✓ 基础模型加载成功")
    
    # 3. 加载LoRA适配器
    print("\n[3/4] 加载LoRA适配器...")
    model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        torch_dtype=torch.bfloat16
    )
    model.eval()
    print("✓ LoRA适配器加载成功")
    
    # 4. 测试推理
    print("\n[4/4] 测试推理...")
    
    # 创建测试图像（随机图像）
    test_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    test_instruction = "pick up the object"
    
    # 处理输入
    inputs = processor(
        text=[test_instruction],
        images=[test_image],
        return_tensors="pt"
    )
    
    # 移动到设备并转换类型
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
             for k, v in inputs.items()}
    if 'pixel_values' in inputs:
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
    
    # 推理
    with torch.no_grad():
        try:
            # 使用bridge_orig数据集的统计数据（OpenVLA常用的数据集）
            action = model.predict_action(
                input_ids=inputs['input_ids'],
                unnorm_key="bridge_orig"  # 指定归一化统计数据
            )
            print(f"✓ 推理成功!")
            print(f"  - 输出动作维度: {action.shape}")
            print(f"  - 动作值范围: [{action.min():.4f}, {action.max():.4f}]")
            print(f"  - 动作值: {action}")
        except Exception as e:
            print(f"✗ 推理失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！微调模型可以正常使用")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    success = test_model_loading()
    sys.exit(0 if success else 1)
