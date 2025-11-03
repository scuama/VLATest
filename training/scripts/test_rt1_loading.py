#!/usr/bin/env python3
"""
测试RT-1模型加载，诊断TensorFlow GPU问题
"""

import os
import sys

# 设置环境变量（在导入TensorFlow之前）
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

print("=" * 60)
print("RT-1模型加载测试")
print("=" * 60)

# 1. 检查TensorFlow
print("\n1. 检查TensorFlow...")
import tensorflow as tf
print(f"   TensorFlow版本: {tf.__version__}")
print(f"   GPU设备: {tf.config.list_physical_devices('GPU')}")

# 2. 配置GPU内存增长
print("\n2. 配置GPU内存...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"   ✓ 已启用GPU内存增长")
    except RuntimeError as e:
        print(f"   ✗ GPU配置失败: {e}")

# 3. 测试简单的GPU操作
print("\n3. 测试GPU操作...")
try:
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
    print(f"   ✓ GPU计算成功: {c.numpy()}")
except Exception as e:
    print(f"   ✗ GPU计算失败: {e}")

# 4. 尝试加载RT-1模型
print("\n4. 加载RT-1模型...")
try:
    sys.path.insert(0, '/mnt/disk1/decom/VLATest')
    from simpler_env.policies.rt1.rt1_model import RT1Inference
    
    ckpt_path = "/mnt/disk1/decom/VLATest/checkpoints/rt_1_x_tf_trained_for_002272480_step"
    policy_setup = "google_robot"
    
    print(f"   模型路径: {ckpt_path}")
    print(f"   策略设置: {policy_setup}")
    print(f"   开始加载...")
    
    model = RT1Inference(
        saved_model_path=ckpt_path,
        policy_setup=policy_setup
    )
    
    print(f"   ✓ RT-1模型加载成功！")
    
except Exception as e:
    print(f"   ✗ RT-1模型加载失败:")
    print(f"   错误类型: {type(e).__name__}")
    print(f"   错误信息: {str(e)}")
    
    import traceback
    print("\n   完整错误堆栈:")
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
