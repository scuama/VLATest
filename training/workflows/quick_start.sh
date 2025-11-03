#!/bin/bash
# OpenVLA微调快速启动脚本

set -e  # 遇到错误立即退出

# 设置路径
PROJECT_DIR="/mnt/disk1/decom/VLATest"
cd $PROJECT_DIR

# 停用conda环境并激活.venv虚拟环境
echo "切换到.venv虚拟环境..."
conda deactivate 2>/dev/null || true
source .venv/bin/activate
echo "当前Python: $(which python)"
python --version

echo "=========================================="
echo "OpenVLA Grasp任务微调 - 快速启动"
echo "=========================================="

# 检查GPU
echo ""
echo "检查GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# 阶段1: 小规模测试（10个样本）
echo ""
echo "=========================================="
echo "阶段1: 小规模测试（10个成功案例）"
echo "=========================================="

echo ""
echo "步骤1: 收集测试数据（约5分钟）..."
PYTHONPATH=$PROJECT_DIR python training/scripts/collect_expert_data.py \
    --task grasp \
    --model rt_1_400k \
    --num_samples 10 \
    --output training/data/grasp_test

echo ""
echo "步骤2: 预处理数据..."
python training/scripts/prepare_training_data.py \
    --input training/data/grasp_test \
    --output training/data/grasp_test_processed

echo ""
echo "步骤3: 开始测试训练（3个epochs）..."
PYTHONPATH=$PROJECT_DIR python training/scripts/finetune_openvla_lora.py \
    --data_dir training/data/grasp_test_processed \
    --output_dir training/checkpoints/openvla_grasp_test \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --run_name "grasp_test_run"

echo ""
echo "✓ 测试流程完成！"
echo ""
echo "如果一切正常，可以开始全量训练："
echo "  bash full_training.sh"
