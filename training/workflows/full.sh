#!/bin/bash
# OpenVLA全量训练脚本（后台运行）

set -e

PROJECT_DIR="/mnt/disk1/decom/VLATest"
cd $PROJECT_DIR

# 停用conda环境并激活.venv虚拟环境
echo "切换到.venv虚拟环境..."
conda deactivate 2>/dev/null || true
source .venv/bin/activate
echo "当前Python: $(which python)"
python --version

echo "=========================================="
echo "OpenVLA Grasp任务 - 全量训练"
echo "=========================================="

# 阶段1: 数据收集（后台运行）
echo ""
echo "阶段1: 收集全量数据（343个成功案例）"
echo "这将在screen会话中后台运行..."
echo ""

screen -dmS collect_full bash -c "
    cd $PROJECT_DIR
    PYTHONPATH=$PROJECT_DIR python3 experiments/collect_expert_data.py \
        --task grasp \
        --model rt_1_400k \
        --num_samples -1 \
        --output data/training/grasp_full \
        2>&1 | tee data/training/grasp_full_collection.log
    echo '数据收集完成！' >> data/training/grasp_full_collection.log
"

echo "✓ 数据收集已在后台启动"
echo "  查看进度: screen -r collect_full"
echo "  查看日志: tail -f data/training/grasp_full_collection.log"
echo ""
echo "等待数据收集完成后，运行以下命令继续："
echo ""
echo "# 预处理数据"
echo "python experiments/prepare_training_data.py \\"
echo "    --input data/training/grasp_full \\"
echo "    --output data/training/grasp_full_processed"
echo ""
echo "# 开始训练（后台）"
echo "screen -dmS train_full bash -c \\"
echo "    'source .venv/bin/activate && PYTHONPATH=$PROJECT_DIR python experiments/finetune_openvla_lora.py \\"
echo "        --data_dir data/training/grasp_full_processed \\"
echo "        --output_dir checkpoints/openvla_grasp_lora \\"
echo "        --num_epochs 10 \\"
echo "        --batch_size 8 \\"
echo "        --gradient_accumulation_steps 4 \\"
echo "        --run_name grasp_full_lora \\"
echo "        2>&1 | tee checkpoints/openvla_grasp_lora/training.log'"
