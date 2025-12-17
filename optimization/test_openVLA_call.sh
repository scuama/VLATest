#!/bin/bash
# 测试 openVLA.py 脚本调用

cd /mnt/disk1/decom/VLATest

echo "========================================"
echo "测试 openVLA.py 推理"
echo "========================================"
echo "数据集: optimization/move/batch_move_dataset.json"
echo "输出目录: optimization/move/test_results/"
echo "========================================"

python3 experiments/openVLA.py \
  --data optimization/move/batch_move_dataset.json \
  --output optimization/move/test_results/ \
  --model openvla-7b

echo ""
echo "========================================"
echo "推理完成，检查结果..."
echo "========================================"

ls -lh optimization/move/test_results/
