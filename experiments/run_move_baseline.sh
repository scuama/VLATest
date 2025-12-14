#!/bin/bash

# 运行无变异的 move 数据集，使用微调前的 OpenVLA 模型
# 最大步数：50

PROJECT_ROOT="/mnt/disk1/decom/VLATest"

# 参数设置
MODEL="openvla-7b"
DATA_PATH="${PROJECT_ROOT}/data/t-move_n-100_o-0_s-3225323079.json"
OUTPUT_ROOT="${PROJECT_ROOT}/results"
SEED=2024

echo "=========================================="
echo "Running baseline OpenVLA on move dataset"
echo "Model: ${MODEL}"
echo "Data: ${DATA_PATH}"
echo "Max steps: 50 (set in openVLA.py)"
echo "Output: ${OUTPUT_ROOT}"
echo "=========================================="

cd "${PROJECT_ROOT}/experiments"

# 直接运行 Python 脚本
PYTHONPATH=${PROJECT_ROOT} python3 openVLA.py \
    -s ${SEED} \
    -m "${MODEL}" \
    -d "${DATA_PATH}" \
    -o "${OUTPUT_ROOT}/" \
    -r True

echo "=========================================="
echo "Execution completed!"
echo "=========================================="
