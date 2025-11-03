#!/bin/bash
#
# 单任务测试脚本 - 快速测试微调模型
# 用于测试微调模型是否能正常运行
#
# 使用方法:
#   bash test_rq1_single.sh [task_type] [num_samples]
#   
# 示例:
#   bash test_rq1_single.sh grasp 10    # 测试10个grasp任务
#   bash test_rq1_single.sh move 50     # 测试50个move任务
#

# 配置
PROJECT_ROOT="/mnt/disk1/decom/VLATest"
LORA_MODEL_PATH="${PROJECT_ROOT}/training/checkpoints/openvla_grasp_test/best_model"
BASE_MODEL="openvla-7b"
SEED=2024

# 参数
TASK_TYPE=${1:-"grasp"}  # 默认grasp
NUM_SAMPLES=${2:-10}     # 默认10个样本

# 数据集映射
declare -A DATASET_MAP
DATASET_MAP["grasp"]="t-grasp_n-1000_o-m3_s-2498586606.json"
DATASET_MAP["move"]="t-move_n-1000_o-m3_s-2263834374.json"
DATASET_MAP["put-on"]="t-put-on_n-1000_o-m3_s-2593734741.json"
DATASET_MAP["put-in"]="t-put-in_n-1000_o-m3_s-2905191776.json"

# 检查任务类型
if [ -z "${DATASET_MAP[$TASK_TYPE]}" ]; then
    echo "错误: 未知的任务类型 '$TASK_TYPE'"
    echo "支持的任务类型: grasp, move, put-on, put-in"
    exit 1
fi

DATASET="${DATASET_MAP[$TASK_TYPE]}"
DATA_PATH="${PROJECT_ROOT}/data/${DATASET}"

echo "=========================================="
echo "微调模型单任务测试"
echo "=========================================="
echo "任务类型: ${TASK_TYPE}"
echo "数据集: ${DATASET}"
echo "测试样本数: ${NUM_SAMPLES}"
echo "微调模型: ${LORA_MODEL_PATH}"
echo "=========================================="

# 检查模型
if [ ! -d "$LORA_MODEL_PATH" ]; then
    echo "错误: 微调模型不存在: $LORA_MODEL_PATH"
    exit 1
fi

# 检查数据集
if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 数据集不存在: $DATA_PATH"
    exit 1
fi

# 激活虚拟环境
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    echo "激活虚拟环境..."
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

# 创建临时测试数据（只包含前N个样本）
TEMP_DATA="/tmp/test_${TASK_TYPE}_${NUM_SAMPLES}.json"
echo "创建临时测试数据: ${TEMP_DATA}"

python3 << EOF
import json

# 读取原始数据
with open('${DATA_PATH}', 'r') as f:
    data = json.load(f)

# 创建子集
subset = {'num': ${NUM_SAMPLES}}
for i in range(${NUM_SAMPLES}):
    subset[str(i)] = data[str(i)]

# 保存
with open('${TEMP_DATA}', 'w') as f:
    json.dump(subset, f, indent=2)

print(f"✓ 创建了包含 ${NUM_SAMPLES} 个样本的测试数据")
EOF

# 运行测试
echo ""
echo "开始运行测试..."
echo "=========================================="

cd "${PROJECT_ROOT}/experiments"

PYTHONPATH=${PROJECT_ROOT} python3 run_fuzzer.py \
    -s ${SEED} \
    -m ${BASE_MODEL} \
    -l ${LORA_MODEL_PATH} \
    -d ${TEMP_DATA} \
    -r False

# 检查结果
DATASET_NAME="test_${TASK_TYPE}_${NUM_SAMPLES}"
OUTPUT_DIR="${PROJECT_ROOT}/results/${DATASET_NAME}/${BASE_MODEL}_finetuned_${SEED}"

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="

if [ -d "$OUTPUT_DIR" ]; then
    LOG_COUNT=$(find "$OUTPUT_DIR" -name 'log.json' 2>/dev/null | wc -l)
    SUCCESS_COUNT=$(grep -l '"success": true' "$OUTPUT_DIR"/*/log.json 2>/dev/null | wc -l)
    
    echo "完成样本: ${LOG_COUNT}/${NUM_SAMPLES}"
    echo "成功样本: ${SUCCESS_COUNT}"
    
    if [ $LOG_COUNT -gt 0 ]; then
        SUCCESS_RATE=$(echo "scale=2; $SUCCESS_COUNT * 100 / $LOG_COUNT" | bc)
        echo "成功率: ${SUCCESS_RATE}%"
    fi
    
    echo ""
    echo "结果目录: ${OUTPUT_DIR}"
else
    echo "警告: 未找到输出目录"
fi

# 清理临时文件
rm -f ${TEMP_DATA}

echo "=========================================="
