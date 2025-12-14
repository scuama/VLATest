#!/bin/bash
# ==============================================
# 一键执行 VLA run_openVLA.sh + 自动恢复
# Author: Gong
# ==============================================

# ---------------- 配置 ----------------
DATA_JSON="../data/t-grasp_n-1_o-0_s-170912623-0.json"
RESULT_ROOT="../newresult/t-grasp_n-1_o-0_s-170912623-0"
MODEL_NAME="openvla-7b"
SEED=$(date +%s)
RUN_SCRIPT="./run_openVLA.sh"
# ---------------------------------------

echo "=============================="
echo "🔥 开始生成 episode"
echo "数据: $DATA_JSON"
echo "输出目录: $RESULT_ROOT"
echo "模型: $MODEL_NAME"
echo "随机种子: $SEED"
echo "=============================="

# Step 1: 调用 run_openVLA.sh
$RUN_SCRIPT "$MODEL_NAME" "$DATA_JSON" "$RESULT_ROOT"

if [ $? -ne 0 ]; then
    echo "❌ run_openVLA.sh 执行失败，请检查"
    exit 1
fi

echo "✅ episode 生成完成"

# Step 2: 自动恢复 replay
echo "=============================="
echo "🔥 开始自动恢复 Replay + 微调 + 重跑"
echo "=============================="

python3 ./rerun.py \
    "$RESULT_ROOT" \
    "$DATA_JSON"

if [ $? -ne 0 ]; then
    echo "❌ 自动恢复脚本执行失败，请检查"
    exit 1
fi

echo "🎉 全部完成！"
echo "成功 episode 保存在 successresult/"
echo "失败 episode 保存在 failresult/"
echo "summary_live.json 已更新"
