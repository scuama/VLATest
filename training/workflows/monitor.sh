#!/bin/bash
# 监控训练进度脚本

echo "=========================================="
echo "OpenVLA微调进度监控"
echo "=========================================="
echo ""

# 检查运行中的进程
echo "1. 运行中的任务:"
ps aux | grep -E "(collect_expert|prepare_training|finetune_openvla)" | grep -v grep | awk '{print "  PID:", $2, "CMD:", $11, $12, $13}'
echo ""

# 显示最新日志
echo "2. 最新日志 (最后30行):"
LATEST_LOG=$(ls -t training/logs/quick_test_*.log 2>/dev/null | head -1)
if [ -f "$LATEST_LOG" ]; then
    echo "  文件: $LATEST_LOG"
    echo "  ----------------------------------------"
    tail -30 "$LATEST_LOG"
else
    echo "  未找到日志文件"
fi
echo ""

# 检查数据收集进度
echo "3. 数据收集状态:"
if [ -d "training/data/grasp_test" ]; then
    SCENE_COUNT=$(ls -d training/data/grasp_test/scene_* 2>/dev/null | wc -l)
    echo "  已收集场景数: $SCENE_COUNT"
    
    if [ -f "training/data/grasp_test/trajectories.pkl" ]; then
        echo "  ✓ 轨迹数据已保存"
    fi
fi
echo ""

# 检查训练进度
echo "4. 训练状态:"
if [ -d "training/checkpoints/openvla_grasp_test" ]; then
    echo "  检查点目录已创建"
    ls -lh training/checkpoints/openvla_grasp_test/ 2>/dev/null | tail -5
fi
echo ""

echo "=========================================="
echo "快捷命令:"
echo "  查看完整日志: tail -f $LATEST_LOG"
echo "  停止任务: pkill -f quick_start_finetuning"
echo "=========================================="
