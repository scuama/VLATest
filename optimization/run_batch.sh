#!/bin/bash
# 批量优化运行脚本

cd /mnt/disk1/decom/VLATest

LOG_FILE="optimization/log/run_optimization_$(date +%Y%m%d_%H%M%S).log"

echo "开始运行优化流程..."
echo "日志文件: $LOG_FILE"

nohup python3 optimization/run_optimization.py move > "$LOG_FILE" 2>&1 &

PID=$!
echo "后台任务已启动，PID: $PID"
echo "使用以下命令查看日志："
echo "  tail -f $LOG_FILE"
echo "使用以下命令检查进程："
echo "  ps aux | grep $PID"
