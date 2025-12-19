#!/bin/bash
# 启动优化流程的脚本
# 用法: bash optimization/start_optimization.sh <task_type>
# 示例: bash optimization/start_optimization.sh move

# 检查参数
if [ -z "$1" ]; then
    echo "❌ 请指定任务类型"
    echo "用法: bash optimization/start_optimization.sh <task_type>"
    echo "示例: bash optimization/start_optimization.sh move"
    exit 1
fi

TASK_TYPE=$1
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="optimization/log"
LOG_FILE="${LOG_DIR}/optimization_${TASK_TYPE}_${TIMESTAMP}.log"

# 项目根目录和虚拟环境Python
PROJECT_ROOT="/mnt/disk1/decom/VLATest"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python3"

# 检查虚拟环境是否存在
if [ ! -f "$VENV_PYTHON" ]; then
    echo "⚠️  虚拟环境不存在，使用系统 Python"
    VENV_PYTHON="python3"
fi

# 创建日志目录
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "启动优化流程 - 后台运行"
echo "=========================================="
echo "任务类型: $TASK_TYPE"
echo "Python: $VENV_PYTHON"
echo "日志文件: $LOG_FILE"
echo ""

# 使用 nohup 后台运行（-u 参数启用无缓冲模式，确保日志实时输出）
nohup "$VENV_PYTHON" -u optimization/run_optimization.py "$TASK_TYPE" > "$LOG_FILE" 2>&1 &
PID=$!

echo "✅ 优化流程已启动"
echo "   进程 ID: $PID"
echo "   日志文件: $LOG_FILE"
echo ""
echo "查看日志: tail -f $LOG_FILE"
echo "查看进程: ps -p $PID"
echo "停止进程: kill $PID"
echo "=========================================="
