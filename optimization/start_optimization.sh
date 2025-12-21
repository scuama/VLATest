#!/bin/bash
# 启动优化流程的脚本
# 用法: bash optimization/start_optimization.sh <task_type> [--skip-optimization]
# 示例: 
#   bash optimization/start_optimization.sh move                    # 正常流程
#   bash optimization/start_optimization.sh move --skip-optimization # 跳过优化，直接推理

# 检查参数
if [ -z "$1" ]; then
    echo "❌ 请指定任务类型"
    echo "用法: bash optimization/start_optimization.sh <task_type> [--skip-optimization]"
    echo "示例:"
    echo "  bash optimization/start_optimization.sh move"
    echo "  bash optimization/start_optimization.sh move --skip-optimization"
    exit 1
fi

TASK_TYPE=$1
EXTRA_ARGS="${@:2}"  # 获取第二个参数开始的所有参数
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
if [[ "$EXTRA_ARGS" == *"--skip-optimization"* ]]; then
    echo "启动推理流程 - 后台运行（跳过优化）"
else
    echo "启动优化流程 - 后台运行"
fi
echo "=========================================="
echo "任务类型: $TASK_TYPE"
echo "参数: $EXTRA_ARGS"
echo "Python: $VENV_PYTHON"
echo "日志文件: $LOG_FILE"
echo ""

# 使用 nohup 后台运行（-u 参数启用无缓冲模式，确保日志实时输出）
nohup "$VENV_PYTHON" -u optimization/run_optimization.py "$TASK_TYPE" $EXTRA_ARGS > "$LOG_FILE" 2>&1 &
PID=$!

if [[ "$EXTRA_ARGS" == *"--skip-optimization"* ]]; then
    echo "✅ 推理流程已启动（跳过优化阶段）"
else
    echo "✅ 优化流程已启动"
fi
echo "   进程 ID: $PID"
echo "   日志文件: $LOG_FILE"
echo ""
echo "查看日志: tail -f $LOG_FILE"
echo "查看进度: watch -n 2 'ls -1 optimization/$TASK_TYPE/results/ | wc -l'"
echo "查看进程: ps -p $PID"
echo "停止进程: kill $PID"
echo "=========================================="
