#!/bin/bash
################################################################################
# 批量推理便捷脚本
# 用途：快速使用nohup后台运行批量推理任务
################################################################################

# 默认配置
DEFAULT_BASE_CONFIG_DIR="results/t-move_n-100_o-0_s-3225323079/openvla-7b_2024"
DEFAULT_OUTPUT_DIR="inference_results"
DEFAULT_MODEL="openvla-7b"
DEFAULT_TASK="move"
DEFAULT_MAX_STEPS=""

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示帮助信息
show_help() {
    cat << EOF
📋 批量推理便捷脚本

用途: 快速启动后台推理任务，自动使用nohup记录日志

用法:
  $0 <episode_ids> [options]

参数:
  episode_ids          场景序号，多个用空格分隔，例如: 7 或 "7 13 15"

选项:
  -b, --base_dir DIR   基础配置目录（默认: $DEFAULT_BASE_CONFIG_DIR）
  -o, --output DIR     输出根目录（默认: $DEFAULT_OUTPUT_DIR）
  -m, --model MODEL    模型名称（默认: $DEFAULT_MODEL）
  -t, --task TASK      任务类型: move 或 grasp（默认: $DEFAULT_TASK）
  -l, --lora PATH      LoRA路径
  --max_steps STEPS    最大步数（默认: 使用配置文件中的值）
  --no_images          不保存图像
  -h, --help           显示此帮助信息

示例:
  # 最简使用 - 只指定场景序号
  $0 7
  $0 7 13 15
  
  # 指定最大步数
  $0 7 --max_steps 100
  
  # 指定不同的基础配置目录
  $0 7 -b results/t-grasp_n-100_o-0_s-123456/openvla-7b_2024
  
  # 使用grasp任务
  $0 7 13 -t grasp
  
  # 使用LoRA微调模型
  $0 7 -l /path/to/lora

日志位置:
  logs/inference_ep<episodes>_<timestamp>.log

EOF
}

# 解析参数
EPISODES=()
BASE_DIR="$DEFAULT_BASE_CONFIG_DIR"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
MODEL="$DEFAULT_MODEL"
TASK="$DEFAULT_TASK"
LORA_PATH=""
MAX_STEPS=""
NO_IMAGES=""

# 解析episode IDs和选项
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -b|--base_dir)
            BASE_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -t|--task)
            TASK="$2"
            shift 2
            ;;
        -l|--lora)
            LORA_PATH="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --no_images)
            NO_IMAGES="--no_images"
            shift
            ;;
        -*)
            echo -e "${RED}❌ 未知选项: $1${NC}"
            echo "使用 -h 查看帮助信息"
            exit 1
            ;;
        *)
            # 收集所有数字作为episode IDs
            EPISODES+=("$1")
            shift
            ;;
    esac
done

# 检查是否提供了episode IDs
if [ ${#EPISODES[@]} -eq 0 ]; then
    echo -e "${RED}❌ 错误: 必须提供至少一个场景序号${NC}"
    echo ""
    show_help
    exit 1
fi

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 创建logs目录
LOGS_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOGS_DIR"

# 生成日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EPISODES_STR=$(IFS=_; echo "${EPISODES[*]}")
LOG_FILE="$LOGS_DIR/inference_ep${EPISODES_STR}_${TIMESTAMP}.log"

# 构建Python命令
PYTHON_CMD="python3 $SCRIPT_DIR/run_full_inference_with_config.py"
PYTHON_CMD="$PYTHON_CMD --episodes ${EPISODES[*]}"
PYTHON_CMD="$PYTHON_CMD --base_config_dir $BASE_DIR"
PYTHON_CMD="$PYTHON_CMD --output $OUTPUT_DIR"
PYTHON_CMD="$PYTHON_CMD --model $MODEL"
PYTHON_CMD="$PYTHON_CMD --task $TASK"

if [ -n "$LORA_PATH" ]; then
    PYTHON_CMD="$PYTHON_CMD --lora_path $LORA_PATH"
fi

if [ -n "$MAX_STEPS" ]; then
    PYTHON_CMD="$PYTHON_CMD --max_steps $MAX_STEPS"
fi

if [ -n "$NO_IMAGES" ]; then
    PYTHON_CMD="$PYTHON_CMD $NO_IMAGES"
fi

# 显示配置信息
echo -e "${BLUE}════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🚀 启动批量推理任务${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════${NC}"
echo -e "场景序号: ${YELLOW}${EPISODES[*]}${NC}"
echo -e "基础配置: ${YELLOW}$BASE_DIR${NC}"
echo -e "输出目录: ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "模型: ${YELLOW}$MODEL${NC}"
echo -e "任务类型: ${YELLOW}$TASK${NC}"
if [ -n "$MAX_STEPS" ]; then
    echo -e "最大步数: ${YELLOW}$MAX_STEPS${NC}"
fi
if [ -n "$LORA_PATH" ]; then
    echo -e "LoRA路径: ${YELLOW}$LORA_PATH${NC}"
fi
echo -e "日志文件: ${YELLOW}$LOG_FILE${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════${NC}"

# 启动后台任务
cd "$PROJECT_DIR" || exit 1
nohup $PYTHON_CMD > "$LOG_FILE" 2>&1 &
PID=$!

# 等待一秒检查进程是否启动成功
sleep 1
if ps -p $PID > /dev/null; then
    echo -e "${GREEN}✅ 任务已启动${NC}"
    echo -e "进程ID: ${YELLOW}$PID${NC}"
    echo ""
    echo -e "${BLUE}💡 有用的命令:${NC}"
    echo -e "  查看实时日志: ${YELLOW}tail -f $LOG_FILE${NC}"
    echo -e "  查看进程状态: ${YELLOW}ps -p $PID${NC}"
    echo -e "  终止任务: ${YELLOW}kill $PID${NC}"
    echo ""
    echo -e "${GREEN}🎉 推理任务正在后台运行...${NC}"
else
    echo -e "${RED}❌ 任务启动失败，请检查日志: $LOG_FILE${NC}"
    exit 1
fi
