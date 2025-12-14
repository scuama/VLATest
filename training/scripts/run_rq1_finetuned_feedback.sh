#!/bin/bash

################################################################################
# RQ1 微调模型评估脚本 - 带提示词反馈版本
# 
# 功能：评估微调后的 OpenVLA 模型，使用提示词反馈机制
# 作者：基于 run_rq1_finetuned.sh 修改
# 日期：2025-11-05
################################################################################

set -euo pipefail

# ============================================================================
# 配置部分
# ============================================================================

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 基础模型
BASE_MODEL="openvla-7b"

# LoRA 路径（微调后的权重）
LORA_PATH="${PROJECT_ROOT}/models/openvla-7b-finetuned-libero-spatial-no-noops-lora"

# 数据目录
DATA_DIR="${PROJECT_ROOT}/data"

# 结果目录
RESULTS_DIR="${PROJECT_ROOT}/results"

# 日志目录
LOG_DIR="${PROJECT_ROOT}/training/logs"
mkdir -p "${LOG_DIR}"

# RQ1测试数据集
# 注意：为避免GPU显存不足，一次只运行一个数据集
# 完成后可以手动修改这里运行其他数据集
DATASETS=(
    "t-grasp_n-1000_o-m3_s-2498586606.json"
    # "t-move_n-1000_o-m3_s-2263834374.json"
    # "t-put-on_n-1000_o-m3_s-2593734741.json"
    # "t-put-in_n-1000_o-m3_s-2905191776.json"
)

# 随机种子
SEED=2024

# 超时时间（不使用超时限制，让任务自然完成）
# TIMEOUT_DURATION="2h"

# 反馈系统配置
ENABLE_STATIC_FEEDBACK=true      # 启用静态提示增强
ENABLE_DYNAMIC_FEEDBACK=true     # 启用动态提示调整
USE_RANDOM_TEMPLATES=true        # 随机选择模板
PRINT_SUMMARY=true               # 打印失败分析摘要

# ============================================================================
# 辅助函数
# ============================================================================

# 打印信息
print_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

# 打印成功
print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

# 打印警告
print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

# 打印错误
print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

# ============================================================================
# 主要函数
# ============================================================================

# 检查环境
check_environment() {
    print_info "检查环境..."
    
    # 检查 Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未安装"
        exit 1
    fi
    
    # 检查 CUDA
    if ! command -v nvidia-smi &> /dev/null; then
        print_warning "nvidia-smi 未找到，可能无法使用 GPU"
    else
        print_info "GPU 信息："
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    fi
    
    # 检查 LoRA 路径
    if [ ! -d "${LORA_PATH}" ]; then
        print_error "LoRA 路径不存在: ${LORA_PATH}"
        exit 1
    fi
    
    # 检查反馈模块
    if [ ! -f "${PROJECT_ROOT}/feedback/run_fuzzer_feedback.py" ]; then
        print_error "反馈模块未找到: ${PROJECT_ROOT}/feedback/run_fuzzer_feedback.py"
        exit 1
    fi
    
    print_success "环境检查通过"
}

# 运行评估
run_evaluation() {
    local dataset=$1
    local data_path="${DATA_DIR}/${dataset}"
    
    print_info "开始评估数据集: ${dataset}"
    
    # 检查数据文件
    if [ ! -f "${data_path}" ]; then
        print_error "数据文件不存在: ${data_path}"
        return 1
    fi
    
    # 构建反馈参数
    local feedback_args=""
    if [ "${ENABLE_STATIC_FEEDBACK}" = true ]; then
        feedback_args="${feedback_args} --enable-static"
    else
        feedback_args="${feedback_args} --disable-static"
    fi
    
    if [ "${ENABLE_DYNAMIC_FEEDBACK}" = true ]; then
        feedback_args="${feedback_args} --enable-dynamic"
    else
        feedback_args="${feedback_args} --disable-dynamic"
    fi
    
    if [ "${USE_RANDOM_TEMPLATES}" = true ]; then
        feedback_args="${feedback_args} --random-templates"
    fi
    
    if [ "${PRINT_SUMMARY}" = true ]; then
        feedback_args="${feedback_args} --print-summary"
    fi
    
    # 运行评估（不使用重试机制，避免GPU显存问题）
    print_info "开始运行评估任务..."
    
    # 运行Python脚本
    cd "${PROJECT_ROOT}/feedback"
    PYTHONPATH=${PROJECT_ROOT} python3 run_fuzzer_feedback.py \
        -s ${SEED} \
        -m ${BASE_MODEL} \
        -l ${LORA_PATH} \
        -d ${data_path} \
        -r True \
        ${feedback_args}
    
    # 检查退出状态
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        print_success "任务执行完成"
        return 0
    else
        print_error "任务失败（退出码: ${exit_code}）"
        return 1
    fi
}

# ============================================================================
# 主流程
# ============================================================================

main() {
    local start_time=$(date +%s)
    
    print_info "=========================================="
    print_info "RQ1 微调模型评估 - 带提示词反馈"
    print_info "=========================================="
    print_info "项目根目录: ${PROJECT_ROOT}"
    print_info "基础模型: ${BASE_MODEL}"
    print_info "LoRA 路径: ${LORA_PATH}"
    print_info "数据集数量: ${#DATASETS[@]}"
    print_info "反馈配置:"
    print_info "  - 静态增强: ${ENABLE_STATIC_FEEDBACK}"
    print_info "  - 动态调整: ${ENABLE_DYNAMIC_FEEDBACK}"
    print_info "  - 随机模板: ${USE_RANDOM_TEMPLATES}"
    print_info "=========================================="
    
    # 检查环境
    check_environment
    
    # 运行评估
    local failed_count=0
    for dataset in "${DATASETS[@]}"; do
        if ! run_evaluation "$dataset"; then
            ((failed_count++))
        fi
    done
    
    # 计算总时间
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    # 打印总结
    print_info "=========================================="
    print_info "评估完成"
    print_info "=========================================="
    print_info "总数据集: ${#DATASETS[@]}"
    print_info "成功: $((${#DATASETS[@]} - failed_count))"
    print_info "失败: ${failed_count}"
    print_info "总耗时: ${hours}h ${minutes}m ${seconds}s"
    print_info "=========================================="
    
    if [ $failed_count -eq 0 ]; then
        print_success "所有评估任务成功完成！"
        exit 0
    else
        print_warning "有 ${failed_count} 个任务失败"
        exit 1
    fi
}

# 运行主函数
main "$@"
