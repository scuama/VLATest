#!/bin/bash
#
# RQ1实验脚本 - OpenVLA模型评估（支持预训练基线和微调模型）
# 评估4个基础任务的性能：grasp, move, put-on, put-in
#
# 使用方法:
#   # 运行预训练基线
#   bash run_rq1_finetuned.sh baseline
#   
#   # 运行微调模型
#   bash run_rq1_finetuned.sh finetuned
#   
#   # 后台运行
#   nohup bash run_rq1_finetuned.sh baseline > ../../training/logs/rq1_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#

set -e  # 遇到错误立即退出

# ==================== 配置区域 ====================

# 项目根目录
PROJECT_ROOT="/mnt/disk1/decom/VLATest"

# 运行模式：baseline（预训练）或 finetuned（微调）
RUN_MODE="${1:-finetuned}"  # 默认运行微调模型

# 微调模型路径（仅在finetuned模式下使用）
LORA_MODEL_PATH="${PROJECT_ROOT}/training/checkpoints/openvla_grasp_test/best_model"

# 基础模型名称
BASE_MODEL="openvla-7b"

# 随机种子
SEED=2024

# RQ1测试数据集
# 注意：为避免GPU显存不足，一次只运行一个数据集
# 完成后可以手动修改这里运行其他数据集
DATASETS=(
    "t-grasp_n-1000_o-m3_s-2498586606.json"
    # "t-move_n-1000_o-m3_s-2263834374.json"
    # "t-put-on_n-1000_o-m3_s-2593734741.json"
    # "t-put-in_n-1000_o-m3_s-2905191776.json"
)

# 超时时间（每个数据集）
TIMEOUT_DURATION="2h"

# 根据运行模式设置标识和LoRA路径
if [ "$RUN_MODE" = "baseline" ]; then
    MODEL_TAG="${BASE_MODEL}"
    LORA_ARG=""
    LORA_DISPLAY="无（预训练基线）"
elif [ "$RUN_MODE" = "finetuned" ]; then
    MODEL_TAG="${BASE_MODEL}_finetuned"
    LORA_ARG="-l ${LORA_MODEL_PATH}"
    LORA_DISPLAY="${LORA_MODEL_PATH}"
else
    echo "错误: 未知的运行模式 '$RUN_MODE'"
    echo "使用方法: bash run_rq1_finetuned.sh [baseline|finetuned]"
    exit 1
fi

# ==================== 函数定义 ====================

# 打印带颜色的消息
print_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

# 检查模型是否存在
check_model() {
    if [ "$RUN_MODE" = "finetuned" ]; then
        if [ ! -d "$LORA_MODEL_PATH" ]; then
            print_error "LoRA模型不存在: $LORA_MODEL_PATH"
            print_info "请先运行微调训练或修改LORA_MODEL_PATH变量"
            exit 1
        fi
        
        if [ ! -f "$LORA_MODEL_PATH/adapter_config.json" ]; then
            print_error "LoRA模型配置文件不存在: $LORA_MODEL_PATH/adapter_config.json"
            exit 1
        fi
        
        print_success "找到微调模型: $LORA_MODEL_PATH"
    else
        print_success "使用预训练基线模型: $BASE_MODEL"
    fi
}

# 检查数据集是否存在
check_datasets() {
    local missing_count=0
    for dataset in "${DATASETS[@]}"; do
        if [ ! -f "${PROJECT_ROOT}/data/${dataset}" ]; then
            print_warning "数据集不存在: ${dataset}"
            ((missing_count++))
        fi
    done
    
    if [ $missing_count -gt 0 ]; then
        print_error "缺少 ${missing_count} 个数据集文件"
        print_info "请确保数据集文件存在于 ${PROJECT_ROOT}/data/ 目录"
        exit 1
    fi
    
    print_success "所有数据集文件检查通过"
}

# 运行单个数据集的评估
run_evaluation() {
    local dataset=$1
    local data_path="${PROJECT_ROOT}/data/${dataset}"
    local dataset_name="${dataset%.json}"
    local output_dir="${PROJECT_ROOT}/results/${dataset_name}/${MODEL_TAG}_${SEED}"
    
    print_info "=========================================="
    print_info "开始评估: ${dataset}"
    print_info "输出目录: ${output_dir}"
    print_info "=========================================="
    
    # 检查是否已完成（有1000个log.json文件）
    if [ -d "$output_dir" ]; then
        local log_count=$(find "$output_dir" -name 'log.json' 2>/dev/null | wc -l)
        if [ "$log_count" -eq 1000 ]; then
            print_success "任务已完成，跳过: ${dataset} (${log_count}/1000)"
            return 0
        else
            print_warning "任务未完成，继续运行: ${dataset} (${log_count}/1000)"
        fi
    fi
    
    # 运行评估（不使用重试机制，避免GPU显存问题）
    print_info "开始运行评估任务..."
    
    # 运行Python脚本
    cd "${PROJECT_ROOT}/experiments"
    timeout "${TIMEOUT_DURATION}" bash -c "
        PYTHONPATH=${PROJECT_ROOT} python3 run_fuzzer.py \
            -s ${SEED} \
            -m ${BASE_MODEL} \
            ${LORA_ARG} \
            -d ${data_path} \
            -r True
    "
    
    # 检查退出状态
    local exit_code=$?
    if [ $exit_code -eq 124 ]; then
        print_error "任务超时（超过 ${TIMEOUT_DURATION}）"
        return 1
    elif [ $exit_code -ne 0 ]; then
        print_error "任务失败（退出码: ${exit_code}）"
        return 1
    fi
    
    print_success "任务执行完成"
    
    # 检查最终结果
    local final_log_count=$(find "$output_dir" -name 'log.json' 2>/dev/null | wc -l)
    if [ "$final_log_count" -eq 1000 ]; then
        print_success "任务完成: ${dataset} (${final_log_count}/1000)"
        return 0
    else
        print_error "任务未完成: ${dataset} (${final_log_count}/1000)"
        return 1
    fi
}

# 生成评估报告
generate_report() {
    print_info "=========================================="
    print_info "生成评估报告"
    print_info "=========================================="
    
    local report_file="${PROJECT_ROOT}/training/logs/rq1_${RUN_MODE}_report_${SEED}.txt"
    
    {
        echo "RQ1 OpenVLA评估报告 - ${RUN_MODE^^} 模式"
        echo "=========================================="
        echo ""
        echo "运行模式: ${RUN_MODE}"
        echo "模型标识: ${MODEL_TAG}"
        echo "LoRA路径: ${LORA_DISPLAY}"
        echo "随机种子: ${SEED}"
        echo "评估时间: $(date)"
        echo ""
        echo "任务完成情况:"
        echo "----------------------------------------"
        
        for dataset in "${DATASETS[@]}"; do
            local dataset_name="${dataset%.json}"
            local output_dir="${PROJECT_ROOT}/results/${dataset_name}/${MODEL_TAG}_${SEED}"
            
            if [ -d "$output_dir" ]; then
                local log_count=$(find "$output_dir" -name 'log.json' 2>/dev/null | wc -l)
                local success_count=$(grep -l '"success": true' "$output_dir"/*/log.json 2>/dev/null | wc -l)
                local success_rate=$(echo "scale=2; $success_count * 100 / $log_count" | bc)
                
                echo "任务: ${dataset_name}"
                echo "  - 完成样本: ${log_count}/1000"
                echo "  - 成功样本: ${success_count}"
                echo "  - 成功率: ${success_rate}%"
                echo ""
            else
                echo "任务: ${dataset_name}"
                echo "  - 状态: 未运行"
                echo ""
            fi
        done
        
        echo "----------------------------------------"
        echo "报告生成时间: $(date)"
    } > "$report_file"
    
    # 显示报告内容
    cat "$report_file"
    
    print_success "报告已保存: ${report_file}"
}

# ==================== 主程序 ====================

main() {
    print_info "=========================================="
    print_info "RQ1 OpenVLA评估 - ${RUN_MODE^^} 模式"
    print_info "=========================================="
    print_info "项目根目录: ${PROJECT_ROOT}"
    print_info "运行模式: ${RUN_MODE}"
    print_info "基础模型: ${BASE_MODEL}"
    print_info "LoRA路径: ${LORA_DISPLAY}"
    print_info "随机种子: ${SEED}"
    print_info "=========================================="
    
    # 检查环境
    print_info "检查环境..."
    check_model
    check_datasets
    
    # 激活虚拟环境（如果存在）
    if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
        print_info "激活虚拟环境..."
        source "${PROJECT_ROOT}/.venv/bin/activate"
    fi
    
    # 运行所有数据集
    local failed_count=0
    for dataset in "${DATASETS[@]}"; do
        if ! run_evaluation "$dataset"; then
            ((failed_count++))
        fi
    done
    
    # 生成报告
    generate_report
    
    # 总结
    print_info "=========================================="
    if [ $failed_count -eq 0 ]; then
        print_success "所有任务完成！"
    else
        print_warning "有 ${failed_count} 个任务未完成"
    fi
    print_info "=========================================="
    
    return $failed_count
}

# 运行主程序
main
exit $?
