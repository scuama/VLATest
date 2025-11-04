# OpenVLA微调训练

本目录包含OpenVLA模型微调的完整工作流，从数据收集到模型评估。

## 📁 目录结构

```
training/
├── README.md                           # 本文件
├── scripts/                            # 核心脚本
│   ├── collect_expert_data.py         # 数据收集
│   ├── prepare_training_data.py       # 数据预处理
│   ├── finetune_openvla_lora.py       # LoRA微调训练
│   ├── model_interface_with_actions.py # VLA接口（数据收集用）
│   ├── run_rq1_finetuned.sh           # RQ1实验评估
│   └── test_rq1_single.sh             # 单任务快速测试
├── workflows/                          # 自动化工作流
│   ├── quick_start.sh                 # 快速测试（10样本）
│   ├── full.sh                        # 全量训练（343样本）
│   └── monitor.sh                     # 进度监控
├── data/                               # 训练数据（运行时生成）
├── checkpoints/                        # 模型检查点（运行时生成）
└── logs/                               # 训练日志（运行时生成）
```

## 🚀 快速开始

### 方式1: 一键启动（推荐）

```bash
cd /mnt/disk1/decom/VLATest/training

# 快速测试（10个样本，约15分钟）
bash workflows/quick_start.sh

# 监控进度
bash workflows/monitor.sh

# 全量训练（343个样本，约3-4小时）
bash workflows/full.sh
```

### 方式2: 手动执行各步骤

```bash
cd /mnt/disk1/decom/VLATest

# 激活虚拟环境
conda deactivate 2>/dev/null || true
source .venv/bin/activate

# 步骤1: 收集专家数据
PYTHONPATH=$PWD python training/scripts/collect_expert_data.py \
    --task grasp \
    --model rt_1_400k \
    --num_samples 10 \
    --output training/data/grasp_test

# 步骤2: 预处理数据
python training/scripts/prepare_training_data.py \
    --input training/data/grasp_test \
    --output training/data/grasp_test_processed

# 步骤3: LoRA微调
PYTHONPATH=$PWD python training/scripts/finetune_openvla_lora.py \
    --data_dir training/data/grasp_test_processed \
    --output_dir training/checkpoints/openvla_grasp_test \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --run_name "grasp_test_run"

# 步骤4: 评估微调模型（RQ1实验）
cd training/scripts
bash run_rq1_finetuned.sh
```

## 📊 完整工作流

### 1. 数据收集阶段

使用RT-1模型收集成功的专家轨迹：

```bash
python training/scripts/collect_expert_data.py \
    --task grasp \              # 任务类型: grasp/move/put-on/put-in
    --model rt_1_400k \         # 专家模型
    --num_samples 343 \         # 样本数（-1表示全部）
    --output training/data/grasp_full
```

**输出**:
- `training/data/grasp_full/scene_*/` - 每个成功场景的数据
- `training/data/grasp_full/trajectories.pkl` - 轨迹数据

### 2. 数据预处理阶段

转换为OpenVLA训练格式：

```bash
python training/scripts/prepare_training_data.py \
    --input training/data/grasp_full \
    --output training/data/grasp_full_processed
```

**输出**:
- `training/data/grasp_full_processed/data.json` - 训练数据
- `training/data/grasp_full_processed/images/` - 图像文件

### 3. LoRA微调阶段

使用LoRA方法微调OpenVLA-7B：

```bash
PYTHONPATH=$PWD python training/scripts/finetune_openvla_lora.py \
    --data_dir training/data/grasp_full_processed \
    --output_dir training/checkpoints/openvla_grasp_lora \
    --num_epochs 10 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-4 \
    --lora_r 32 \
    --lora_alpha 16 \
    --run_name "grasp_full_lora"
```

**输出**:
- `training/checkpoints/openvla_grasp_lora/best_model/` - 最佳模型
- `training/checkpoints/openvla_grasp_lora/final_model/` - 最终模型
- `training/checkpoints/openvla_grasp_lora/training.log` - 训练日志

### 4. 模型评估阶段

#### 4.1 快速测试（单任务）

```bash
cd training/scripts
bash test_rq1_single.sh grasp 10    # 测试10个grasp任务
bash test_rq1_single.sh move 50     # 测试50个move任务
```

#### 4.2 完整RQ1评估

评估4个基础任务（grasp, move, put-on, put-in），每个1000个样本：

```bash
cd training/scripts

# Baseline评估（预训练模型）
bash run_rq1_finetuned.sh baseline
# 或后台运行
nohup bash run_rq1_finetuned.sh baseline > ../logs/rq1_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Finetuned评估（微调模型）
bash run_rq1_finetuned.sh finetuned
# 或后台运行
nohup bash run_rq1_finetuned.sh finetuned > ../logs/rq1_finetuned_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**输出**:
- Baseline结果: `results/t-{task}_*/openvla-7b_2024/`
- Finetuned结果: `results/t-{task}_*/openvla-7b_finetuned_2024/`
- 评估报告: `training/logs/rq1_{baseline|finetuned}_report_2024.txt`

## 🔧 核心脚本说明

### 数据收集脚本

**`collect_expert_data.py`** - 使用专家模型收集成功轨迹

参数：
- `--task`: 任务类型（grasp/move/put-on/put-in）
- `--model`: 专家模型（rt_1_400k推荐）
- `--num_samples`: 目标样本数（-1=全部）
- `--output`: 输出目录

### 数据预处理脚本

**`prepare_training_data.py`** - 转换为OpenVLA训练格式

参数：
- `--input`: 原始数据目录
- `--output`: 处理后数据目录

### 微调训练脚本

**`finetune_openvla_lora.py`** - LoRA微调OpenVLA

关键参数：
- `--data_dir`: 训练数据目录
- `--output_dir`: 模型输出目录
- `--num_epochs`: 训练轮数（推荐3-10）
- `--batch_size`: 批次大小（根据GPU调整）
- `--lora_r`: LoRA秩（推荐32）
- `--learning_rate`: 学习率（推荐5e-4）

### 评估脚本

**`run_rq1_finetuned.sh`** - RQ1完整评估（支持baseline和finetuned）

使用方式：
```bash
bash run_rq1_finetuned.sh baseline    # 评估预训练模型
bash run_rq1_finetuned.sh finetuned   # 评估微调模型
```

功能：
- 自动评估4个基础任务（grasp, move, put-on, put-in）
- 支持断点续传
- 生成评估报告
- 结果自动区分baseline/finetuned

**`test_rq1_single.sh`** - 单任务快速测试
- 用于开发调试
- 支持自定义样本数

## 📈 监控与调试

### 查看训练进度

```bash
# 使用监控脚本
bash workflows/monitor.sh

# 或手动查看日志
tail -f training/checkpoints/openvla_grasp_lora/training.log

# 查看GPU使用
watch -n 1 nvidia-smi
```

### 查看评估进度

```bash
# 查看完成的样本数
find results/t-grasp_*/openvla-7b_finetuned_2024 -name 'log.json' | wc -l

# 查看成功率
grep -l '"success": true' results/t-grasp_*/openvla-7b_finetuned_2024/*/log.json | wc -l
```

## 📦 模型说明

### Baseline模型（预训练）
- **来源**: HuggingFace Hub (`openvla/openvla-7b`)
- **本地缓存**: `~/.cache/huggingface/hub/models--openvla--openvla-7b/`
- **大小**: 15GB（首次运行自动下载）
- **加载方式**: `AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b")`

### Finetuned模型（微调）
- **基础模型**: 同上（openvla/openvla-7b）
- **LoRA适配器**: `training/checkpoints/openvla_grasp_test/best_model/`
- **大小**: ~100MB（仅LoRA权重）
- **加载方式**: 基础模型 + `PeftModel.from_pretrained(base_model, lora_path)`

### 运行对比
```bash
# Baseline评估（预训练模型）
bash scripts/run_rq1_finetuned.sh baseline

# Finetuned评估（微调模型）
bash scripts/run_rq1_finetuned.sh finetuned
```

**结果目录区分**:
- Baseline: `results/t-{task}_*/openvla-7b_2024/`
- Finetuned: `results/t-{task}_*/openvla-7b_finetuned_2024/`

## ⚙️ 环境要求

- **Python**: 3.10 (.venv虚拟环境)
- **CUDA**: 12.0+
- **GPU**: RTX 3090 24GB（推荐）
- **磁盘**: 至少50GB可用空间（含模型缓存）
- **依赖**: 已通过 `requirements_full_install.txt` 安装

## 💡 最佳实践

1. **数据收集**: 先用小样本（10个）测试流程，确认无误后再收集全量数据
2. **训练**: 从3个epochs开始，观察loss曲线，避免过拟合
3. **评估**: 使用 `test_rq1_single.sh` 快速验证模型，再运行完整评估
4. **资源**: 训练时关闭其他GPU程序，确保显存充足

## 🐛 常见问题

**Q: 数据收集很慢怎么办？**
A: 正常现象，RT-1模型推理较慢。可以先用10个样本测试流程。

**Q: 训练时显存不足？**
A: 减小 `batch_size` 或增加 `gradient_accumulation_steps`。

**Q: 如何修改微调的任务？**
A: 修改 `collect_expert_data.py` 的 `--task` 参数，支持 grasp/move/put-on/put-in。

**Q: 评估时如何跳过已完成的任务？**
A: `run_rq1_finetuned.sh` 会自动检测并跳过已完成的任务（1000个样本）。
