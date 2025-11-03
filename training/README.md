# OpenVLA微调训练

本目录包含OpenVLA模型微调的所有脚本和工作流。

## 📁 目录结构

```
training/
├── README.md                    # 本文件
├── scripts/                     # Python脚本
│   ├── collect_expert_data.py  # 数据收集
│   ├── prepare_training_data.py # 数据预处理
│   ├── finetune_openvla_lora.py # LoRA微调
│   └── eval_finetuned_model.py  # 模型评估
├── workflows/                   # 自动化工作流
│   ├── quick_start.sh          # 快速测试（10样本）
│   ├── full.sh                 # 全量训练（343样本）
│   └── monitor.sh              # 进度监控
├── docs/                        # 文档
│   └── WORKFLOW.md             # 详细流程说明
├── data/                        # 训练数据（运行时生成）
├── checkpoints/                 # 模型检查点（运行时生成）
└── logs/                        # 训练日志（运行时生成）
```

## 🚀 快速开始

```bash
cd /mnt/disk1/decom/VLATest/training

# 1. 快速测试（约15分钟）
bash workflows/quick_start.sh

# 2. 监控进度
bash workflows/monitor.sh

# 3. 全量训练（约3-4小时）
bash workflows/full.sh
```

## 📝 详细文档

查看 [WORKFLOW.md](docs/WORKFLOW.md) 了解完整流程和配置说明。

## ⚙️ 环境要求

- Python 3.10 (.venv虚拟环境)
- CUDA 12.0+
- GPU: RTX 3090 24GB
