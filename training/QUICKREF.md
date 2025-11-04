# OpenVLA微调快速参考

## 🚀 一键启动

```bash
cd /mnt/disk1/decom/VLATest/training

# 快速测试（10样本，15分钟）
bash workflows/quick_start.sh

# 全量训练（343样本，3-4小时）
bash workflows/full.sh

# 监控进度
bash workflows/monitor.sh
```

## 📋 完整工作流

```bash
# 1. 数据收集
PYTHONPATH=$PWD python training/scripts/collect_expert_data.py \
    --task grasp --model rt_1_400k --num_samples 10 \
    --output training/data/grasp_test

# 2. 数据预处理
python training/scripts/prepare_training_data.py \
    --input training/data/grasp_test \
    --output training/data/grasp_test_processed

# 3. LoRA微调
PYTHONPATH=$PWD python training/scripts/finetune_openvla_lora.py \
    --data_dir training/data/grasp_test_processed \
    --output_dir training/checkpoints/openvla_grasp_test \
    --num_epochs 3 --batch_size 4

# 4. 评估模型
cd training/scripts
bash run_rq1_finetuned.sh                    # 完整评估
bash test_rq1_single.sh grasp 10             # 快速测试
```

## 🔧 核心脚本

| 脚本 | 功能 | 用途 |
|------|------|------|
| `collect_expert_data.py` | 数据收集 | 使用RT-1收集成功轨迹 |
| `prepare_training_data.py` | 数据预处理 | 转换为OpenVLA格式 |
| `finetune_openvla_lora.py` | LoRA微调 | 训练OpenVLA模型 |
| `run_rq1_finetuned.sh` | RQ1评估 | 评估4个基础任务 |
| `test_rq1_single.sh` | 快速测试 | 单任务小样本测试 |

## 📊 监控命令

```bash
# 查看训练日志
tail -f training/checkpoints/openvla_grasp_test/training.log

# 查看GPU使用
watch -n 1 nvidia-smi

# 查看评估进度
find results -name 'log.json' | wc -l

# 查看成功率
grep -l '"success": true' results/*/openvla-7b_finetuned_*/*/log.json | wc -l
```

## ⚙️ 常用参数

### 数据收集
- `--task`: grasp/move/put-on/put-in
- `--model`: rt_1_400k（推荐）
- `--num_samples`: 样本数（-1=全部）

### 微调训练
- `--num_epochs`: 3-10（推荐）
- `--batch_size`: 4-8（根据GPU）
- `--learning_rate`: 5e-4（默认）
- `--lora_r`: 32（默认）

## 💡 最佳实践

1. **先测试后全量**: 用10个样本验证流程
2. **监控显存**: 不足时减小batch_size
3. **保存检查点**: 训练会自动保存best_model
4. **断点续传**: 脚本支持中断后继续

## 📁 目录结构

```
training/
├── scripts/          # 核心脚本
├── workflows/        # 自动化工作流
├── data/            # 训练数据
├── checkpoints/     # 模型检查点
└── logs/            # 训练日志
```

详细文档请查看 [README.md](README.md)
