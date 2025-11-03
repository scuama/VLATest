# OpenVLA Grasp任务微调流程

## 环境信息
- GPU: RTX 3090 24GB
- 任务: google_robot_pick (grasp)
- 方法: LoRA微调
- 数据源: RT-1-400k成功案例

## 快速开始

### 方式1: 一键测试（推荐首次使用）
```bash
# 运行小规模测试，验证整个流程（约10-15分钟）
bash quick_start_finetuning.sh

# 监控进度
bash monitor_training.sh
```

### 方式2: 手动执行

#### 1. 数据收集（后台运行，约2-3小时）
```bash
# 小规模测试（10个成功案例，约5分钟）
screen -S collect_test
PYTHONPATH=/mnt/disk1/decom/VLATest python3 experiments/collect_expert_data.py \
    --task grasp \
    --model rt_1_400k \
    --num_samples 10 \
    --output data/training/grasp_test
# Ctrl+A+D 退出screen

# 全量收集（343个成功案例，约2-3小时）
screen -S collect_full
PYTHONPATH=/mnt/disk1/decom/VLATest python3 experiments/collect_expert_data.py \
    --task grasp \
    --model rt_1_400k \
    --num_samples -1 \
    --output data/training/grasp_full
# Ctrl+A+D 退出screen

# 查看进度
screen -r collect_test  # 或 collect_full
```

### 2. 数据预处理（约1分钟）
```bash
# 测试数据
python3 experiments/prepare_training_data.py \
    --input data/training/grasp_test \
    --output data/training/grasp_test_processed

# 全量数据
python3 experiments/prepare_training_data.py \
    --input data/training/grasp_full \
    --output data/training/grasp_full_processed
```

#### 3. LoRA微调（后台运行）
```bash
# 测试训练（快速验证流程）
screen -S train_test
PYTHONPATH=/mnt/disk1/decom/VLATest python3 experiments/finetune_openvla_lora.py \
    --data_dir data/training/grasp_test_processed \
    --output_dir checkpoints/openvla_grasp_test \
    --num_epochs 3 \
    --batch_size 8
# Ctrl+A+D 退出screen

# 正式训练
screen -S train_full
PYTHONPATH=/mnt/disk1/decom/VLATest python3 experiments/finetune_openvla_lora.py \
    --data_dir data/training/grasp_full_processed \
    --output_dir checkpoints/openvla_grasp_lora \
    --num_epochs 10 \
    --batch_size 8
# Ctrl+A+D 退出screen

# 查看训练进度
screen -r train_test  # 或 train_full
```

#### 4. 评估微调模型
```bash
# 在测试集上评估
python3 experiments/eval_finetuned_model.py \
    --model_path checkpoints/openvla_grasp_lora \
    --test_data data/t-grasp_n-1000_o-m3_s-2498586606.json \
    --output results/openvla_finetuned_eval
```

## 数据统计

### 可用训练数据
- RT-1-400k grasp成功: 343个场景
- 预计训练样本: ~5000步 (343场景 × 平均15步)
- 数据大小: ~5GB

### 训练配置
- LoRA rank: 32
- Batch size: 8 (gradient_accumulation=4, 有效batch=32)
- Learning rate: 5e-4
- Epochs: 10
- 预计训练时间: 1.5-2小时

## 目录结构
```
data/training/
├── grasp_test/              # 测试数据（10个场景）
│   ├── trajectories.pkl
│   └── metadata.json
├── grasp_test_processed/    # 预处理后
│   ├── train/
│   └── val/
├── grasp_full/              # 全量数据（343个场景）
└── grasp_full_processed/

checkpoints/
├── openvla_grasp_test/      # 测试训练检查点
└── openvla_grasp_lora/      # 正式训练检查点
```

## 进度检查命令
```bash
# 查看所有screen会话
screen -ls

# 重新连接到会话
screen -r <session_name>

# 查看GPU使用情况
nvidia-smi

# 查看数据收集进度
tail -f data/training/grasp_*/collection.log

# 查看训练日志
tail -f checkpoints/openvla_grasp_*/training.log
```

## 故障排除

### 显存不足
- 减小batch_size到4或2
- 增加gradient_accumulation_steps

### 数据收集中断
- 脚本支持断点续传，重新运行即可

### 训练中断
- 自动保存检查点，使用--resume参数继续训练

## 重要提示

### 环境要求
- **必须使用.venv虚拟环境**（脚本会自动切换）
- 如果当前在conda环境，脚本会自动停用并激活.venv

### 首次运行建议
1. **先运行测试**: `bash quick_start_finetuning.sh`
2. **监控进度**: `bash monitor_training.sh`
3. **验证流程**: 确保数据收集、预处理、训练都正常
4. **再运行全量**: `bash full_training.sh`

### 常见问题
- **环境错误**: 确保使用.venv而非conda环境
- **显存不足**: 减小batch_size到4或2
- **数据收集慢**: 正常，每个场景约10-15秒
- **训练中断**: 检查日志，可能需要调整参数

## 下一步计划
1. ⏳ Grasp任务测试训练
2. ⏳ Grasp任务全量训练
3. ⏳ Move任务数据收集（152个样本）
4. ⏳ Put-on/Put-in任务数据收集（需补充数据）
5. ⏳ 多任务联合微调
