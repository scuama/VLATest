# Optimization 优化流程

## 快速开始

### 运行已有任务

```bash
# 后台运行 move 任务
nohup python3 optimization/run_optimization.py move > optimization/log/run_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 查看日志
tail -f optimization/log/run_*.log
```

## 添加新任务

以添加 `grasp` 任务为例：

### 1. 创建任务目录结构

```bash
mkdir -p optimization/grasp/{episodes,results,success}
```

### 2. 创建配置文件 `optimization/grasp/batch_config.json`

```json
{
  "base_dir": "results/t-grasp_n-100_o-0_s-xxxx/openvla-7b_2024",
  "task": "grasp",
  "model": "openvla-7b",
  "lora_path": null,
  "cases": [
    {
      "episode_id": "13",
      "strategy": "optimize_grasp",
      "direction": "left-up"
    },
    {
      "episode_id": "25",
      "strategy": "move_closer",
      "move_ratio": 0.5
    },
    {
      "episode_id": "37",
      "strategy": "replace_object",
      "new_object": "coke_can"
    }
  ]
}
```

**配置说明：**
- `base_dir`: 原始推理结果目录（包含各个 episode 的 options.json）
- `task`: 任务类型
- `model`: 模型名称
- `lora_path`: LoRA 权重路径（可选）
- `cases`: 案例列表，每个案例包含：
  - `episode_id`: Episode ID
  - `strategy`: 策略名称（`optimize_grasp` / `move_closer` / `replace_object`）
  - 策略特定参数（如 `direction`、`move_ratio`、`new_object`）

### 3. 运行任务

```bash
python3 optimization/run_optimization.py grasp
```

## 三种优化策略

### 1. optimize_grasp - 优化抓取位置
搜索最佳抓取位置

**参数：**
- `direction`: 搜索方向 (`left`, `right`, `up`, `down`, `left-up`, `left-down`, `right-up`, `right-down`)
- `attempts`: 搜索次数（默认 10）

### 2. move_closer - 移近机械臂
将物体向机械臂方向移动

**参数：**
- `move_ratio`: 移动比例 0-1（默认 0.5）

### 3. replace_object - 替换物体
替换场景中的物体

**参数：**
- `new_object`: 新物体名称

## 工作流程

1. **阶段1 - 策略优化**：依次对所有案例应用优化策略，生成修改后的配置
2. **阶段2 - 批量推理**：将所有优化后的配置整合成一个数据集，统一运行推理
3. **结果收集**：检查推理结果，收集成功案例到 `success/` 目录

## 输出文件

```
optimization/{task}/
├── batch_config.json      # 任务配置
├── batch_dataset.json     # 批量推理数据集（临时文件）
├── batch_report.json      # 运行报告（自动覆盖）
├── episodes/              # 优化后的配置
│   ├── 7/
│   │   ├── origin.json         # 原始配置备份
│   │   └── options.json        # 优化后的配置
│   └── ...
├── results/               # 推理结果
│   ├── batch_inference/        # 批量推理原始输出
│   ├── 7/                      # 各 episode 的结果
│   │   ├── log.json
│   │   └── images/
│   └── ...
└── success/               # 成功案例（自动收集）
    └── 7/
```

## 命令行选项

```bash
# 只处理指定 episode
python3 optimization/run_optimization.py move --episode 7

# 不跳过已成功的案例（重新处理）
python3 optimization/run_optimization.py move --no_skip
```

## 备注

- 所有策略都基于 `origin.json`（原始配置）进行修改，避免累积误差
- 批量推理一次性处理所有场景，提高效率
- 运行报告固定为 `batch_report.json`，每次运行自动覆盖，时间戳记录在报告内部
