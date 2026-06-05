# 精简消融实验配置说明

## 📊 实验概览

本目录包含了跨数据集的精简消融实验配置，用于验证ASP方法各个创新模块的有效性。

### 支持的数据集
- **NWPU-RESISC45**: 45类，25-way 5-shot，初始25类 + 4个增量任务(每次4类) - **GPU0**
- **UCMerced**: 21类，12-way 3-shot，初始12类 + 3个增量任务(每次3类) - **GPU1**
- **SIRI-WHU**: 12类，6-way 2-shot，初始6类 + 3个增量任务(每次2类) - **GPU1**
- **MSTAR**: 暂时注释，不执行消融实验

## 🧪 消融实验设计

每个数据集包含4个消融实验配置：

### 1. **Baseline** (`*_baseline.json`)
- **目的**: 技术基线，仅使用原始ASP方法
- **启用模块**: 无（所有创新模块禁用）
- **描述**: 验证原始ASP方法的基础性能

### 2. **Efficiency Optimization** (`*_efficiency_optimization.json`)
- **目的**: 验证轻量级特征调制v1的效果
- **启用模块**: 轻量级特征调制v1
- **描述**: ASP + 轻量级特征调制v1

### 3. **Efficiency + New Class Aware** (`*_efficiency_plus_newclass.json`)
- **目的**: 验证新类感知分类器的增量效果
- **启用模块**: 轻量级特征调制v1 + 新类感知分类器
- **描述**: 在效率优化基础上添加新类感知能力

### 4. **Full System** (`*_full_system.json`)
- **目的**: 验证完整系统的最佳性能
- **启用模块**: 所有创新模块
- **描述**: 空间提示 + 轻量级调制v1 + 新类感知 + 阶段1 + 阶段2

## 🔧 数据预处理配置

### FACT风格预处理 (UCMerced, SIRI-WHU, NWPU)
```json
{
  "_comment_fact_preprocessing": "=== FACT风格数据预处理配置 ===",
  "enable_fact_preprocessing": true,
  "enable_fact_autoaugment": true
}
```

### SAR专用预处理 (MSTAR)
```json
{
  "_comment_sar_preprocessing": "=== SAR图像专用数据预处理配置 ===",
  "enable_sar_augmentation": false,
  "enable_sar_mixup": false
}
```

## 📈 最佳参数配置

基于跨数据集参数敏感性分析，各数据集使用以下最佳参数：

### UCMerced最佳配置 (GPU1)
```json
{
  "init_lr": 0.007,
  "anchor_lambda": 0.06,
  "weight_decay": 0.0005,
  "spatial_relation_weight": 0.35,
  "prompt_token_num": 6,
  "EMA_beta": 0.995,
  "kl_weight": 0.001,
  "fs_lr": 0.001
}
```

### SIRI-WHU最佳配置 (GPU1)
```json
{
  "init_lr": 0.008,
  "anchor_lambda": 0.06,
  "weight_decay": 0.0005,
  "spatial_relation_weight": 0.35,
  "prompt_token_num": 6,
  "EMA_beta": 0.995,
  "kl_weight": 0.001,
  "fs_lr": 0.001
}
```

### NWPU-RESISC45最佳配置 (GPU0)
```json
{
  "init_lr": 0.00695,
  "anchor_lambda": 0.076,
  "weight_decay": 0.000554,
  "spatial_relation_weight": 0.385,
  "prompt_token_num": 6,
  "EMA_beta": 0.999,
  "kl_weight": 0.002,
  "fs_lr": 0.001
}
```

### MSTAR最佳配置
```json
{
  "init_lr": 0.007,
  "anchor_lambda": 0.076,
  "weight_decay": 0.0005,
  "spatial_relation_weight": 0.35,
  "prompt_token_num": 6,
  "EMA_beta": 0.995,
  "kl_weight": 0.0015,
  "fs_lr": 0.001
}
```

## 🚀 运行方法

### 🔥 推荐：双GPU并行执行所有消融实验
```bash
# 自动并行执行所有消融实验（GPU0: NWPU, GPU1: UCMerced+SIRI-WHU）
python run_simplified_ablation_experiments.py
```

### 单个实验
```bash
# NWPU基线实验 (GPU0)
python main.py --config=./exps/simplified_ablation/nwpu_baseline.json

# UCMerced基线实验 (GPU1)
python main.py --config=./exps/simplified_ablation/ucmerced_baseline.json

# SIRI-WHU完整系统实验 (GPU1)
python main.py --config=./exps/simplified_ablation/siri_whu_full_system.json
```

### 按数据集批量运行
```bash
# 运行所有NWPU消融实验 (GPU0)
for config in ./exps/simplified_ablation/nwpu_*.json; do
    python main.py --config="$config"
done

# 运行所有UCMerced消融实验 (GPU1)
for config in ./exps/simplified_ablation/ucmerced_*.json; do
    python main.py --config="$config"
done

# 运行所有SIRI-WHU消融实验 (GPU1)
for config in ./exps/simplified_ablation/siri_whu_*.json; do
    python main.py --config="$config"
done
```

### GPU分配检查
```bash
# 检查GPU分配情况
python scripts/check_gpu_allocation.py
```

## 📊 预期结果分析

### 性能提升预期
1. **Baseline → Efficiency**: +1-2% (轻量级调制效果)
2. **Efficiency → Efficiency+NewClass**: +2-3% (新类感知效果)
3. **Efficiency+NewClass → Full System**: +3-5% (完整系统协同效果)

### 数据集特异性
- **SIRI-WHU**: 对学习率和锚点损失最敏感
- **UCMerced**: 新类学习能力最强
- **NWPU**: 对空间关系权重最敏感
- **MSTAR**: SAR图像需要特殊处理策略

## 🔍 实验验证要点

1. **模块有效性**: 每个模块都应带来性能提升
2. **数据集适应性**: 不同数据集的最佳配置可能不同
3. **计算效率**: 轻量级模块应保持较低的计算开销
4. **稳定性**: 多种子实验验证结果的稳定性

## ⚡ GPU并行执行优势

### 🎯 高效资源利用
- **GPU0**: 专门处理NWPU数据集（4个实验）
- **GPU1**: 处理UCMerced和SIRI-WHU数据集（8个实验）
- **并行执行**: 两个GPU同时工作，大幅缩短总实验时间

### 📊 负载分配策略
- **NWPU**: 数据集最大（45类），单独使用GPU0
- **UCMerced + SIRI-WHU**: 数据集较小，共享GPU1
- **智能调度**: 自动检测实验完成状态，避免重复执行

### ⏱️ 时间估算
- **串行执行**: 约12-16小时
- **并行执行**: 约6-8小时（提升50%+效率）

## 📁 日志保存结构

### 🗂️ 双重日志保存机制
所有实验日志都会**同时保存到两个位置**：

1. **原始位置**: `logs/asp/数据集/配置/` (保持兼容性)
2. **消融实验结果**: `消融实验结果/` (便于结果整理)

### 📊 消融实验结果目录结构
```
消融实验结果/
├── simplified_ablation_YYYYMMDD_HHMMSS.log          # 主执行日志
├── simplified_ablation_report_YYYYMMDD_HHMMSS.json  # 实验总报告
├── multi_seed_aggregation_report_YYYYMMDD_HHMMSS.json # 多seed聚合报告
├── nwpu/                                             # NWPU实验日志
│   ├── nwpu_baseline_2024_pretrained_vit_b16_224_vpt.log
│   ├── nwpu_baseline_2024_info.json
│   ├── nwpu_efficiency_optimization_2025_pretrained_vit_b16_224_vpt.log
│   └── ...
├── ucmerced/                                         # UCMerced实验日志
│   ├── ucmerced_baseline_2024_pretrained_vit_b16_224_vpt.log
│   ├── ucmerced_baseline_2024_info.json
│   └── ...
├── siri_whu/                                         # SIRI-WHU实验日志
│   ├── siri_whu_baseline_42_pretrained_vit_b16_224_vpt.log
│   ├── siri_whu_baseline_42_info.json
│   └── ...
└── 可视化/                                           # 可视化文件
    ├── multi_seed_analysis_YYYYMMDD_HHMMSS.png
    └── ...
```

### 📋 实验信息文件
每个实验日志都对应一个`*_info.json`文件，包含：
- 实验名称和类型
- 数据集信息
- 配置文件路径
- 种子值
- 日志文件路径
- 复制时间

## 📝 注意事项

1. **环境要求**: 确保在asp-rtx5090环境中运行，支持双GPU
2. **数据准备**: 确保所有数据集已正确预处理
3. **FACT预处理**: UCMerced、SIRI-WHU、NWPU使用FACT预处理
4. **GPU监控**: 可使用`nvidia-smi`监控GPU使用情况
5. **多种子验证**: 所有实验都启用了多种子验证以确保结果可靠性
6. **日志管理**: 双重保存机制确保日志不丢失，便于结果分析
7. **结果整理**: 所有消融实验结果集中在`消融实验结果/`目录，便于论文撰写
