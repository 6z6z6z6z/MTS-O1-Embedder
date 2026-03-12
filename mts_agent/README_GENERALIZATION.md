# MTS-O1-Embedder: 通用多维时序分类器

本系统是一个**完全通用**的多维时间序列嵌入模型，支持任意数据集，无需修改核心代码。

## 支持的输入格式

- 时序维度: `[T]`, `[C, T]`, 或 `[T, C]`（自动识别）
- 数据集格式:
  - 文件夹 `{dataset}/X_train.npy + y_train.npy` (UCR 风格)
  - 单个 `.npy` 或 `.npz` 文件（含 `X` / `y` 键）
- 通道数和时序长度均自动推断

## 新数据集快速上手（5步）

```bash
# Step 1: 准备数据文件夹
mkdir my_dataset
# 放入 X_train.npy (N, T, C) 和 y_train.npy (N,)
# 以及 X_valid.npy / y_valid.npy（可选）

# Step 2: 生成训练数据（JSONL格式）
python -m mts_agent.main --mode gen_data \
    --raw_data_path my_dataset \
    --data_path mts_agent/data/processed/my_train.jsonl

# Step 3: 训练
python -m mts_agent.main --mode train \
    --config mts_agent/configs/example_config.json

# Step 4（可选）: 查看数据集元信息
python mts_agent/inspect_dataset.py --data_path my_dataset

# Step 5（可选）: 生成实验报告
python mts_agent/report_experiment.py --run_dir checkpoints_my_dataset
```

## 配置文件（推荐使用JSON配置）

| 文件 | 用途 |
| :--- | :--- |
| `configs/example_config.json` | 最小化训练示例（修改路径即可用） |
| `configs/generic_timeseries_config.json` | 通用参数参考 |
| `configs/minimal_new_dataset_config.json` | 数据生成阶段模板 |

关键参数说明：

- `data.ts_dim`: 通道数，设 `null` 则自动推断
- `data.domain_info`: 可选领域描述（输入LLM的背景知识）
- `training.contrastive_weight`: 对比损失权重（默认0.7）
- `training.lm_weight`: 语言模型损失权重（默认0.3）
- `training.gradient_accumulation_steps`: 梯度累积步数（GPU显存受限时使用）

## 检索系统默认值

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `retrieval.k` | `5` | 检索邻居数 |
| `retrieval.alpha` | `0.8` | 语义相似度权重（余弦），剩余0.2为DTW结构相似度 |
| `retrieval.dtw_window_size` | `null` | DTW窗口自适应 |
| `retrieval.fast_dtw_max_len` | `100` | 超长序列先降采样再DTW |

## 扩展点（无需修改核心代码）

### 注册数据集特定的提示词模板

```python
from mts_agent.data.prompt_templates import register_prompt_template

def my_contexts(dataset_name):
    return [f"ECG recording from dataset '{dataset_name}'."]

def my_reasoning(features, label):
    return f"3. Cross-channel analysis shows {features['dominance']} dominance.\n4. Conclusion: class '{label}'."

register_prompt_template("my_ecg", my_contexts, my_reasoning)
```

### 注册自定义数据加载器

```python
from mts_agent.data.adapters import register_dataset_adapter

def my_loader(dataset_path, mode):
    # 返回 (X: np.ndarray [N, C, T], y: np.ndarray [N])
    ...

register_dataset_adapter("my_ecg", my_loader)
```

## 工具脚本

- `mts_agent/inspect_dataset.py`: 预览数据集元信息（维度、标签分布）
- `mts_agent/report_experiment.py`: 汇总实验结果（训练历史、最佳指标）


To add a new dataset family, prefer adding:

1. a new adapter rule
2. a new prompt/template rule

instead of hardcoding behavior in the training loop.