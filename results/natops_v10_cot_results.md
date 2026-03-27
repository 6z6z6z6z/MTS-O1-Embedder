# NATOPS v10_cot 实验结果（2026-03-21）

## 实验配置
- **模型**：CNN TSEncoder + Qwen3.5-4B LoRA
- **关键差异 vs v9**：`alignment_text_mode: "full"`（带 DeepSeek CoT thought）
- **max_length**：512（v9 是 320）
- **早停**：ep31，best ep19（train LOO=80.56%）

## 训练 LOO 曲线（关键 epoch）
| Epoch | Train LOO | 备注 |
|-------|-----------|------|
| ep8  | 67.22% | 快速上升 |
| ep12 | 69.44% | |
| ep14 | 73.89% | |
| ep18 | 77.78% | |
| **ep19** | **80.56%** | **Best checkpoint** |
| ep26 | 78.33% | |
| ep31 | 75.00% | Early stop |

## 最终测试结果

### Trainer 报告（k=5, alpha=0.9, vote=weighted）
- **Test Accuracy: 80.56%**
- **Macro-F1: 79.57%**

### Recall@K（gallery=train context-only，query=thought-enriched）
| 指标 | 值 |
|------|-----|
| P@1  | 80.6% |
| P@3  | 78.7% |
| P@5  | 78.1% |
| Hit@1 | 80.6% |
| Hit@3 | 91.1% |
| Hit@5 | 97.2% |
| Perfect@5（5/5全中） | 58.9% |

### Top-5 同类分布
| 同类数/5 | query 数 |
|---------|---------|
| 0/5 | 5 |
| 1/5 | 14 |
| 2/5 | 21 |
| 3/5 | 19 |
| 4/5 | 15 |
| 5/5 | **106** |
