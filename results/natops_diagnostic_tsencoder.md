# MTSEmbedder 诊断结果（2026-03-21）

## 核心发现：TS Encoder 是瓶颈

**各组件拆解测试结果（NATOPS test set，gallery=train，纯余弦相似度）：**

| 方法 | P@1 | P@3 | P@5 | Hit@5 | 完美率(5/5) |
|------|-----|-----|-----|-------|------------|
| TS-only（绕过LLM，仅TS encoder+projector） | 63.9% | 60.6% | 59.4% | 95.6% | 17.8% |
| Full model，非对称（gallery=context，query=thought）| 80.6% | 81.3% | 79.7% | 93.9% | 61.7% |
| **Full model，对称（双侧都用context-only）** | **80.6%** | **80.4%** | **81.4%** | **98.3%** | **64.4%** |
| Euclidean（原始z-norm特征） | 84.4% | 81.3% | 76.0% | 100% | 50.0% |
| DTW（多通道均值） | 84.4% | 83.5% | 80.9% | 100% | 58.9% |

## 问题分析

### 问题1（已定位）：推理分布偏移
- 训练时 `alignment_text_mode=context`（双侧context-only）
- 推理时 query 用了 thought-enriched（与训练分布不一致）
- **修复**：推理时也用context-only（对称），Hit@5 从93.9% → 98.3%，完美率 61.7% → 64.4%

### 问题2（核心瓶颈）：TS Encoder 信息损失
- 原始特征（z-norm）→ Euclidean P@1=84.4%
- TS encoder 压缩后 P@1=63.9%（损失约20%）
- CNN stem + ResBlock + AdaptiveAvgPool 的有损压缩导致判别信息丢失
- TS encoder 从随机初始化，仅靠180个样本的对比损失监督，无法学到好的表示

### LLM 的贡献
- TS-only: 63.9% → Full model: 80.6%，LLM forward 提升 +16.7%
- 说明LLM attention确实在整合TS信息，但上限受限于TS encoder质量

## 解决方向
1. **对称推理**（立即可用）：双侧都用context-only
2. **替换TS encoder**：CNN → Patch Embedding（无损压缩，参考PatchTST）
3. **预训练TS encoder**：MAE重建辅助任务
4. **使用预训练TS基础模型**：MOMENT等
