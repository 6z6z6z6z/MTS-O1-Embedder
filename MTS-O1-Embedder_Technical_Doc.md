技术文档# 多维时间序列多模态思考型嵌入模型 (MTS-O1-Embedder) 技术设计文档

**版本**：v2.0 (独立架构版)
**日期**：2026-01-26
**目标**：构建一个**完全自研**的时间序列嵌入模型框架。该框架吸收了 O1 Embedder “先思考、再检索”的核心思想，但代码实现不依赖原 O1 仓库。通过融合数值时序数据与文本背景知识，生成具备显式推理能力的 Embedding，用于少样本/零样本时序分类任务。

---

## 1. 核心架构设计

本系统为独立的 Python 项目，旨在将 LLM 的推理能力引入时序数据挖掘领域。

### 1.1 核心理念：从“数值匹配”到“语义推理”
传统时序分类依赖数值相似度（如 Euclidean, DTW），容易受噪声和相位偏移影响。本系统引入“Agentic Reasoning”：
1.  **感知 (Perception)**：通过 Projector 将数值序列映射到 LLM 语义空间。
2.  **认知 (Cognition)**：LLM 根据背景知识生成自然语言“思考（Thought）”。
3.  **表征 (Representation)**：将“感知 + 认知” 聚合为高维向量。
4.  **决策 (Decision)**：基于混合维度（数值+语义）进行近邻检索。

### 1.2 系统架构图

```mermaid
flowchart TB
    subgraph Data_Layer [数据输入]
        direction TB
        X[多维时序矩阵 X]
        Meta[元数据/背景描述 B]
    end

    subgraph Perception_Layer [感知投射模块]
        Encoder[TS-Backbone\n(e.g., Simple ResNet/PatchTST)]
        Proj[Cross-Modal Projector\n(Linear/MLP)]
        X --> Encoder --> Proj
    end

    subgraph Cognition_Layer [推理引擎 (LLM)]
        Tokenizer[Text Tokenizer]
        Meta --> Tokenizer
        
        Input[Multimodal Prompt\n< TS_Embeds > + < Text_Tokens >]
        Proj & Tokenizer --> Input
        Input --> LLM[Backbone LLM\n(Llama/Qwen)]
        
        Thinking[Thought Generation\n(Analysis, Logic, Reasoning)]
        LLM --> Thinking
    end

    subgraph Representation_Layer [特征融合与输出]
        Combined[Query + Thought]
        Thinking --> Combined -.->|Re-encode| LLM
        Pool[Token Pooling\n(Last Token/Mean)]
        LLM --> Pool --> Embed[Final MTS-Embedding]
    end
```

---

## 2. 关键模块详解

### 2.1 时序-语言对齐模块 (The Projector)
**职责**：充当“翻译官”，将连续的数值信号转换为 LLM 可以理解的离散语义向量。
*   **输入**：$X \in \mathbb{R}^{B \times T \times D}$ (Batch, Time, Dim)
*   **架构**：
    1.  **Feature Extractor**: 采用 `TimeSeiresEncoder` (如轻量级 1D CNN 或 Transformer Encoder) 提取局部特征，输出 $H \in \mathbb{R}^{B \times L \times d_{model}}$。
    2.  **Projection Head**: 简单的 `nn.Linear(d_{model}, d_{llm})`，将时序特征维度对齐到 LLM 的 Hidden Size。
*   **训练策略**：在第一阶段主要训练此模块，冻结 LLM 参数，使时序 Token 能够激活 LLM 的相关语义区域。

### 2.2 思考生成模块 (Thought Generator)
**职责**：显式生成推理文本，作为“虚拟标签”增强 Query。这是典型的 Chain-of-Thought (CoT) 过程，但扩展到了多模态领域。

*   **输入机制**：
    *   思维链的生成，是**预训练 LLM**（Backbone）直接对**统一向量空间（Unified Embedding Space）**中的混合序列进行的。
    *   序列构成：`[Text_Embeds, TS_Embeds]`。
    *   LLM 并不区分哪些向量来自文本，哪些来自时序 Projector，它看到的只是一个连续的 context window。

*   **数据增强与标签构建 (Synthetic Data Strategy)**：
    *   **Context 生成 (Input)**：针对无描述的样本，利用 `统计特征` + `规则模版` 生成**基础背景文本 (Context)**。这仅作为给模型的“提示/上下文”。
    *   **Thought 生成 (Label)**：采用 **Teacher-Student 蒸馏策略**。
        *   利用外部脚本 (基于统计规则/GPT-4) 生成一批高质量、逻辑准确的 "Teacher Thoughts"。
        *   这些合成的思维链作为 Ground Truth，通过监督微调 (SFT) 教会模型学习类人的时序分析逻辑。

*   **Prompt 机制**：
    ```python
    SYSTEM_PROMPT = "You are a time series analyst. Analyze the provided data trend and context."
    USER_TEMPLATE = "Context: {background}\nData: <|ts_start|>{ts_tokens}<|ts_end|>\nTask: Analyze patterns."
    ```
*   **训练流程优化**：
    1.  **Stage 1 (Alignment)**: 训练 Projector。使用 InfoNCE Loss 或 Captioning Loss，拉近 `TS_Embeds` 与 `Text_Description` 的距离，使 LLM 能理解时序特征的语义。
    2.  **Stage 2 (Reasoning SFT)**: 保持 Projector 和 LLM (或者 LoRA) 开启。使用 `(User_Template, Teacher_Thought)` 对进行标准的 SFT 训练。目标是让模型在推理阶段能模仿 Teacher 生成类似的分析。


### 2.3 混合检索器 (Hybrid Retriever)
**职责**：在 Embedding 空间与原始时序空间同时寻找邻居。
*   **综合得分计算**：
    $$ S(q, d) = w_1 \cdot \text{CosSim}(\mathbf{e}_q, \mathbf{e}_d) + w_2 \cdot \text{NormSim}_{DTW}(\mathbf{x}_q, \mathbf{x}_d) $$
    *   **语义侧 ($w_1$)**：捕捉“趋势下降”、“震荡”、“阀门故障”等高层概念。
    *   **波形侧 ($w_2$)**：捕捉细节上的对齐程度，弥补 Embedding 可能丢失的细微形态差异。
*   **当前工程默认值**：
    *   `k = 5`
    *   `alpha = 0.8`，即当前默认更偏向语义相似度，再辅以 DTW 结构约束。
    *   `dtw_window_size = null`，表示默认采用检索器内部的自适应窗口策略。
    *   `fast_dtw_max_len = 100`，表示超长序列会先降采样，再进行 DTW 计算。
*   **实现说明**：这些默认值已经在配置模板、训练期 retrieval validation 与独立检索评估脚本中统一，避免“配置文件默认值”和“运行时默认值”不一致。

### 2.4 数据增强 (Data Augmentation) - New in v2.1
**职责**：针对小样本场景（如 AtrialFibrillation 仅 15 个样本），在训练时动态生成变换数据，防止过拟合。
*   **策略**：
    1.  **Jittering**: 添加高斯噪声 ($\sigma=0.03$)，模拟传感器噪声。
    2.  **Scaling**: 随机缩放振幅 ($\times N(1, 0.1)$)，模拟不同个体的信号强度差异。
    3.  **Shifting**: 随机数值偏移，模拟基线漂移。
*   **实现**：集成在 `MTSDataset` 中，仅在 `train` 模式下启用。

---

## 3. 代码项目结构规划 (Standalone)

建立全新的项目目录 `mts_agent`，不依赖任何 O1 Embedder 的代码。

```text
mts_agent/
├── data/                   # 数据处理
│   ├── generator.py        # 基于 GPT-4 的 Thought 数据生成脚本
│   ├── loader.py           # 自定义 Dataset (加载 .npy + .json)
│   └── collator.py         # 处理多模态输入的 Padding 与拼接
├── models/                 # 模型定义
│   ├── __init__.py
│   ├── ts_encoder.py       # 时序特征提取器 (ResNet/Transformer)
│   ├── projector.py        # 简单的线性映射层
│   └── mts_embedder.py     # 主模型 (继承 nn.Module, 组合 LLM+Projector)
├── engine/                 # 训练与推理流程
│   ├── trainer.py          # 自定义训练循环 (Generation Loss + Contrastive Loss)
│   └── inference.py        # 推理逻辑 (Think -> Embed -> Mean)
├── retrieval/              # 检索模块
│   └── hybrid_search.py    # 实现 FAISS + DTW 双路检索
└── main.py                 # 入口脚本
```

### 3.2 技术难点与解决方案大纲

| 难点 | 描述 | 解决方案 (Solution) |
| :--- | :--- | :--- |
| **模态对齐** | LLM 无法理解原始数值矩阵 | 引入 **Perception Layer**，使用 `Encoder + Projector` 将时序特征映射到 LLM 语义空间。 |
| **数据缺失** | 开源数据集缺乏样本级文本描述 | 实施 **Template-based Synthesis**，利用元数据+统计特征+规则模版合成伪基准文本。 |
| **特征融合** | 数值特征与文本 Prompt 如何拼接 | 在 **Representation Layer** 的统一向量空间 (Embedding Space) 中进行维度级联 (Concatenation)，LLM 自注意力机制负责融合。 |
| **相似度量** | 传统距离 (DTW) 与语义距离不兼容 | 采用 **Hybrid Retrieval Strategy**，归一化后加权融合余弦相似度与 DTW 距离。 |

### 3.3 核心模型类定义 (Preview)

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class MTSEmbedder(nn.Module):
    def __init__(self, llm_path, ts_input_dim, ts_hidden_dim):
        super().__init__()
        # 1. LLM Backbone
        self.llm = AutoModelForCausalLM.from_pretrained(llm_path)
        
        # 2. Time Series Encoder (Perception Layer)
        self.ts_encoder = TimeSeriesBackbone(input_dim=ts_input_dim, output_dim=ts_hidden_dim)
        
        # 3. Modal Projector (Perception Layer)
        self.projector = nn.Linear(ts_hidden_dim, self.llm.config.hidden_size)
        
    def forward(self, ts_data, text_input_ids, attention_mask, labels=None):
        # --- Perception Phase ---
        # Step 1: Encode Time Series to Semantic Space
        # ts_data: [Batch, Time, Dim] -> ts_feat: [Batch, Seq, H_ts]
        ts_feat = self.ts_encoder(ts_data) 
        # Project to LLM dim: [Batch, Seq, H_llm]
        ts_embeds = self.projector(ts_feat) 
        
        # Step 2: Get Text Embeddings from Tokenizer IDs
        # text_input_ids: [Batch, Text_Len] -> [Batch, Text_Len, H_llm]
        text_embeds = self.llm.model.embed_tokens(text_input_ids)
        
        # --- Cognition Phase ---
        # Step 3: Multimodal Concatenation (Early Fusion)
        # 拼接逻辑：[Text_Prefix, TS_Embeds, Text_Suffix]
        # 注意：需要根据实际 mask 动态处理拼接位置
        inputs_embeds = torch.cat([text_embeds[:, :prefix_len], ts_embeds, text_embeds[:, prefix_len:]], dim=1)
        
        # Step 4: LLM Forward (Deep Reasoning)
        # LLM reads the unified sequence and generates thoughts/embeddings
        outputs = self.llm(inputs_embeds=inputs_embeds, labels=labels, output_hidden_states=True)
        
        # --- Representation Phase ---
        # Step 5: Pooling for Final Embedding
        # Extract the last hidden state of the last token (or specific <EMB> token)
        last_hidden_state = outputs.hidden_states[-1] 
        # embeddings = self.pool(last_hidden_state, attention_mask)
        
        return outputs # Returns loss during training, embeddings during inference
```


---

## 4. 实施**Context 生成**: 为 UCR 数据集附加规则生成的基础背景文本。
    *   **Teacher Thought 生成**: 编写脚本，基于统计规则批量生成 "冷启动用" 的思维标签 (Teacher Labels)。
1.  **数据就绪 (Data Readiness)**
    *   构造示例数据集：使用开源时序数据 (如 UCR)，附加规则生成的**基础背景文本 (Context)**。
    *   *注：Thought 标签将在 Stage 1.5 由模型自己生成，不再依赖外部脚本。*

2.  **原型开发 (Prototype)**
    *   实现 `MTSEmbedder` 的最小可行版本 (MVP)。
    *   跑通 Forward Pass，确保维度对齐。

3.  **训练与验证 (Train & Val)**
    *   实现双目标 Loss (Next Token Prediction + Contrastive Loss)。
    *   在小规模数据上验证 Loss 收敛性。

4.  **检索系统集成 (Integration)**
    *   编写混合检索器，封装为 Agent Tool。