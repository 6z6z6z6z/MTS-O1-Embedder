# MTS-O1-Embedder

**Multimodal Time Series Retrieval Embedding via LLM-Enriched Contrastive Learning**

MTS-O1-Embedder maps multivariate time series (MTS) into the semantic space of a large language model (LLM) via contrastive learning, producing dense retrieval embeddings. Classification is performed by hybrid retrieval (cosine similarity + DTW) rather than a trainable head, making the approach naturally few-shot and label-agnostic.

---

## Core Idea

```
Query side (view1, heavy augmentation):
  TS signal ──► TSEncoder (CNN) ──► Projector ──► [ LLM + LoRA ] ──► ts_tokens pool ──► q embedding
                                                    ▲
                                           context-only prompt (with_grad)

Gallery side (view2, safe augmentation):
  TS signal ──► TSEncoder (CNN) ──► Projector ──► [ LLM + LoRA ] ──► ts_tokens pool ──► g embedding
                                                    ▲
                                           context-only prompt (no_grad, EMA params)

Training loss: Asymmetric InfoNCE (same-label positives) + SoftCLT + ProtoNCE
               + EMA gallery encoder + Memory Bank

Inference: kNN retrieval over gallery ──► weighted-vote classification
           score = α · cosine(q, g) + (1-α) · DTW_sim(ts_q, ts_g)
```

---

## Architecture (v18)

```
Input: [B, C=24, T=51] multivariate time series
       │
       ▼
TSEncoder (Channel-Independent CNN)
  Stem Conv1d(1→64, stride=2) + GroupNorm + GELU
  ResBlock1D × 2 (64→64→128)
  Conv1d Patching (k=5, stride=5) → P=6 patches per channel
  Output: [B, C×P=144, 128]
       │
       ▼
TimeSeriesProjector (128 → 3584)
  LayerNorm → Linear(128→7168) → GELU → Linear(7168→3584)
  + projector_norm (LayerNorm)
  Output: [B, 144, 3584]  ← injected as TS tokens into LLM
       │
       ▼
Qwen3.5-4B (LoRA r=8, α=16, gradient checkpointing)
  Prompt: "Context: {domain_desc}\nTask: Analyze patterns.\nData: [TS tokens]"
  TS tokens placed at prompt END (full causal visibility over text)
       │
       ▼
Pooling: ts_tokens mean pooling (over TS token positions)
  → final_norm (LayerNorm)
  → final_projection (Linear 3584→1024)
  → L2 normalize
       │
       ▼
Output: [B, 1024] retrieval embedding
```

| Component | Details |
|---|---|
| **TSEncoder** | Channel-independent 1D ResNet. Each channel processed as `[B*C, 1, T]`. |
| **Projector** | 2-layer MLP `128 → 7168 → 3584` with residual connection. |
| **LLM** | Qwen3.5-4B with LoRA on q/k/v/o/gate/up/down/g projections. |
| **Retrieval** | HybridRetriever: cosine embedding similarity + DTW structural similarity. |

---

## Training

### Loss Functions

| Loss | Weight | Description |
|------|--------|-------------|
| **Asymmetric InfoNCE** | 1.0 | SupCon Form 2 with SoftCLT soft positive weights |
| **ProtoNCE** | 0.1 | EMA class prototypes as additional contrastive targets |
| **Learnable τ** | — | Temperature `exp(log_tau)`, jointly optimized, clamped to [0.01, 1.0] |

### Key Techniques

- **Dual-View Augmentation (v18)**: view1 (heavy: jitter+scale+shift+channel_dropout+window_slicing) as query, view2 (safe: no window slicing) for EMA gallery
- **SoftCLT**: DTW-based soft positive weights (C×T flatten + L2 normalize → dist ∈ [0,2])
- **EMA Gallery Encoder**: Momentum-updated shadow parameters for stable gallery embeddings
- **Memory Bank**: FIFO queue (size=90) of detached gallery embeddings, flushed at warmup end
- **Balanced Sampling**: PK-style (classes_per_batch × samples_per_class)

---

## Results — NATOPS Benchmark

[NATOPS](https://www.timeseriesclassification.com/description.php?Dataset=NATOPS): 6-class aviation hand-gesture dataset, 24 channels (8 body locations × 3 axes), T=51, 180 train / 180 test samples.

### Best Results (v18, Gallery=Train, Query=Test, alpha=0.9)

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **85.56%** |
| **Macro F1** | **85.65%** |
| P@1 | 88.9% |
| P@3 | 85.6% |
| P@5 | 84.9% |
| Hit@5 | 100.0% |
| Perfect@5 | 66.1% |

### Comparison with Pure TS Baselines

| Method | P@1 | kNN@5 | Parameters |
|--------|-----|-------|------------|
| Euclidean-raw | 83.9% | 85.0% | 0 |
| DTW-multichan (global z-norm) | **88.9%** | **88.9%** | 0 |
| MTSEmbedder v9 (ts_tokens) | **90.0%** | 87.8% | ~12M |
| **MTSEmbedder v18 (ts_tokens)** | 88.9% | 87.8% | ~12M |

### Experiment History

| Version | Train LOO | Test Acc | Key Change |
|---------|-----------|----------|------------|
| v9 | 84.44% | 83.89% | CNN + context symmetric + SoftCLT + EMA + ProtoNCE |
| v12 | 81.67% | 83.89% | System fixes: memory bank flush, TS-at-end, LayerNorm, ts_tokens pooling |
| **v18** | 81.67% | **85.56%** | Dual-view augmentation (label mode, view2 gallery) |

---

## Installation

```bash
# Clone
git clone https://github.com/6z6z6z6z/MTS-O1-Embedder.git
cd MTS-O1-Embedder

# Install dependencies
pip install -r requirements.txt

# Download LLM weights (not included in repo)
# Place under local_model/ or specify path in config
```

> **GPU**: ~20GB VRAM for Qwen3.5-4B with LoRA + gradient checkpointing.
> Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid memory fragmentation.

---

## Quick Start

### Train

```bash
python -m mts_agent.main --config mts_agent/configs/natops_v18_arch.json
```

### Evaluate Retrieval

```bash
python -m mts_agent.retrieval.evaluate_retrieval \
  --ckpt_path checkpoints_natops_v18/model_best.pt \
  --config mts_agent/configs/natops_v18_arch.json \
  --k 5 --alpha 0.9 --vote_strategy weighted
```

---

## Configuration

All settings in a single JSON file. Key fields:

```json
{
  "experiment_name": "natops_v18",
  "mode": "train",
  "data": {
    "data_path": "mts_agent/data/processed/natops_train.jsonl",
    "batch_size": 18,
    "ts_dim": 24,
    "augment": true
  },
  "model": {
    "llm_path": "local_model",
    "ts_hidden_dim": 128,
    "output_dim": 1024,
    "encoder_type": "cnn",
    "embedding_pooling": "ts_tokens",
    "tuning_strategy": "lora",
    "lora_rank": 8
  },
  "training": {
    "epochs": 40,
    "lr": 1e-4,
    "contrastive_weight": 1.0,
    "proto_weight": 0.1,
    "use_soft_clt": true,
    "ema_momentum": 0.99,
    "memory_bank_size": 90,
    "balanced_sampling": true,
    "classes_per_batch": 6,
    "samples_per_class": 3,
    "dual_view": true
  },
  "retrieval": { "k": 5, "alpha": 0.9 }
}
```

See `mts_agent/configs/` for all experiment configurations.

---

## Dataset Format

### Raw data (`.npy`)

```
my_dataset/
├── X_train.npy     # (N, C, T) or (N, T, C), auto-detected
├── y_train.npy     # (N,)
├── X_test.npy
└── y_test.npy
```

### Processed JSONL

Each line:
```json
{
  "time_series": [[c1_t1, c1_t2, ...], [c2_t1, c2_t2, ...], ...],
  "context": "Domain description",
  "label": "class_id",
  "dataset": "dataset_name"
}
```

---

## Project Structure

```
MTS-O1-Embedder/
├── mts_agent/
│   ├── main.py                    # Entry point (train mode)
│   ├── config.py                  # Dataclass config system
│   ├── models/
│   │   ├── mts_embedder.py        # MTSEmbedder (core model)
│   │   ├── ts_encoder.py          # TimeSeriesEncoder, PatchTokenizer, ChannelMixer
│   │   └── projector.py           # TimeSeriesProjector
│   ├── engine/
│   │   └── trainer.py             # MTSTrainer (contrastive + EMA + ProtoNCE)
│   ├── data/
│   │   ├── loader.py              # MTSDataset
│   │   ├── collator.py            # MultimodalCollator
│   │   ├── augmentations.py       # TimeSeriesAugmentor (dual-view)
│   │   ├── samplers.py            # BalancedBatchSampler, MultiDatasetBatchSampler
│   │   ├── prompt_builder.py      # Prompt construction
│   │   ├── prompt_templates.py    # Per-dataset prompt registry
│   │   ├── adapters.py            # Raw dataset loaders
│   │   ├── ts_context_gen.py      # Context generation for datasets
│   │   ├── process_uea_datasets.py # UEA Archive processing
│   │   └── processed/             # JSONL data files
│   ├── retrieval/
│   │   ├── hybrid_search.py       # HybridRetriever (cosine + DTW)
│   │   ├── evaluate_retrieval.py  # Evaluation pipeline
│   │   └── eval_recall.py         # Recall@K analysis
│   └── configs/                   # Experiment JSON configs (v9–v18)
├── eval_ts_baselines.py           # Pure TS baseline evaluation
├── ts_baseline_comparison.py      # Baseline comparison scripts
├── recall_at_k_analysis.py        # Recall analysis
├── results/                       # Experiment reports
├── docs/                          # Technical documentation (Chinese)
├── requirements.txt
└── README.md
```

---

## Documentation

Technical documentation (Chinese) is available in the `docs/` directory:

- `架构改进计划.md` — Architecture improvement plan (P1–P8)
- `技术框架图.md` — Detailed architecture diagrams
- `系统缺陷分析与改进方向.md` — System deficiency analysis
- `项目进展与思路整理.md` — Progress summary
- `时序检索基础模型思考` — Vision for TS retrieval foundation model
- `时序检索基础模型重构方案.md` — Refactoring plan

---

## License

MIT
