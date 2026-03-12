# MTS-O1-Embedder

**Multimodal Time Series Semantic Embedding via LLM-Enriched Reasoning**

MTS-O1-Embedder is a framework that maps multivariate time series signals into the semantic space of a large language model (LLM). The key idea is *asymmetric embedding*: at query time the model generates a chain-of-thought "reasoning trace" about the signal and embeds the enriched representation; at gallery time a lightweight context-only embedding is used. Classification is then performed by hybrid retrieval (cosine similarity + DTW) rather than a trainable head, making the approach naturally few-shot and label-agnostic.

---

## Core Idea

```
Query side  (thought-enriched):
  TS signal ──► TSEncoder ──► Projector ──► [ LLM + LoRA ]  ──► mean-pool ──► q embedding
                                             ▲
                                    full prompt (context + chain-of-thought)

Gallery side  (context-only):
  TS signal ──► TSEncoder ──► Projector ──► [ LLM + LoRA ]  ──► mean-pool ──► g embedding
                                             ▲
                                    context-only prompt  (no_grad)

Training loss: asymmetric InfoNCE  (same-label pairs as positives)

Inference: kNN retrieval over gallery  ──► weighted-vote classification
           score = α · cosine(q, g) + (1-α) · DTW_sim(ts_q, ts_g)
```

The chain-of-thought traces are pre-generated offline by a reasoning LLM (e.g. DeepSeek-R1 via API) — they describe observable signal patterns **without revealing the class label**. This lets the embedding model learn *why* two signals are similar, not just *that* they have the same label.

---

## Architecture

| Component | Details |
|---|---|
| **TSEncoder** | Conv1D stem + 3 × ResBlock1D (stride-2 downsampling) + AdaptiveAvgPool → `[B, target_tokens, hidden_dim]` |
| **Projector** | 2-layer MLP: `ts_hidden_dim → llm_dim` + LayerNorm |
| **LLM backbone** | Any Qwen-family causal LM (default: Qwen2.5-0.5B). TS token sequence is inserted between `<|vision_start|>` markers. |
| **LLM tuning** | LoRA (r=8, α=16) applied to attention layers; TSEncoder + Projector always fully trainable. |
| **HybridRetriever** | Cosine index over L2-normalized embeddings. DTW computed with optional Sakoe-Chiba band. |

---

## Results — NATOPS Benchmark

[NATOPS](https://www.timeseriesclassification.com/description.php?Dataset=NATOPS) is a 6-class aviation hand-gesture dataset recorded with motion-capture sensors (24 channels, 8 body locations × 3-axis, 51 time steps). There are 180 training samples and 180 test samples, with perfectly balanced classes (30 per class).

### Configuration (natops_v2)

| Setting | Value |
|---|---|
| LLM backbone | Qwen2.5-0.5B (local) |
| Tuning strategy | LoRA (r=8, α=16) |
| Epochs | 30 (early stopping, patience=8) |
| Batch | 30 (6 classes × 5 samples, balanced) |
| Temperature (InfoNCE) | 0.1 |
| Contrastive weight | 1.0 |
| CLS auxiliary weight | 0.3 |
| Retrieval k | 5 |
| Alpha (cosine vs DTW) | 0.9 |
| Vote strategy | weighted |

### Test Results

| Metric | v1 | v2 (best) |
|---|---|---|
| Test Accuracy | 89.4% | **90.0%** |
| Macro F1 | 89.4% | **90.0%** |

The model uses **no test labels at any point** — inference is pure kNN retrieval over a gallery built from the training set.

### Training Curve (v2 — retrieval accuracy on train LOO)

| Epoch | Train-LOO Acc | Internal-Val Acc |
|---|---|---|
| 1 | 79.2% | 80.6% |
| 6 | 79.2% | 86.1% |
| 11 | 85.4% | 77.8% |
| 13 | 84.7% | 83.3% |
| *(final test)* | — | **90.0%** |

---

## Installation

```bash
# 1. Clone
git clone https://github.com/<your-username>/MTS-O1-Embedder.git
cd MTS-O1-Embedder

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download a Qwen2.5-0.5B model (or any compatible Qwen model)
#    Place weights under  local_model/
#    e.g. using huggingface-cli:
huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir local_model
```

> **GPU requirement:** ~6–8 GB VRAM for Qwen2.5-0.5B with LoRA. CPU inference is supported but slow.

---

## Quick Start — NATOPS Example

### Step 1 — Generate chain-of-thought traces (optional, pre-generated files included)

The repository already ships `mts_agent/data/processed/natops_train.jsonl` and `natops_valid.jsonl` with pre-generated DeepSeek thoughts, so you can skip this step.

To regenerate thoughts with your own API key:

```bash
python -m mts_agent.main \
  --mode gen_data \
  --raw_data_path mts_agent/NATOPS \
  --dataset_name natops \
  --data_path mts_agent/data/processed/natops_train.jsonl
```

Each output record in the JSONL file looks like:

```json
{
  "ts":      [[...], [...], ...],
  "context": "Motion capture recording of a pilot performing an aviation hand gesture ...",
  "thought": "The signal exhibits asymmetric arm movement: the right-hand channels show ...",
  "label":   "3.0"
}
```

The `thought` field is generated **without the label** — the LLM is asked only to describe the observable signal patterns.

### Step 2 — Train

```bash
python -m mts_agent.main --config mts_agent/configs/natops_v2.json
```

Training output (saved to `checkpoints_natops_v2/`):

```
=== Starting Training Setup: natops_v2_improved ===
Resolved time-series input dimension: 24
Internal split: 144 train / 36 internal-val (stratified 80/20).
Loading held-out test set from mts_agent/data/processed/natops_valid.jsonl...
...
[Epoch 1] train_loss=5.259 | val_loss=1.787 | retrieval_acc=79.2%
...
[Final Test] accuracy=0.9000  macro_f1=0.8996
```

### Step 3 — Standalone Retrieval Evaluation

After training, you can run the retrieval evaluator independently:

```bash
python -m mts_agent.retrieval.evaluate_retrieval \
  --ckpt_path checkpoints_natops_v2/model_best.pt \
  --config    mts_agent/configs/natops_v2.json \
  --k 5 \
  --alpha 0.9 \
  --vote_strategy weighted
```

---

## Configuration Reference

All settings live in a single JSON file. Key sections:

```json
{
  "experiment_name": "my_experiment",
  "mode": "train",
  "data": {
    "data_path":    "path/to/train.jsonl",
    "raw_data_path":"path/to/raw_npy_folder",
    "dataset_name": "my_dataset",
    "batch_size":   30,
    "ts_dim":       24,
    "augment":      true,
    "domain_info":  "Short free-text description of the domain."
  },
  "model": {
    "llm_path":             "local_model",
    "ts_input_dim":         24,
    "ts_hidden_dim":        128,
    "encoder_base_channels":64,
    "encoder_target_tokens":16,
    "tuning_strategy":      "lora"
  },
  "training": {
    "epochs":                   30,
    "lr":                       1e-4,
    "contrastive_temperature":  0.1,
    "contrastive_weight":       1.0,
    "balanced_sampling":        true,
    "classes_per_batch":        6,
    "samples_per_class":        5,
    "early_stopping_patience":  8,
    "save_dir":                 "checkpoints_my_experiment",
    "retrieval_eval_enabled":   true,
    "retrieval_eval_k":         5,
    "retrieval_eval_alpha":     0.9
  },
  "retrieval": {
    "k":     5,
    "alpha": 0.9
  },
  "environment": {
    "force_offline": true,
    "device":        "cuda"
  }
}
```

See `mts_agent/configs/` for complete examples including `natops_v1.json`, `natops_v2.json`, `generic_timeseries_config.json`, and `minimal_new_dataset_config.json`.

---

## Dataset Format

MTS-O1-Embedder expects a folder containing `.npy` arrays and a JSONL file generated by the `gen_data` step.

### Raw data folder layout

```
my_dataset/
├── X_train.npy     # float32, shape (N_train, T, C)  or  (N_train, C, T)
├── y_train.npy     # shape (N_train,)
├── X_test.npy      # float32, shape (N_test, T, C)
└── y_test.npy      # shape (N_test,)
```

### Processed JSONL layout

Each line is a JSON object:

```json
{
  "ts":      [[c1_t1, c2_t1, ...], [c1_t2, c2_t2, ...], ...],
  "context": "Domain description shown to the LLM",
  "thought": "Chain-of-thought trace generated offline",
  "label":   "class_id"
}
```

---

## Adding a Custom Dataset

1. **Prepare `.npy` arrays** as described above and place them in `mts_agent/<DatasetName>/`.

2. **Register a prompt template** (optional but recommended):

```python
# mts_agent/data/prompt_templates.py
from mts_agent.data.prompt_templates import register_prompt_template

register_prompt_template("my_dataset", {
    "context": "EEG recording from a BCI experiment. 22 channels, 500 Hz.",
    "fallback_thought": "The signal shows amplitude modulation in the alpha band."
})
```

3. **Generate thoughts:**

```bash
python -m mts_agent.main \
  --mode gen_data \
  --raw_data_path mts_agent/MyDataset \
  --dataset_name my_dataset \
  --data_path mts_agent/data/processed/my_dataset_train.jsonl
```

4. **Create a config file** — copy `mts_agent/configs/minimal_new_dataset_config.json` and update `data_path`, `raw_data_path`, `dataset_name`, `ts_dim`, and `domain_info`.

5. **Train:**

```bash
python -m mts_agent.main --config mts_agent/configs/my_dataset_config.json
```

See `mts_agent/README_GENERALIZATION.md` for a more detailed walkthrough.

---

## Project Structure

```
MTS-O1-Embedder/
├── mts_agent/
│   ├── main.py                    # Entry point (gen_data / train / inference)
│   ├── config.py                  # All config dataclasses, JSON load/save
│   ├── tokenization.py            # DummyTokenizer fallback for offline mode
│   ├── inspect_dataset.py         # CLI: print dataset metadata
│   ├── report_experiment.py       # CLI: summarize a training run
│   │
│   ├── models/
│   │   ├── ts_encoder.py          # 1D ResNet time-series encoder
│   │   ├── projector.py           # TS → LLM dimension projector
│   │   └── mts_embedder.py        # MTSEmbedder (core model)
│   │
│   ├── engine/
│   │   └── trainer.py             # MTSTrainer: training loop + retrieval eval
│   │
│   ├── data/
│   │   ├── loader.py              # MTSDataset (JSONL loader)
│   │   ├── collator.py            # MultimodalCollator
│   │   ├── prompt_builder.py      # Prompt utilities (system/user/retrieval)
│   │   ├── prompt_templates.py    # Per-dataset prompt template registry
│   │   ├── generator.py           # Offline thought generation via LLM API
│   │   ├── augmentations.py       # Jitter / scale / shift augmentations
│   │   ├── samplers.py            # Balanced PK-style batch sampler
│   │   ├── adapters.py            # .npy / .npz dataset loader utilities
│   │   └── processed/
│   │       ├── natops_train.jsonl # Pre-generated NATOPS training thoughts
│   │       └── natops_valid.jsonl # Pre-generated NATOPS test thoughts
│   │
│   ├── retrieval/
│   │   ├── hybrid_search.py       # HybridRetriever (cosine + DTW)
│   │   ├── evaluate_retrieval.py  # Retrieval evaluation pipeline + CLI
│   │   └── arm.py                 # AdaptiveRetrievalMixer (cross-attention reranker)
│   │
│   ├── configs/
│   │   ├── natops_v1.json
│   │   ├── natops_v2.json
│   │   ├── example_config.json
│   │   ├── generic_timeseries_config.json
│   │   └── minimal_new_dataset_config.json
│   │
│   ├── NATOPS/                    # NATOPS dataset (.npy files)
│   └── README_GENERALIZATION.md   # Guide for adding new datasets
│
├── requirements.txt
└── README.md
```

---

## Key Design Choices

**Why asymmetric embedding?**
Gallery embeddings use only a short context prompt (fast, memory-efficient). Query embeddings include a full reasoning trace, allowing the model to exploit semantic patterns not visible in raw signal statistics alone.

**Why retrieval instead of a classifier head?**
A fixed classifier head requires re-training when new classes are added. Retrieval-based classification is naturally open-set and few-shot: adding a new class means adding examples to the gallery, not retraining.

**Why LoRA?**
The full Qwen2.5-0.5B backbone has ~500M parameters. LoRA reduces the number of trainable LLM parameters to ~2M while keeping the pre-trained knowledge intact, which is critical when the training set is small (e.g., 180 NATOPS samples).

**Why InfoNCE with label-based positives?**
Standard contrastive learning uses only diagonal positives (one query per anchor). With label-based positives, all same-class samples within the batch act as positives, providing a much richer training signal on small balanced datasets.

---

## License

MIT
