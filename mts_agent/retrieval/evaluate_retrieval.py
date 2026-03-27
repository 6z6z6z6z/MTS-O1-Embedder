"""
Evaluation script for hybrid retrieval system.
Tests retrieval accuracy using k-NN classification.
"""
import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from mts_agent.models.mts_embedder import MTSEmbedder
from mts_agent.data.adapters import infer_ts_input_dim
from mts_agent.data.loader import MTSDataset
from mts_agent.data.collator import MultimodalCollator
from mts_agent.data.prompt_builder import build_retrieval_prompt, build_full_prompt
from mts_agent.retrieval.hybrid_search import HybridRetriever
from transformers import AutoTokenizer


def _score_to_weight(score, temperature=1.0):
    temperature = max(float(temperature), 1e-6)
    return max(float(score), 0.0) / temperature


def _normalize_score_dict(score_dict):
    if not score_dict:
        return {}
    max_score = max(score_dict.values())
    min_score = min(score_dict.values())
    if max_score <= min_score:
        return {key: 1.0 for key in score_dict}
    return {key: (value - min_score) / (max_score - min_score) for key, value in score_dict.items()}


def _clamp_alpha(alpha):
    return float(np.clip(float(alpha), 0.0, 1.0))


def _safe_k(k):
    return max(1, int(k))


def _aligned_length_with_warning(name, *seqs):
    lengths = [len(seq) for seq in seqs]
    if not lengths:
        return 0
    min_len = min(lengths)
    max_len = max(lengths)
    if min_len != max_len:
        print(f"Warning: {name} length mismatch {lengths}. Using first {min_len} samples.")
    return min_len


def _load_checkpoint_filtered(model, ckpt_path, device):
    """Load checkpoint weights with shape filtering for compatibility."""
    if not ckpt_path:
        return
    if not os.path.exists(ckpt_path):
        print(f"Warning: checkpoint not found: {ckpt_path}")
        return

    print(f"Loading checkpoint from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model_state = model.state_dict()
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if key in model_state and hasattr(value, "shape") and hasattr(model_state[key], "shape"):
            if value.shape == model_state[key].shape:
                filtered_state_dict[key] = value
            else:
                print(f"Skipping layer {key} due to shape mismatch: {value.shape} vs {model_state[key].shape}")
        elif key in model_state:
            filtered_state_dict[key] = value
        else:
            print(f"Skipping unknown layer {key}")

    model.load_state_dict(filtered_state_dict, strict=False)


def build_class_prototypes(embeddings, labels):
    """Build class prototype statistics from cached embeddings."""
    prototype_sums = {}
    prototype_counts = {}
    for embedding, label in zip(embeddings, labels):
        label = str(label)
        if label not in prototype_sums:
            prototype_sums[label] = embedding.copy()
            prototype_counts[label] = 0
        else:
            prototype_sums[label] += embedding
        prototype_counts[label] += 1
    return prototype_sums, prototype_counts


def compute_prototype_scores(query_emb, true_label, prototype_sums, prototype_counts):
    """Compute leave-one-out cosine similarity against class prototypes."""
    scores = {}
    q_norm = query_emb / (float((query_emb ** 2).sum()) ** 0.5 + 1e-10)
    true_label = str(true_label)

    for label, proto_sum in prototype_sums.items():
        count = prototype_counts[label]
        centroid = proto_sum
        denom = count
        if label == true_label and count > 1:
            centroid = proto_sum - query_emb
            denom = count - 1
        if denom <= 0:
            continue
        centroid = centroid / denom
        c_norm = centroid / (float((centroid ** 2).sum()) ** 0.5 + 1e-10)
        scores[label] = float((q_norm * c_norm).sum())
    return _normalize_score_dict(scores)


def aggregate_neighbor_labels(neighbors, vote_strategy="weighted", weight_temperature=1.0):
    """Aggregate neighbor labels with configurable voting rules."""
    if not neighbors:
        return None, {}

    class_scores = {}
    for rank, neighbor in enumerate(neighbors, start=1):
        label = neighbor.get('label')
        if label is None:
            label = neighbor['id'].split('_')[-1]
        label = str(label)

        if vote_strategy == "majority":
            weight = 1.0
        elif vote_strategy == "rank":
            weight = 1.0 / rank
        elif vote_strategy == "semantic":
            weight = _score_to_weight(neighbor.get('sem_score', 0.0), weight_temperature)
        elif vote_strategy == "structural":
            weight = _score_to_weight(neighbor.get('struct_score', 0.0), weight_temperature)
        else:
            weight = _score_to_weight(neighbor.get('score', 0.0), weight_temperature)

        class_scores[label] = class_scores.get(label, 0.0) + weight

    pred_label = max(class_scores.items(), key=lambda x: (x[1], x[0]))[0]
    return pred_label, class_scores


def aggregate_with_prototypes(neighbor_scores, prototype_scores, prototype_weight=0.5):
    """Fuse neighbor vote scores with class prototype similarities."""
    neighbor_scores = _normalize_score_dict(neighbor_scores)
    prototype_scores = _normalize_score_dict(prototype_scores)
    labels = set(neighbor_scores) | set(prototype_scores)
    fused_scores = {}
    for label in labels:
        fused_scores[label] = (1.0 - prototype_weight) * neighbor_scores.get(label, 0.0) + prototype_weight * prototype_scores.get(label, 0.0)
    if not fused_scores:
        return None, {}
    pred_label = max(fused_scores.items(), key=lambda x: (x[1], x[0]))[0]
    return pred_label, fused_scores


def _extract_embeddings_for_dataset(
    model,
    dataset,
    collator,
    device,
    *,
    use_full_prompt,
    ts_only_embedding,
    truncation,
):
    """Encode a dataset into retrieval embeddings plus aligned labels/series.

    truncation=False keeps legacy behavior for cache/LOO mode.
    truncation=True keeps gallery-building behavior for cross-set evaluation.
    """
    if len(dataset) == 0:
        print("Warning: dataset is empty, no embeddings extracted.")
        return [], [], []

    embeddings, labels, ts_data_list = [], [], []

    print("Extracting embeddings for index...")
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        ts_input = item['time_series'].unsqueeze(0).to(device).float()

        with torch.no_grad():
            if ts_only_embedding:
                embedding = model.get_ts_only_embedding(ts_input)
            else:
                context = item['context']
                thought = item.get('teacher_thought')
                if use_full_prompt and thought:
                    prompt = build_full_prompt(context, thought=thought, include_response_stub=False)
                else:
                    prompt = build_retrieval_prompt(context)

                tokenizer_kwargs = {'return_tensors': 'pt'}
                if truncation:
                    tokenizer_kwargs.update({'truncation': True, 'max_length': collator.max_length})

                inputs = collator.tokenizer(prompt, **tokenizer_kwargs)
                text_input_ids = inputs.input_ids.to(device)
                attention_mask = inputs.attention_mask.to(device) if hasattr(inputs, 'attention_mask') else None
                embedding = model.get_embedding(ts_input, text_input_ids, attention_mask=attention_mask)
            embedding = embedding.cpu().numpy().flatten()

        embeddings.append(embedding)
        labels.append(item['label'])
        ts_data_list.append(item['time_series'].cpu().numpy())

    return embeddings, labels, ts_data_list

def build_retrieval_cache(model, dataset, collator, device, dtw_window_size=None, fast_dtw_max_len=100, use_full_prompt=True, ts_only_embedding=False):
    """Extract embeddings once and build the retrieval index cache.

    Args:
        use_full_prompt: If True, use context+thought prompts when available.
            If False, always use context-only prompts.
        ts_only_embedding: If True, bypass the LLM and embed using only the
            TS encoder + projector. Eliminates text-identity bias.
    """
    print("Building retrieval index...")
    retriever = HybridRetriever(dtw_window_size=dtw_window_size, fast_dtw_max_len=fast_dtw_max_len)
    embeddings, labels, ts_data_list = _extract_embeddings_for_dataset(
        model,
        dataset,
        collator,
        device,
        use_full_prompt=use_full_prompt,
        ts_only_embedding=ts_only_embedding,
        truncation=False,
    )

    for i, (emb, ts, label) in enumerate(zip(embeddings, ts_data_list, labels)):
        retriever.add(f"sample_{i}_{label}", emb, ts, label=label)
    if embeddings:
        retriever.build_index()

    print(f"Index built with {len(embeddings)} samples")
    return retriever, embeddings, labels, ts_data_list


def build_gallery(model, dataset, collator, device, dtw_window_size=None, fast_dtw_max_len=100, use_full_prompt=True, ts_only_embedding=False):
    """Build retrieval gallery from a labeled dataset (typically training set).

    Args:
        use_full_prompt: If True, embed each item using its full prompt (context +
            teacher_thought). If False, use context-only prompts.
        ts_only_embedding: If True, bypass the LLM and embed using only the
            TS encoder + projector. Eliminates text-identity bias.
    """
    retriever = HybridRetriever(dtw_window_size=dtw_window_size, fast_dtw_max_len=fast_dtw_max_len)
    embeddings, labels, ts_data_list = _extract_embeddings_for_dataset(
        model,
        dataset,
        collator,
        device,
        use_full_prompt=use_full_prompt,
        ts_only_embedding=ts_only_embedding,
        truncation=True,
    )

    for i, (emb, ts, label) in enumerate(zip(embeddings, ts_data_list, labels)):
        retriever.add(f"train_{i}_{label}", emb, ts, label=label)
    if embeddings:
        retriever.build_index()
    print(f"Index built with {len(embeddings)} samples")
    return retriever, embeddings, labels, ts_data_list


def embed_queries(model, dataset, collator, device, ts_only_embedding=False, use_full_prompt=True):
    """Embed query samples.

    Args:
        use_full_prompt: When True (default), uses thought-enriched prompts for
            queries matching the asymmetric O1-Embedder training distribution.
            When False, uses context-only prompts — use this when training with
            ``alignment_text_mode != "full"`` to keep eval format consistent.
    """
    embeddings, labels, ts_data_list = [], [], []

    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        ts_input = item['time_series'].unsqueeze(0).to(device).float()

        with torch.no_grad():
            if ts_only_embedding:
                emb = model.get_ts_only_embedding(ts_input)
            else:
                if use_full_prompt:
                    thought = str(item.get('teacher_thought') or item.get('thought') or '').strip()
                    if thought:
                        prompt = build_full_prompt(item['context'], thought=thought, include_response_stub=False)
                    else:
                        prompt = build_retrieval_prompt(item['context'])
                else:
                    prompt = build_retrieval_prompt(item['context'])
                inputs = collator.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=collator.max_length)
                text_input_ids = inputs.input_ids.to(device)
                attention_mask = inputs.attention_mask.to(device)
                emb = model.get_embedding(ts_input, text_input_ids, attention_mask=attention_mask)
            emb = emb.cpu().numpy().flatten()

        embeddings.append(emb)
        labels.append(item['label'])
        ts_data_list.append(item['time_series'].cpu().numpy())

    return embeddings, labels, ts_data_list


def evaluate_gallery_vs_queries(
    retriever,
    query_embeddings,
    query_labels,
    query_ts_list,
    k=5,
    alpha=0.8,
    vote_strategy="weighted",
):
    """Evaluate retrieval accuracy when gallery and queries are different datasets.

    No self-exclusion is needed since query samples are not in the gallery.
    """
    print("Evaluating retrieval accuracy...")
    total = _aligned_length_with_warning(
        "gallery-vs-query evaluation",
        query_embeddings,
        query_labels,
        query_ts_list,
    )
    if total == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "predictions": [], "references": []}

    k = _safe_k(k)
    alpha = _clamp_alpha(alpha)
    predictions, references = [], []

    for i in tqdm(range(total)):
        query_emb = query_embeddings[i]
        query_ts = query_ts_list[i]
        true_label = query_labels[i]

        results = retriever.search(query_emb, query_ts, k=k, alpha=alpha)
        pred_label, _ = aggregate_neighbor_labels(results, vote_strategy=vote_strategy)
        if pred_label is None:
            pred_label = true_label

        predictions.append(str(pred_label))
        references.append(str(true_label))

    accuracy = accuracy_score(references, predictions) if references else 0.0
    macro_f1 = f1_score(references, predictions, average='macro') if references else 0.0
    print(
        f"Retrieval Accuracy (k={k}, alpha={alpha}, vote={vote_strategy}): {accuracy:.2%} | "
        f"Macro-F1: {macro_f1:.2%}"
    )
    return {"accuracy": accuracy, "macro_f1": macro_f1, "predictions": predictions, "references": references}



def evaluate_retrieval_from_cache(
    retriever,
    embeddings,
    labels,
    ts_data_list,
    k=5,
    alpha=0.8,
    vote_strategy="weighted",
    weight_temperature=1.0,
    prototype_weight=0.5
):
    """Evaluate retrieval performance from cached embeddings and index."""
    print("Evaluating retrieval accuracy...")
    total = _aligned_length_with_warning(
        "cache evaluation",
        embeddings,
        labels,
        ts_data_list,
    )
    if total == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "predictions": [],
            "references": [],
        }

    k = _safe_k(k)
    alpha = _clamp_alpha(alpha)
    predictions = []
    references = []
    prototype_sums, prototype_counts = build_class_prototypes(embeddings[:total], labels[:total])

    for i in tqdm(range(total)):
        query_emb = embeddings[i]
        query_ts = ts_data_list[i]
        true_label = labels[i]
        query_id = f"sample_{i}_{true_label}"

        results = retriever.search(query_emb, query_ts, k=k+1, alpha=alpha)

        filtered_neighbors = [result for result in results if result['id'] != query_id][:k]
        pred_label, class_scores = aggregate_neighbor_labels(
            filtered_neighbors,
            vote_strategy="weighted" if vote_strategy in ["prototype", "hybrid_prototype"] else vote_strategy,
            weight_temperature=weight_temperature
        )
        if vote_strategy in ["prototype", "hybrid_prototype"]:
            prototype_scores = compute_prototype_scores(query_emb, true_label, prototype_sums, prototype_counts)
            if vote_strategy == "prototype":
                pred_label, class_scores = aggregate_with_prototypes({}, prototype_scores, prototype_weight=1.0)
            else:
                pred_label, class_scores = aggregate_with_prototypes(class_scores, prototype_scores, prototype_weight=prototype_weight)
        if pred_label is None:
            pred_label = true_label

        predictions.append(str(pred_label))
        references.append(str(true_label))

    accuracy = accuracy_score(references, predictions) if references else 0.0
    macro_f1 = f1_score(references, predictions, average='macro') if references else 0.0
    print(
        f"Retrieval Accuracy (k={k}, alpha={alpha}, vote={vote_strategy}): {accuracy:.2%} | "
        f"Macro-F1: {macro_f1:.2%}"
    )
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "predictions": predictions,
        "references": references
    }


def evaluate_retrieval(model, dataset, collator, device, k=5, alpha=0.8, dtw_window_size=None, fast_dtw_max_len=100):
    """
    Evaluate retrieval performance using hybrid search.
    """
    retriever, embeddings, labels, ts_data_list = build_retrieval_cache(
        model,
        dataset,
        collator,
        device,
        dtw_window_size=dtw_window_size,
        fast_dtw_max_len=fast_dtw_max_len
    )
    metrics = evaluate_retrieval_from_cache(retriever, embeddings, labels, ts_data_list, k=k, alpha=alpha)
    return metrics, retriever


def parse_alpha_grid(alpha, alpha_grid):
    if alpha_grid:
        values = []
        for value in alpha_grid.split(','):
            value = value.strip()
            if value:
                try:
                    values.append(float(value))
                except ValueError as exc:
                    raise ValueError(f"Invalid alpha value in --alpha_grid: {value}") from exc
        if values:
            return values
    return [alpha]

def main():
    parser = argparse.ArgumentParser(description="Evaluate hybrid retrieval system")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--llm_path", type=str, default="Qwen/Qwen2.5-0.5B", help="LLM model path")
    parser.add_argument("--ckpt_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--ts_dim", type=int, default=None, help="Time series dimension. If omitted, infer automatically")
    parser.add_argument("--ts_base_channels", type=int, default=64, help="Base channel width for the TS encoder")
    parser.add_argument("--ts_tokens", type=int, default=16, help="Target number of TS tokens after adaptive pooling")
    parser.add_argument("--ts_dropout", type=float, default=0.1, help="Dropout used inside the TS encoder")
    parser.add_argument("--ts_norm", type=str, default="group", choices=["group", "batch"], help="Normalization type used by the TS encoder")
    parser.add_argument("--embedding_pooling", type=str, default="mean", choices=["mean", "last"], help="Pooling strategy used to derive retrieval embeddings")
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors for retrieval")
    parser.add_argument("--alpha", type=float, default=0.8, help="Weight for semantic similarity")
    parser.add_argument("--alpha_grid", type=str, default=None, help="Comma-separated alpha values to evaluate, e.g. 0.3,0.5,0.7")
    parser.add_argument("--dtw_window_size", type=int, default=None, help="Optional Sakoe-Chiba DTW window size")
    parser.add_argument("--fast_dtw_max_len", type=int, default=100, help="Maximum length before downsampling for DTW")
    parser.add_argument("--save_results", type=str, default=None, help="Optional path to save alpha sweep results as JSON")
    parser.add_argument("--vote_strategy", type=str, default="weighted", choices=["majority", "weighted", "rank", "semantic", "structural", "prototype", "hybrid_prototype"], help="How to aggregate labels from retrieved neighbors")
    parser.add_argument("--weight_temperature", type=float, default=1.0, help="Temperature scaling for weighted voting")
    parser.add_argument("--prototype_weight", type=float, default=0.5, help="Fusion weight for prototype-assisted voting")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--lora", action="store_true", help="Use lora for loading")
    parser.add_argument("--ts_only_embedding", action="store_true", help="Use TS-only embeddings (bypass LLM) for gallery/query encoding")
    parser.add_argument(
        "--gallery_path", type=str, default=None,
        help="If set, use this dataset as the gallery (train set) and --data_path as queries (test set). "
             "Embeddings are context-only for both sides."
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer (offline-first)
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = MTSDataset(args.data_path, mode='inference')
    collator = MultimodalCollator(tokenizer, mode='inference')
    resolved_ts_dim = args.ts_dim if args.ts_dim is not None else infer_ts_input_dim(data_path=args.data_path)
    if resolved_ts_dim is None:
        resolved_ts_dim = 1
    print(f"Resolved time-series input dimension: {resolved_ts_dim}")

    # Load model
    model = MTSEmbedder(
        args.llm_path,
        ts_hidden_dim=128,
        encoder_base_channels=args.ts_base_channels,
        encoder_dropout=args.ts_dropout,
        encoder_norm=args.ts_norm,
        embedding_pooling=args.embedding_pooling
    )

    # Added dynamically for peft models from previous steps
    if args.lora:
        print("Applying PEFT LoRA before loading weights...")
        model.apply_lora(r=8, lora_alpha=16, lora_dropout=0.05)

    _load_checkpoint_filtered(model, args.ckpt_path, device)

    model.to(device)
    model.eval()

    alpha_values = parse_alpha_grid(args.alpha, args.alpha_grid)

    # ── Gallery + Query mode (train set as gallery, test set as queries) ──────
    if args.gallery_path:
        gallery_dataset = MTSDataset(args.gallery_path, mode='inference')
        query_dataset = MTSDataset(args.data_path, mode='inference')
        print(f"Gallery: {args.gallery_path} ({len(gallery_dataset)} samples)")
        print(f"Queries: {args.data_path} ({len(query_dataset)} samples)")

        retriever, _, _, _ = build_gallery(
            model, gallery_dataset, collator, device,
            dtw_window_size=args.dtw_window_size,
            fast_dtw_max_len=args.fast_dtw_max_len,
            use_full_prompt=False,
            ts_only_embedding=args.ts_only_embedding
        )
        query_embeddings, query_labels, query_ts_list = embed_queries(
            model, query_dataset, collator, device,
            ts_only_embedding=args.ts_only_embedding
        )

        results = []
        best_result = None
        for alpha in alpha_values:
            metrics = evaluate_gallery_vs_queries(
                retriever, query_embeddings, query_labels, query_ts_list,
                k=args.k, alpha=alpha, vote_strategy=args.vote_strategy
            )
            result = {
                "alpha": alpha,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "k": args.k,
                "gallery_size": len(gallery_dataset),
                "query_size": len(query_dataset),
                "vote_strategy": args.vote_strategy
            }
            results.append(result)
            if best_result is None or metrics["accuracy"] > best_result["accuracy"]:
                best_result = result

        if len(results) > 1 and best_result is not None:
            print(
                f"Best alpha: {best_result['alpha']:.4f} | "
                f"Accuracy: {best_result['accuracy']:.2%} | "
                f"Macro-F1: {best_result['macro_f1']:.2%}"
            )
        if args.save_results:
            with open(args.save_results, 'w', encoding='utf-8') as f:
                json.dump({"results": results, "best": best_result}, f, indent=2)
            print(f"Saved results to {args.save_results}")
        return best_result["accuracy"] if best_result is not None else 0.0

    # ── Single-dataset leave-one-out mode ────────────────────────────────────
    retriever, embeddings, labels, ts_data_list = build_retrieval_cache(
        model,
        dataset,
        collator,
        device,
        dtw_window_size=args.dtw_window_size,
        fast_dtw_max_len=args.fast_dtw_max_len,
        ts_only_embedding=args.ts_only_embedding
    )

    results = []
    best_result = None
    for alpha in alpha_values:
        metrics = evaluate_retrieval_from_cache(
            retriever,
            embeddings,
            labels,
            ts_data_list,
            k=args.k,
            alpha=alpha,
            vote_strategy=args.vote_strategy,
            weight_temperature=args.weight_temperature,
            prototype_weight=args.prototype_weight
        )
        result = {
            "alpha": alpha,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "k": args.k,
            "dataset_size": len(labels),
            "vote_strategy": args.vote_strategy,
            "prototype_weight": args.prototype_weight
        }
        results.append(result)
        if best_result is None or metrics["accuracy"] > best_result["accuracy"]:
            best_result = result

    if len(results) > 1 and best_result is not None:
        print(
            f"Best alpha: {best_result['alpha']:.4f} | "
            f"Accuracy: {best_result['accuracy']:.2%} | "
            f"Macro-F1: {best_result['macro_f1']:.2%}"
        )

    if args.save_results:
        with open(args.save_results, 'w', encoding='utf-8') as f:
            json.dump({"results": results, "best": best_result}, f, indent=2)
        print(f"Saved sweep results to {args.save_results}")

    # Save index for future use
    index_path = "retrieval_index.pkl"
    retriever.save_index(index_path)
    print(f"Retrieval index saved to {index_path}")

    return best_result["accuracy"] if best_result is not None else 0.0

if __name__ == "__main__":
    main()