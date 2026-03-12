"""
MTS-O1-Embedder Entry Point
"""
import argparse
import os
import random
import warnings

import numpy as np

# Configuration for offline/network-restricted environments
def configure_environment(disable_ssl_verify=False, force_offline=False):
    """
    Configure environment settings for network-restricted environments.

    Args:
        disable_ssl_verify: Disable SSL certificate verification (unsafe, use only in controlled environments)
        force_offline: Force HuggingFace to work offline
    """
    if force_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print("Forced offline mode enabled")

    if disable_ssl_verify:
        import requests
        from requests.packages.urllib3.exceptions import InsecureRequestWarning

        warnings.filterwarnings("ignore", category=InsecureRequestWarning)
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

        # Monkey patch to disable SSL verification (USE WITH CAUTION)
        old_request = requests.Session.request
        def new_request(self, method, url, *args, **kwargs):
            kwargs['verify'] = False
            return old_request(self, method, url, *args, **kwargs)
        requests.Session.request = new_request
        print("SSL verification disabled (UNSAFE - use only in controlled environments)")

import torch
from transformers import AutoTokenizer

# Internal modules
from mts_agent.config import ExperimentConfig, create_config_from_args
from mts_agent.data.adapters import infer_ts_input_dim
from mts_agent.data.generator import generate_thought_data
from mts_agent.data.loader import MTSDataset
from mts_agent.data.collator import MultimodalCollator
from mts_agent.models.mts_embedder import MTSEmbedder
from mts_agent.engine.trainer import MTSTrainer
from mts_agent.tokenization import DummyTokenizer


def set_global_seed(seed, deterministic=True):
    """Set Python, NumPy and PyTorch seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Global random seed set to {seed} (deterministic={deterministic})")


def main():
    parser = argparse.ArgumentParser(description="MTS-O1-Embedder: Multimodal Time Series Embedding Agent")
    parser.add_argument("--mode", type=str, default="train", choices=["gen_data", "train", "inference"], help="Operation mode")
    
    # Paths
    parser.add_argument("--data_path", type=str, default="mts_agent/data/processed/mock_train.jsonl")
    parser.add_argument("--llm_path", type=str, default="Qwen/Qwen2.5-0.5B", help="HuggingFace model path or local path")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible training and evaluation")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--contrastive_temperature", type=float, default=0.07, help="Temperature used by the InfoNCE contrastive loss")
    parser.add_argument(
        "--alignment_text_mode",
        type=str,
        default="full",
        choices=["full", "context", "label", "context_label"],
        help="Text target used by the contrastive alignment branch"
    )
    parser.add_argument(
        "--contrastive_positive_mode",
        type=str,
        default="diagonal",
        choices=["diagonal", "label"],
        help="Positive-pair definition for the contrastive branch"
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--balanced_sampling", action="store_true", help="Use class-balanced PK-style batches during training")
    parser.add_argument("--classes_per_batch", type=int, default=None, help="Number of classes sampled per batch when balanced sampling is enabled")
    parser.add_argument("--samples_per_class", type=int, default=None, help="Number of samples drawn per class when balanced sampling is enabled")
    parser.add_argument("--early_stopping_patience", type=int, default=None, help="Stop training if validation loss does not improve for N epochs")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0, help="Minimum validation loss improvement required to reset early stopping")
    parser.add_argument("--ts_dim", type=int, default=None, help="Time Series input dimension (channels). If omitted, infer automatically")
    parser.add_argument("--ts_base_channels", type=int, default=64, help="Base channel width for the TS encoder")
    parser.add_argument("--ts_tokens", type=int, default=16, help="Target number of TS tokens after adaptive pooling")
    parser.add_argument("--ts_dropout", type=float, default=0.1, help="Dropout used inside the TS encoder")
    parser.add_argument("--ts_norm", type=str, default="group", choices=["group", "batch"], help="Normalization type used by the TS encoder")
    parser.add_argument("--embedding_pooling", type=str, default="mean", choices=["mean", "last", "ts_tokens"], help="Pooling strategy used to derive retrieval embeddings")
    parser.add_argument("--tuning_strategy", type=str, default="lora", choices=["full", "freeze", "partial", "lora"], help="Training strategy: lora (PEFT), full (all), freeze (proj only), partial (top layers)")
    parser.add_argument("--training_stage", type=str, default="alignment", choices=["alignment", "reasoning"], help="Training stage: alignment (contrastive) or reasoning (SFT)")
    parser.add_argument("--contrastive_weight", type=float, default=0.5, help="Weight for contrastive loss in alignment stage")
    parser.add_argument("--hard_negative_weight", type=float, default=0.0, help="Weight for lightweight hard-negative ranking loss")
    parser.add_argument("--hard_negative_margin", type=float, default=0.05, help="Margin used by the hard-negative ranking loss")
    parser.add_argument("--neighbor_weight", type=float, default=0.0, help="Weight for label-aware neighborhood loss in alignment stage")
    parser.add_argument("--neighbor_margin", type=float, default=0.1, help="Margin used by the neighborhood ranking loss")
    parser.add_argument("--lm_weight", type=float, default=1.0, help="Weight for language modeling loss")
    parser.add_argument("--retrieval_eval_enabled", action=argparse.BooleanOptionalAction, default=None, help="Run retrieval metrics on the validation set during training")
    parser.add_argument("--retrieval_eval_k", type=int, default=None, help="Neighbor count used for validation retrieval metrics")
    parser.add_argument("--retrieval_eval_alpha", type=float, default=None, help="Semantic/DTW fusion weight used for validation retrieval metrics")
    parser.add_argument("--retrieval_eval_vote_strategy", type=str, default="weighted", choices=["majority", "weighted", "rank", "semantic", "structural", "prototype", "hybrid_prototype"], help="Voting strategy used for validation retrieval metrics")
    parser.add_argument("--dtw_window_size", type=int, default=None, help="Optional Sakoe-Chiba DTW window size for retrieval")
    parser.add_argument("--fast_dtw_max_len", type=int, default=100, help="Maximum length before DTW downsampling")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--raw_data_path", type=str, default="dummy.npy", help="Path to raw .npy file or folder for data generation")
    parser.add_argument("--dataset_name", type=str, default=None, help="Optional dataset name override for generic data generation")
    parser.add_argument("--use_dummy_tokenizer", action="store_true", help="Force use of DummyTokenizer for offline mode")
    parser.add_argument("--disable_ssl_verify", action="store_true", help="Disable SSL certificate verification (UNSAFE)")
    parser.add_argument("--force_offline", action="store_true", help="Force HuggingFace offline mode")
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    parser.add_argument("--save_config", type=str, help="Save current configuration to file")

    args = parser.parse_args()

    # Load configuration
    if args.config:
        print(f"Loading configuration from {args.config}")
        config = ExperimentConfig.load(args.config)
    else:
        config = create_config_from_args(args)

    # Save configuration if requested
    if args.save_config:
        config.save(args.save_config)
        print(f"Configuration saved to {args.save_config}")
        return

    # Configure environment based on config
    configure_environment(
        disable_ssl_verify=config.environment.disable_ssl_verify,
        force_offline=config.environment.force_offline
    )
    set_global_seed(config.training.seed)

    inferred_ts_dim = infer_ts_input_dim(
        data_path=config.data.data_path,
        raw_data_path=config.data.raw_data_path,
        dataset_name=config.data.dataset_name
    )
    if config.data.ts_dim is None and inferred_ts_dim is not None:
        config.data.ts_dim = inferred_ts_dim
    if config.model.ts_input_dim is None and inferred_ts_dim is not None:
        config.model.ts_input_dim = inferred_ts_dim
    if config.model.ts_input_dim is None:
        config.model.ts_input_dim = config.data.ts_dim or 1
    if config.data.ts_dim is None:
        config.data.ts_dim = config.model.ts_input_dim
    print(f"Resolved time-series input dimension: {config.model.ts_input_dim}")

    if config.mode == "gen_data":
        print(f"Starting data generation from {config.data.raw_data_path}...")
        generate_thought_data(
            config.data.raw_data_path,
            config.data.data_path,
            dataset_name=config.data.dataset_name or "General_TS",
            api_key=config.inference.api_key,
        )

    elif config.mode == "train":
        print(f"=== Starting Training Setup: {config.experiment_name} ===")

        # 1. Load Tokenizer
        if config.environment.use_dummy_tokenizer:
            print("Force using DummyTokenizer (Offline Mode)...")
            tokenizer = DummyTokenizer(vocab_size=32000)
        else:
            print(f"Loading Tokenizer from {config.model.llm_path}...")
            try:
                print(f" -> Checking local cache for {config.model.llm_path}...")
                tokenizer = AutoTokenizer.from_pretrained(config.model.llm_path, trust_remote_code=True, local_files_only=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                print(f" -> Successfully loaded {config.model.llm_path} from local cache.")
            except Exception as e_local:
                print(f" -> Not found in local cache: {e_local}")
                print(" -> Network is blocked/unstable. Falling back to DummyTokenizer immediately.")
                tokenizer = DummyTokenizer(vocab_size=32000)

        # 2. Data Preparation
        if not os.path.exists(config.data.data_path):
            print(f"Data file {config.data.data_path} not found. Running generation first.")
            generate_thought_data("dummy.npy", config.data.data_path)

        print("Initializing Train Dataset...")
        full_train_dataset = MTSDataset(
            config.data.data_path,
            mode='train',
            augment=config.data.augment,
            domain_info=config.data.domain_info
        )

        # Split: internal validation (20%, stratified by label) for CE val_loss early stopping.
        # The held-out test file (*_valid.jsonl) is NEVER seen during training.
        import random as _random
        _rng = _random.Random(config.training.seed)
        from collections import defaultdict as _defaultdict
        _per_class = _defaultdict(list)
        for s in full_train_dataset.samples:
            _per_class[str(s.get('label', ''))].append(s)
        _train_samples, _val_samples = [], []
        for _cls, _cls_samples in _per_class.items():
            _cls_samples = list(_cls_samples)
            _rng.shuffle(_cls_samples)
            _n_val = max(1, int(len(_cls_samples) * 0.2))
            _val_samples.extend(_cls_samples[:_n_val])
            _train_samples.extend(_cls_samples[_n_val:])
        train_dataset = MTSDataset(
            config.data.data_path, mode='train',
            augment=config.data.augment,
            domain_info=config.data.domain_info,
            samples=_train_samples
        )
        eval_dataset = MTSDataset(
            config.data.data_path, mode='train', augment=False,
            domain_info=config.data.domain_info,
            samples=_val_samples
        )
        print(f"Internal split: {len(_train_samples)} train / {len(_val_samples)} internal-val (stratified 80/20).")

        # Load the held-out test file: used ONLY in final_test_retrieval(), never during training.
        test_dataset = None
        if "train" in config.data.data_path:
            _test_path = config.data.data_path.replace("train", "valid")
            if os.path.exists(_test_path):
                print(f"Loading held-out test set from {_test_path}...")
                test_dataset = MTSDataset(_test_path, mode='train', augment=False,
                                          domain_info=config.data.domain_info)

        collator = MultimodalCollator(
            tokenizer,
            mode='train',
            max_length=config.data.max_length,
            alignment_text_mode=config.training.alignment_text_mode
        )

        # 3. Model Initialization
        print("Initializing Model...")
        model = MTSEmbedder(
            config.model.llm_path,
            ts_input_dim=config.model.ts_input_dim,
            ts_hidden_dim=config.model.ts_hidden_dim,
            encoder_base_channels=config.model.encoder_base_channels,
            encoder_target_tokens=config.model.encoder_target_tokens,
            encoder_dropout=config.model.encoder_dropout,
            encoder_norm=config.model.encoder_norm,
            embedding_pooling=config.model.embedding_pooling,
        )

        # Tuning Strategy
        if config.model.tuning_strategy == "freeze":
            print("Strategy: Freezing LLM Backbone (Stage 1)...")
            model.freeze_llm()
        elif config.model.tuning_strategy == "partial":
            print(f"Strategy: Smart Freezing (Top {config.model.num_trainable_layers} Layers + Head)...")
            model.smart_freeze_llm(num_trainable_layers=config.model.num_trainable_layers)
        elif config.model.tuning_strategy == "lora":
            print("Strategy: LoRA Adaptation...")
            model.apply_lora(r=8, lora_alpha=16)
        else:
            print("Strategy: Full Parameter Tuning...")

        # Ensure Projector + Encoder are always trainable
        for param in model.projector.parameters():
            param.requires_grad = True
        for param in model.ts_encoder.parameters():
            param.requires_grad = True

        # Load Checkpoint if provided
        if config.training.ckpt_path and os.path.exists(config.training.ckpt_path):
            print(f"Loading checkpoint from {config.training.ckpt_path}...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state_dict = torch.load(config.training.ckpt_path, map_location=device)

            # Filter out shape mismatches
            model_state = model.state_dict()
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k in model_state:
                    if v.shape == model_state[k].shape:
                        filtered_state_dict[k] = v
                    else:
                        print(f"Skipping layer {k} due to shape mismatch: {v.shape} vs {model_state[k].shape}")
                else:
                    print(f"Skipping unknown layer {k}")

            try:
                model.load_state_dict(filtered_state_dict, strict=False)
                print("Checkpoint loaded (partial/filtered) successfully.")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")

        # 4. Trainer Initialization
        run_config = {
            'epochs': config.training.epochs,
            'batch_size': config.data.batch_size,
            'lr': config.training.lr,
            'contrastive_temperature': config.training.contrastive_temperature,
            'alignment_text_mode': config.training.alignment_text_mode,
            'contrastive_positive_mode': config.training.contrastive_positive_mode,
            'gradient_accumulation_steps': config.training.gradient_accumulation_steps,
            'balanced_sampling': config.training.balanced_sampling,
            'classes_per_batch': config.training.classes_per_batch,
            'samples_per_class': config.training.samples_per_class,
            'early_stopping_patience': config.training.early_stopping_patience,
            'early_stopping_min_delta': config.training.early_stopping_min_delta,
            'save_dir': config.training.save_dir,
            'training_stage': config.training.training_stage,
            'contrastive_weight': config.training.contrastive_weight,
            'hard_negative_weight': config.training.hard_negative_weight,
            'hard_negative_margin': config.training.hard_negative_margin,
            'neighbor_weight': config.training.neighbor_weight,
            'neighbor_margin': config.training.neighbor_margin,
            'lm_weight': config.training.lm_weight,
            'cls_weight': config.training.cls_weight,
            'retrieval_eval_enabled': config.training.retrieval_eval_enabled,
            'retrieval_eval_k': config.training.retrieval_eval_k or config.retrieval.k,
            'retrieval_eval_alpha': config.training.retrieval_eval_alpha if config.training.retrieval_eval_alpha is not None else config.retrieval.alpha,
            'retrieval_eval_vote_strategy': config.training.retrieval_eval_vote_strategy,
            'retrieval_eval_interval': config.training.retrieval_eval_interval,
            'use_amp': config.training.use_amp,
            'memory_bank_size': config.training.memory_bank_size,
            'inference_max_new_tokens': config.inference.max_new_tokens,
            'inference_temperature': config.inference.temperature,
            'inference_do_sample': config.inference.do_sample,
            'retrieval_eval_dtw_window_size': config.retrieval.dtw_window_size,
            'retrieval_eval_fast_dtw_max_len': config.retrieval.fast_dtw_max_len,
            # ARM (Adaptive Retrieval Mixer)
            'arm_enabled':         config.arm.enabled,
            'arm_k':               config.arm.k,
            'arm_epochs':          config.arm.epochs,
            'arm_lr':              config.arm.lr,
            'arm_num_heads':       config.arm.num_heads,
            'arm_dropout':         config.arm.dropout,
            'arm_augment_gallery': config.arm.augment_gallery,
            'arm_augment_factor':  config.arm.augment_factor,
            'ts_only_embedding':   config.training.ts_only_embedding,
        }

        trainer = MTSTrainer(model, train_dataset, collator, run_config,
                              eval_dataset=eval_dataset, test_dataset=test_dataset)

        # 5. Start Training
        trainer.train()
        
    elif config.mode == "inference":
        print(f"=== Starting Inference: {config.experiment_name} ===")

        # 1. Load Tokenizer
        if config.environment.use_dummy_tokenizer:
            print("Using DummyTokenizer (Offline Mode)...")
            tokenizer = DummyTokenizer(vocab_size=32000)
        else:
            print(f"Loading Tokenizer from {config.model.llm_path}...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(config.model.llm_path, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            except Exception:
                print("Warning: Network issue. Using DummyTokenizer.")
                tokenizer = DummyTokenizer(vocab_size=32000)

        # 2. Dataset
        print(f"Loading Test Data from {config.data.data_path}...")
        dataset = MTSDataset(
            config.data.data_path,
            mode='inference',
            domain_info=config.data.domain_info
        )
        collator = MultimodalCollator(
            tokenizer,
            mode='inference',
            max_length=config.data.max_length,
            alignment_text_mode=config.training.alignment_text_mode
        )

        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=1, collate_fn=collator)

        # 3. Model
        print("Initializing Model...")
        model = MTSEmbedder(
            config.model.llm_path,
            ts_input_dim=config.model.ts_input_dim,
            ts_hidden_dim=config.model.ts_hidden_dim,
            encoder_base_channels=config.model.encoder_base_channels,
            encoder_target_tokens=config.model.encoder_target_tokens,
            encoder_dropout=config.model.encoder_dropout,
            encoder_norm=config.model.encoder_norm,
            embedding_pooling=config.model.embedding_pooling,
        )

        # 4. Load Checkpoint
        ckpt_path = config.training.ckpt_path
        if not ckpt_path or not os.path.exists(ckpt_path):
            # Fallback logic
            default_ckpt = os.path.join(config.training.save_dir, "model_last.pt")
            if os.path.exists(default_ckpt):
                ckpt_path = default_ckpt
            else:
                for f in os.listdir(config.training.save_dir):
                    if f.endswith(".pt"):
                        ckpt_path = os.path.join(config.training.save_dir, f)
                        break

        # Set device from config
        if config.environment.device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")

        if ckpt_path and os.path.exists(ckpt_path):
            print(f"Loading weights from {ckpt_path}...")
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict)
        else:
            print("No checkpoint found! Using random weights.")

        model.to(device)
        model.eval()
        
        # 5. Generate and Evaluate
        print("\n--- Generating Thoughts & Evaluating ---")
        correct_count = 0
        total_count = 0
        
        # Helper to parse label
        def parse_label(text):
            text = text.lower()
            # Simple heuristic: The later mention is likely the conclusion
            idx_left = text.rfind("left")
            idx_right = text.rfind("right")
            
            if idx_left == -1 and idx_right == -1:
                return "unknown"
            if idx_left > idx_right:
                return "left"
            else:
                return "right"

        for i, batch in enumerate(loader):
            # Process all samples
            # if i >= 3: break 
            
            ts_input = batch['ts_input'].to(device).float()
            text_input_ids = batch['text_input_ids'].to(device)
            
            # Ground truth
            item = dataset.samples[i]
            true_label = item.get('label', 'unknown').lower()
            
            with torch.no_grad():
                outputs = model.generate(
                    ts_input,
                    text_input_ids,
                    max_new_tokens=config.inference.max_new_tokens,
                    do_sample=config.inference.do_sample,
                    temperature=config.inference.temperature,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Predict
            pred_label = parse_label(decoded)
            
            is_correct = (pred_label == true_label)
            if is_correct:
                correct_count += 1
            total_count += 1
            
            # Print periodic progress
            if i < 3 or i % 10 == 0:
                print(f"\n[Sample {i+1}]")
                # print(f"Generated: {decoded[-200:]}...") # Print end of thought
                print(f"Prediction: {pred_label.upper()} | Truth: {true_label.upper()} | Correct: {is_correct}")
                if i < 3: # Print full thought for first few
                     print(f"Full Thought:\n{decoded}")

        accuracy = correct_count / total_count if total_count > 0 else 0
        print(f"\n=== Evaluation Result ===")
        print(f"Total Samples: {total_count}")
        print(f"Correct: {correct_count}")
        print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
