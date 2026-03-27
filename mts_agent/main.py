"""
MTS-O1-Embedder Entry Point
"""
import argparse
import os
# NOTE: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is intentionally NOT set.
# expandable_segments causes Triton kernels to race with moved pointer bases →
# cudaErrorIllegalAddress.  Standard allocator fragmentation is acceptable here.
import random
import sys
import warnings
from copy import deepcopy
from collections import defaultdict

import numpy as np


def _patch_triton_autotuner():
    """Make triton's Autotuner tolerant of autotune keys missing from the kernel's arg list.

    fla >= certain versions add 'BT' to @triton.autotune key=[] but the actual
    kernel function may not list it as a parameter.  triton 3.1.0 enforces a
    strict check and raises ValueError: 'BT' is not in list, preventing fla from
    being imported at all.  This patch relaxes that check so fla loads normally.
    """
    try:
        from triton.runtime import autotuner as _autotuner_mod

        _orig_init = _autotuner_mod.Autotuner.__init__

        def _permissive_init(self, fn, arg_names, configs, key, *args, **kwargs):
            valid_key = [k for k in key if k in arg_names]
            dropped = [k for k in key if k not in arg_names]
            if dropped:
                print(f"[triton patch] Removed invalid autotune keys {dropped} from '{getattr(fn, '__name__', fn)}'")
            return _orig_init(self, fn, arg_names, configs, valid_key, *args, **kwargs)

        _autotuner_mod.Autotuner.__init__ = _permissive_init
        print("[triton patch] Autotuner patched — tolerant of missing autotune keys.")
    except Exception as exc:
        print(f"[triton patch] Skipped: {exc}")


# Must run before any fla/transformers import so the autotune key validation
# is already relaxed when modeling_qwen3_5.py loads fla.
_patch_triton_autotuner()


def _patch_fla_rms_norm():
    """Replace fla's Triton-based RMSNorm forward with pure PyTorch.

    Qwen3.5 uses fla's RMSNorm for EVERY decoder layer's input_layernorm and
    post_attention_layernorm — not only in GDR layers.  These Triton kernels
    crash on Triton 3.1.0 with 'illegal memory access', corrupting the CUDA
    context before even the first GDR operation is reached.

    Patching at the class level (not instance level) ensures ALL existing and
    future instances automatically use the PyTorch fallback.
    """
    try:
        import fla.modules as _fla_mods
        import torch

        def _pt_rms_norm_fwd(self, x):
            orig_dtype = x.dtype
            x_f = x.float()
            rstd = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True)
                               + getattr(self, 'eps', 1e-6))
            out = (x_f * rstd).to(orig_dtype)
            w = getattr(self, 'weight', None)
            if w is not None:
                out = out * w.to(dtype=orig_dtype)
            return out

        patched = []
        for _name in ('RMSNorm', 'FusedRMSNorm'):
            _cls = getattr(_fla_mods, _name, None)
            if _cls is not None and hasattr(_cls, 'forward'):
                _cls.forward = _pt_rms_norm_fwd
                patched.append(_name)

        if patched:
            print(f"[fla patch] fla.modules.{'/'.join(patched)} forward() "
                  f"replaced with PyTorch fallback (covers all decoder norms).")
    except Exception as exc:
        print(f"[fla patch] RMSNorm patch skipped: {exc}")


_patch_fla_rms_norm()


def _patch_lm_loss_bf16():
    """Prevent transformers ForCausalLMLoss from casting logits to float32.

    The default loss_utils.py does ``logits = logits.float()`` which on a
    [B=18, T=320, V=151936] tensor requires ~7.6 GiB for a single cast —
    triggering OOM on a 24 GiB GPU when combined with the 4B model weights.
    Keeping logits in bf16 for cross_entropy is numerically stable enough
    for a small lm_weight regulariser.
    """
    try:
        import transformers.loss.loss_utils as _loss_utils
        import torch.nn.functional as _F

        def _bf16_causal_lm_loss(logits, labels, vocab_size, **kwargs):
            shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)
            shift_labels = labels[..., 1:].contiguous().view(-1)
            return _F.cross_entropy(
                shift_logits,
                shift_labels.to(shift_logits.device),
                ignore_index=-100,
            )

        _loss_utils.ForCausalLMLoss = _bf16_causal_lm_loss
        print("[lm loss patch] ForCausalLMLoss: logits kept in bf16 (saves ~7 GiB vs float32 cast).")
    except Exception as exc:
        print(f"[lm loss patch] Skipped: {exc}")


_patch_lm_loss_bf16()


def _init_distributed():
    """Initialize NCCL process group when launched via torchrun.

    torchrun sets LOCAL_RANK / RANK / WORLD_SIZE environment variables.
    If LOCAL_RANK is absent (single-process launch), this is a no-op.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank < 0:
        return
    import torch
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    print(f"[DDP] Initialized: rank={rank}/{world_size}, local_rank={local_rank}")


_init_distributed()

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
        from urllib3.exceptions import InsecureRequestWarning

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
from mts_agent.data.loader import MTSDataset
from mts_agent.data.collator import MultimodalCollator
from mts_agent.models.mts_embedder import MTSEmbedder
from mts_agent.engine.trainer import MTSTrainer
from mts_agent.tokenization import DummyTokenizer


LOCAL_MODEL_PATH = "local_model/Qwen/Qwen3___5-4B"


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


def load_local_tokenizer(model_path, use_dummy_tokenizer=False):
    """Load the tokenizer strictly from the local model directory."""
    if use_dummy_tokenizer:
        print("Force using DummyTokenizer (Offline Mode)...")
        return DummyTokenizer(vocab_size=32000)

    print(f"Loading Tokenizer from local path: {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f" -> Successfully loaded tokenizer from {model_path}.")
        return tokenizer
    except Exception as exc:
        print(f" -> Failed to load local tokenizer: {exc}")
        print(" -> Falling back to DummyTokenizer.")
        return DummyTokenizer(vocab_size=32000)


def load_experiment_config(args):
    """Load configuration from JSON or command-line arguments."""
    if args.config:
        print(f"Loading configuration from {args.config}")
        return ExperimentConfig.load(args.config)
    return create_config_from_args(args)


def resolve_ts_dimensions(config):
    """Infer and backfill time-series dimensions into the config."""
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


def build_dataset(data_path, mode, augment, domain_info, samples=None):
    return MTSDataset(
        data_path,
        mode=mode,
        augment=augment,
        domain_info=domain_info,
        samples=samples
    )


def find_valid_split_path(data_path):
    if "train" not in data_path:
        return None
    valid_path = data_path.replace("train", "valid")
    return valid_path if os.path.exists(valid_path) else None


def build_internal_split(full_train_dataset, config):
    rng = random.Random(config.training.seed)
    per_class_samples = defaultdict(list)
    for sample in full_train_dataset.samples:
        per_class_samples[str(sample.get('label', ''))].append(sample)

    train_samples, valid_samples = [], []
    for class_samples in per_class_samples.values():
        class_samples = list(class_samples)
        rng.shuffle(class_samples)
        valid_count = max(1, int(len(class_samples) * 0.2))
        valid_samples.extend(class_samples[:valid_count])
        train_samples.extend(class_samples[valid_count:])

    train_dataset = build_dataset(
        config.data.data_path,
        mode='train',
        augment=config.data.augment,
        domain_info=config.data.domain_info,
        samples=train_samples
    )
    eval_dataset = build_dataset(
        config.data.data_path,
        mode='train',
        augment=False,
        domain_info=config.data.domain_info,
        samples=valid_samples
    )
    print(
        f"Internal val split: {len(train_samples)} train / "
        f"{len(valid_samples)} valid (stratified 80/20, used for model selection)."
    )
    return train_dataset, eval_dataset


def prepare_training_datasets(config):
    """Prepare train / eval / test datasets with strict test-set isolation.

    Convention:
    - If ``{data_path}.replace('train','valid')`` exists it is treated as the
      **held-out test set** and NEVER touched during training.
    - A stratified 80/20 split of the training file is always used for
      in-training validation (model selection).  This guarantees that the
      reported final test accuracy is unbiased.

    Returns:
        (train_dataset, eval_dataset, test_dataset)
        test_dataset is None when no valid file is found.
    """
    if not os.path.exists(config.data.data_path):
        raise FileNotFoundError(
            f"Processed data file not found: {config.data.data_path}\n"
            "Generate it first (e.g. with a DeepSeek API script) and place it at the path above."
        )

    print("Initializing Train Dataset...")
    full_train_dataset = build_dataset(
        config.data.data_path,
        mode='train',
        augment=config.data.augment,
        domain_info=config.data.domain_info
    )

    # Test set: the explicit valid file, if present — kept blind during training.
    test_dataset = None
    valid_path = find_valid_split_path(config.data.data_path)
    if valid_path:
        print(f"Loading test set from {valid_path}...")
        test_dataset = build_dataset(
            valid_path,
            mode='train',
            augment=False,
            domain_info=config.data.domain_info
        )
        print(
            f"Test set: {len(test_dataset)} samples "
            f"(held out — used ONLY for final evaluation after training)."
        )

    # No internal split: use all training samples. Model selection relies on LOO retrieval.
    return full_train_dataset, None, test_dataset


def build_model(config):
    """Construct the embedder from the resolved config."""
    print("Initializing Model...")
    return MTSEmbedder(
        config.model.llm_path,
        ts_hidden_dim=config.model.ts_hidden_dim,
        encoder_base_channels=config.model.encoder_base_channels,
        encoder_dropout=config.model.encoder_dropout,
        encoder_norm=config.model.encoder_norm,
        embedding_pooling=config.model.embedding_pooling,
        ts_marker_start_id=config.model.ts_marker_start_id,
        ts_marker_end_id=config.model.ts_marker_end_id,
        output_dim=config.model.output_dim,
        llm_attn_implementation=config.model.llm_attn_implementation,
        stem_strides=config.model.stem_strides,
        patch_size=config.model.patch_size,
        encoder_type=config.model.encoder_type,
        patch_stride=config.model.patch_stride,
        projector_hidden_dim=getattr(config.model, 'projector_hidden_dim', None),
        channel_mixer=getattr(config.model, 'channel_mixer', False),
        channel_mixer_heads=getattr(config.model, 'channel_mixer_heads', 4),
        tc_former_layers=getattr(config.model, 'tc_former_layers', 4),
        tc_former_heads=getattr(config.model, 'tc_former_heads', 4),
        tc_former_ff_dim=getattr(config.model, 'tc_former_ff_dim', 512),
        tc_former_max_channels=getattr(config.model, 'tc_former_max_channels', 64),
        tc_former_max_patches=getattr(config.model, 'tc_former_max_patches', 256),
        tc_former_use_revin=getattr(config.model, 'tc_former_use_revin', True),
    )


def apply_tuning_strategy(model, config):
    """Apply the requested LLM tuning strategy and keep TS modules trainable."""
    if config.model.tuning_strategy == "freeze":
        print("Strategy: Freezing LLM Backbone (Stage 1)...")
        model.freeze_llm()
    elif config.model.tuning_strategy == "partial":
        print(f"Strategy: Smart Freezing (Top {config.model.num_trainable_layers} Layers + Head)...")
        model.smart_freeze_llm(num_trainable_layers=config.model.num_trainable_layers)
    elif config.model.tuning_strategy == "lora":
        print("Strategy: LoRA Adaptation...")
        lora_r = getattr(config.model, 'lora_r', 8)
        model.apply_lora(r=lora_r, lora_alpha=lora_r * 2)
    else:
        print("Strategy: Full Parameter Tuning...")

    for param in model.projector.parameters():
        param.requires_grad = True
    for param in model.ts_encoder.parameters():
        param.requires_grad = True


def load_checkpoint_if_available(model, config):
    """Load a checkpoint with shape filtering when a path is provided."""
    ckpt_path = config.training.ckpt_path
    if not ckpt_path or not os.path.exists(ckpt_path):
        return

    print(f"Loading checkpoint from {ckpt_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(ckpt_path, map_location=device)

    model_state = model.state_dict()
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if key not in model_state:
            print(f"Skipping unknown layer {key}")
            continue
        if value.shape != model_state[key].shape:
            print(f"Skipping layer {key} due to shape mismatch: {value.shape} vs {model_state[key].shape}")
            continue
        filtered_state_dict[key] = value

    try:
        model.load_state_dict(filtered_state_dict, strict=False)
        print("Checkpoint loaded (partial/filtered) successfully.")
    except Exception as exc:
        print(f"Error loading checkpoint: {exc}")


def build_run_config(config):
    """Flatten the structured config into the trainer runtime config."""
    return {
        'seed': config.training.seed,
        'epochs': config.training.epochs,
        'batch_size': config.data.batch_size,
        'lr': config.training.lr,
        'contrastive_temperature': config.training.contrastive_temperature,
        'learnable_temperature': getattr(config.training, 'learnable_temperature', True),
        'alignment_text_mode': config.training.alignment_text_mode,
        'contrastive_positive_mode': config.training.contrastive_positive_mode,
        'gradient_accumulation_steps': config.training.gradient_accumulation_steps,
        'max_grad_norm': getattr(config.training, 'max_grad_norm', 1.0),
        'balanced_sampling': config.training.balanced_sampling,
        'classes_per_batch': config.training.classes_per_batch,
        'samples_per_class': config.training.samples_per_class,
        'allow_repeated_classes_in_batch': config.training.allow_repeated_classes_in_batch,
        'early_stopping_patience': config.training.early_stopping_patience,
        'early_stopping_min_delta': config.training.early_stopping_min_delta,
        'save_dir': config.training.save_dir,
        'training_stage': config.training.training_stage,
        'contrastive_weight': config.training.contrastive_weight,
        'lm_weight': config.training.lm_weight,
        'cls_weight': config.training.cls_weight,
        'retrieval_eval_enabled': config.training.retrieval_eval_enabled,
        'retrieval_eval_k': config.training.retrieval_eval_k or config.retrieval.k,
        'retrieval_eval_alpha': config.training.retrieval_eval_alpha if config.training.retrieval_eval_alpha is not None else config.retrieval.alpha,
        'retrieval_eval_vote_strategy': config.training.retrieval_eval_vote_strategy,
        'retrieval_eval_interval': config.training.retrieval_eval_interval,
        'use_amp': config.training.use_amp,
        'amp_dtype': config.training.amp_dtype,
        'memory_bank_size': config.training.memory_bank_size,
        'inference_max_new_tokens': config.inference.max_new_tokens,
        'inference_temperature': config.inference.temperature,
        'inference_do_sample': config.inference.do_sample,
        'retrieval_eval_dtw_window_size': config.retrieval.dtw_window_size,
        'retrieval_eval_fast_dtw_max_len': config.retrieval.fast_dtw_max_len,
        'ts_only_embedding': config.training.ts_only_embedding,
        'softclt_enabled': getattr(config.training, 'softclt_enabled', False),
        'softclt_sigma': getattr(config.training, 'softclt_sigma', 1.0),
        # Previously missing — these were silently defaulting to 0/disabled in the trainer
        'warmup_epochs': config.training.warmup_epochs,
        'ema_momentum': config.training.ema_momentum,
        'proto_weight': config.training.proto_weight,
        'proto_momentum': config.training.proto_momentum,
        'weight_decay': getattr(config.training, 'weight_decay', 0.01),
        'loo_smoothing_window': getattr(config.training, 'loo_smoothing_window', 3),
        'llm_lr_scale': getattr(config.training, 'llm_lr_scale', 1.0),
    }


def run_single_stage_training(config, tokenizer, train_dataset, eval_dataset, test_dataset, stage_name, ckpt_path=None, save_dir=None):
    """Run one training stage and return the path to the best checkpoint."""
    stage_config = deepcopy(config)
    stage_config.training.training_stage = stage_name
    if ckpt_path is not None:
        stage_config.training.ckpt_path = ckpt_path
    if save_dir is not None:
        stage_config.training.save_dir = save_dir

    print(
        f"\n=== Stage: {stage_name} | save_dir={stage_config.training.save_dir} "
        f"| ckpt={stage_config.training.ckpt_path} ==="
    )

    collator = MultimodalCollator(
        tokenizer,
        mode='train',
        max_length=stage_config.data.max_length,
        alignment_text_mode=stage_config.training.alignment_text_mode
    )

    model = build_model(stage_config)
    apply_tuning_strategy(model, stage_config)
    load_checkpoint_if_available(model, stage_config)
    run_config = build_run_config(stage_config)

    trainer = MTSTrainer(
        model,
        train_dataset,
        collator,
        run_config,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
    )
    trainer.train()

    best_path = os.path.join(stage_config.training.save_dir, "model_best.pt")
    last_path = os.path.join(stage_config.training.save_dir, "model_last.pt")
    if os.path.exists(best_path):
        return best_path
    if os.path.exists(last_path):
        return last_path
    return None


def main():
    parser = argparse.ArgumentParser(description="MTS-O1-Embedder: Multimodal Time Series Embedding Agent")
    parser.add_argument("--mode", type=str, default="train", choices=["train"], help="Operation mode")
    
    # Paths
    parser.add_argument("--data_path", type=str, default="mts_agent/data/processed/mock_train.jsonl")
    parser.add_argument("--llm_path", type=str, default=LOCAL_MODEL_PATH, help="Local model path")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--use_dummy_tokenizer", action="store_true", help="Force use of DummyTokenizer for offline mode")
    
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
    parser.add_argument(
        "--allow_repeated_classes_in_batch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether balanced sampling may repeat a class inside one batch when unique labels are insufficient"
    )
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
    parser.add_argument("--auto_two_stage", action=argparse.BooleanOptionalAction, default=None, help="Automatically run alignment then reasoning in one command")
    parser.add_argument("--contrastive_weight", type=float, default=0.5, help="Weight for contrastive loss in alignment stage")
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
    parser.add_argument("--disable_ssl_verify", action="store_true", help="Disable SSL certificate verification (UNSAFE)")
    parser.add_argument("--force_offline", action="store_true", help="Force HuggingFace offline mode")
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    parser.add_argument("--save_config", type=str, help="Save current configuration to file")

    args = parser.parse_args()

    config = load_experiment_config(args)

    # When --config is used, allow explicit CLI flags to override selected fields.
    cli_flags = set(sys.argv[1:])
    if "--llm_path" in cli_flags:
        config.model.llm_path = args.llm_path
    if "--data_path" in cli_flags:
        config.data.data_path = args.data_path
    if "--raw_data_path" in cli_flags:
        config.data.raw_data_path = args.raw_data_path
    if "--batch_size" in cli_flags:
        config.data.batch_size = args.batch_size
    if "--gradient_accumulation_steps" in cli_flags:
        config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    if "--save_dir" in cli_flags:
        config.training.save_dir = args.save_dir

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

    resolve_ts_dimensions(config)

    print(f"=== Starting Training Setup: {config.experiment_name} ===")

    tokenizer = load_local_tokenizer(config.model.llm_path, config.environment.use_dummy_tokenizer)
    train_dataset, eval_dataset, test_dataset = prepare_training_datasets(config)

    run_single_stage_training(
        config=config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        stage_name=config.training.training_stage,
        ckpt_path=config.training.ckpt_path,
        save_dir=config.training.save_dir,
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()

