"""
Configuration management for MTS-O1-Embedder.
Centralizes all configurable parameters.
"""
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple


ModeType = Literal["gen_data", "train"]
EncoderNormType = Literal["group", "batch"]
EmbeddingPoolingType = Literal["mean", "last", "ts_tokens", "latent"]
TuningStrategyType = Literal["full", "freeze", "partial", "lora"]
TrainingStageType = Literal["alignment", "reasoning"]
AlignmentTextModeType = Literal["full", "context", "label", "context_label"]
ContrastivePositiveModeType = Literal["diagonal", "label"]
RetrievalVoteStrategyType = Literal[
    "majority", "weighted", "rank", "semantic", "structural", "prototype", "hybrid_prototype"
]
DeviceType = Literal["cuda", "cpu"]
AmpDTypeType = Literal["bf16", "fp16"]
AttnImplementationType = Literal["eager", "sdpa", "flash_attention_2"]


def _filter_dataclass_kwargs(dataclass_type: type, kwargs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only keys that are valid dataclass fields."""
    valid_keys = {item.name for item in fields(dataclass_type)}
    return {key: value for key, value in kwargs_dict.items() if key in valid_keys}


def _build_nested_config(dataclass_type: type, payload: Dict[str, Any]) -> Any:
    return dataclass_type(**_filter_dataclass_kwargs(dataclass_type, payload))


def _set_nested_attr(config: "ExperimentConfig", path: Sequence[str], value: Any) -> None:
    target = config
    for part in path[:-1]:
        target = getattr(target, part)
    setattr(target, path[-1], value)


def _assign_if_present(
    config: "ExperimentConfig",
    args: Any,
    arg_name: str,
    target_path: Sequence[str],
    *,
    skip_none: bool = False,
) -> None:
    if not hasattr(args, arg_name):
        return
    value = getattr(args, arg_name)
    if skip_none and value is None:
        return
    _set_nested_attr(config, target_path, value)

@dataclass
class DataConfig:
    """Data loading and processing configuration."""
    data_path: str = "mts_agent/data/processed/mock_train.jsonl"
    raw_data_path: str = "dummy.npy"
    dataset_name: Optional[str] = None
    batch_size: int = 2
    ts_dim: Optional[int] = None
    augment: bool = True
    max_length: int = 1024
    domain_info: Optional[str] = None

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    llm_path: str = "local_model/Qwen/Qwen3___5-4B"
    ts_input_dim: Optional[int] = None
    ts_hidden_dim: int = 128
    output_dim: int = 1024
    encoder_base_channels: int = 64
    encoder_target_tokens: Optional[int] = None  # Legacy field kept for old JSON configs.
    encoder_dropout: float = 0.1
    encoder_norm: EncoderNormType = "group"
    embedding_pooling: EmbeddingPoolingType = "mean"
    llm_attn_implementation: AttnImplementationType = "eager"
    tuning_strategy: TuningStrategyType = "lora"
    lora_r: int = 8               # LoRA rank; increase for more capacity (e.g. 16 for universal pre-training)
    num_trainable_layers: Optional[int] = None
    stem_strides: List[int] = field(default_factory=lambda: [5, 5])
    patch_size: int = 5
    # TS encoder type: "cnn" (CNN ResNet, original) or "patch" (PatchTST-style linear tokenizer)
    encoder_type: str = "cnn"
    patch_stride: Optional[int] = None   # PatchTokenizer stride; None → equals patch_size (non-overlapping)
    # P5: Projector hidden dim override; None → use expansion_ratio=2.0 (default 7168 for llm_dim=3584)
    projector_hidden_dim: Optional[int] = None
    # P6: Cross-channel mixer; lightweight attention over C channels at each patch position
    channel_mixer: bool = False
    channel_mixer_heads: int = 4
    # TC-Former encoder settings (encoder_type="tc_former")
    tc_former_layers: int = 4          # number of TC-Former layers (Small=4, Base=6, Large=8)
    tc_former_heads: int = 4           # attention heads (must divide ts_hidden_dim)
    tc_former_ff_dim: int = 512        # FFN hidden dim (typically 4×ts_hidden_dim)
    tc_former_max_channels: int = 64   # max channels for learnable channel embedding
    tc_former_max_patches: int = 256   # max patches per channel for position embedding
    tc_former_use_revin: bool = True   # use RevIN normalization (recommended)
    # Phase 2: Bidirectional attention (NV-Embed style) — safe only with standard MHA models
    # (NOT for Qwen3.5 GDR hybrid layers which require causal/recurrent attention).
    use_bidirectional_attn: bool = False
    # Phase 2: Latent Attention Pooling (NV-Embed style) — replaces ts_tokens mean pooling
    # when embedding_pooling="latent". K learnable queries cross-attend all hidden states.
    latent_pooling_num_latents: int = 8   # number of learnable latent queries
    latent_pooling_heads: int = 8         # attention heads in cross-attention
    # Token IDs for TS-segment markers; Qwen2/3 use 151652/151653 by default.
    # For other model families, set these to the IDs of your chosen delimiter tokens.
    ts_marker_start_id: int = 151652
    ts_marker_end_id: int = 151653

@dataclass
class TrainingConfig:
    """Training configuration."""
    seed: int = 42
    epochs: int = 3
    lr: float = 1e-4
    weight_decay: float = 0.01      # AdamW weight decay; 0.0 = disabled
    warmup_epochs: int = 0
    contrastive_temperature: float = 0.07
    learnable_temperature: bool = True  # Jointly optimise log(τ) — eliminates manual temperature tuning
    softclt_enabled: bool = False     # SoftCLT (ICLR 2024): DTW-based soft positive weights
    softclt_sigma: float = 1.0        # Temperature for exp(-dist/sigma) soft weight kernel
    loo_smoothing_window: int = 3     # Rolling-mean window for LOO model selection; 1 = no smoothing
    alignment_text_mode: AlignmentTextModeType = "full"
    contrastive_positive_mode: ContrastivePositiveModeType = "diagonal"
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0       # Gradient clipping max norm; 0.0 = disabled
    balanced_sampling: bool = False
    classes_per_batch: Optional[int] = None
    samples_per_class: Optional[int] = None
    allow_repeated_classes_in_batch: bool = True
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 0.0
    training_stage: TrainingStageType = "alignment"
    contrastive_weight: float = 0.5
    lm_weight: float = 1.0
    cls_weight: float = 0.0          # Weight for direct CE classification loss on the embedding
    ts_only_embedding: bool = False   # When True, bypass LLM; embed with ts_encoder+projector only
    retrieval_eval_enabled: bool = True
    retrieval_eval_k: Optional[int] = None
    retrieval_eval_alpha: Optional[float] = None
    retrieval_eval_vote_strategy: RetrievalVoteStrategyType = "weighted"
    retrieval_eval_interval: int = 1
    use_amp: bool = True
    amp_dtype: AmpDTypeType = "bf16"
    save_dir: str = "checkpoints"
    ckpt_path: Optional[str] = None
    memory_bank_size: int = 64
    auto_two_stage: bool = True
    ema_momentum: float = 0.0  # EMA gallery encoder momentum; 0.0 = disabled; 0.995 recommended
    proto_weight: float = 0.0  # Weight for prototype-NCE loss; 0.0 = disabled
    proto_momentum: float = 0.9  # EMA momentum for class prototype update
    llm_lr_scale: float = 1.0   # LR multiplier for LLM LoRA params; < 1.0 = layer-wise LR decay

@dataclass
class RetrievalConfig:
    """Retrieval system configuration."""
    k: int = 5
    alpha: float = 0.8  # weight for semantic similarity
    dtw_window_size: Optional[int] = None
    fast_dtw_max_len: int = 100

@dataclass
class InferenceConfig:
    """Inference configuration."""
    max_new_tokens: int = 200
    temperature: float = 0.7
    do_sample: bool = True
    use_api: bool = False
    api_key: Optional[str] = None

@dataclass
class EnvironmentConfig:
    """Environment and system configuration."""
    disable_ssl_verify: bool = False
    force_offline: bool = False
    use_dummy_tokenizer: bool = False
    device: DeviceType = "cuda"

@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    mode: ModeType = "train"  # Supported entry modes in main.py: "train".
    experiment_name: str = "default_experiment"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        data_config = _build_nested_config(DataConfig, config_dict.get('data', {}))

        model_values = dict(config_dict.get('model', {}))
        if 'ts_input_dim' not in model_values:
            model_values['ts_input_dim'] = config_dict.get('data', {}).get('ts_dim', None)
        model_config = _build_nested_config(ModelConfig, model_values)

        training_config = _build_nested_config(TrainingConfig, config_dict.get('training', {}))
        retrieval_config = _build_nested_config(RetrievalConfig, config_dict.get('retrieval', {}))
        inference_config = _build_nested_config(InferenceConfig, config_dict.get('inference', {}))
        environment_config = _build_nested_config(EnvironmentConfig, config_dict.get('environment', {}))

        # Extract top-level configs
        mode = config_dict.get('mode', 'train')
        experiment_name = config_dict.get('experiment_name', 'default_experiment')

        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            retrieval=retrieval_config,
            inference=inference_config,
            environment=environment_config,
            mode=mode,
            experiment_name=experiment_name
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, filepath: str):
        """Save config to JSON file."""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Config saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """Load config from JSON file."""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

# Default configuration
DEFAULT_CONFIG = ExperimentConfig()


ARG_TO_CONFIG_PATHS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = (
    ("data_path", ("data", "data_path"), False),
    ("train_path", ("data", "data_path"), False),
    ("raw_data_path", ("data", "raw_data_path"), False),
    ("dataset_name", ("data", "dataset_name"), False),
    ("batch_size", ("data", "batch_size"), False),
    ("ts_base_channels", ("model", "encoder_base_channels"), False),
    ("ts_tokens", ("model", "encoder_target_tokens"), False),
    ("ts_dropout", ("model", "encoder_dropout"), False),
    ("ts_norm", ("model", "encoder_norm"), False),
    ("embedding_pooling", ("model", "embedding_pooling"), False),
    ("llm_attn_implementation", ("model", "llm_attn_implementation"), False),
    ("llm_path", ("model", "llm_path"), False),
    ("tuning_strategy", ("model", "tuning_strategy"), False),
    ("epochs", ("training", "epochs"), False),
    ("seed", ("training", "seed"), False),
    ("lr", ("training", "lr"), False),
    ("warmup_epochs", ("training", "warmup_epochs"), False),
    ("contrastive_temperature", ("training", "contrastive_temperature"), False),
    ("alignment_text_mode", ("training", "alignment_text_mode"), False),
    ("contrastive_positive_mode", ("training", "contrastive_positive_mode"), False),
    ("gradient_accumulation_steps", ("training", "gradient_accumulation_steps"), False),
    ("balanced_sampling", ("training", "balanced_sampling"), False),
    ("classes_per_batch", ("training", "classes_per_batch"), False),
    ("samples_per_class", ("training", "samples_per_class"), False),
    ("allow_repeated_classes_in_batch", ("training", "allow_repeated_classes_in_batch"), True),
    ("early_stopping_patience", ("training", "early_stopping_patience"), False),
    ("early_stopping_min_delta", ("training", "early_stopping_min_delta"), False),
    ("training_stage", ("training", "training_stage"), False),
    ("auto_two_stage", ("training", "auto_two_stage"), True),
    ("ema_momentum", ("training", "ema_momentum"), False),
    ("proto_weight", ("training", "proto_weight"), False),
    ("proto_momentum", ("training", "proto_momentum"), False),
    ("contrastive_weight", ("training", "contrastive_weight"), False),
    ("lm_weight", ("training", "lm_weight"), False),
    ("retrieval_eval_enabled", ("training", "retrieval_eval_enabled"), True),
    ("retrieval_eval_k", ("training", "retrieval_eval_k"), False),
    ("retrieval_eval_alpha", ("training", "retrieval_eval_alpha"), False),
    ("retrieval_eval_vote_strategy", ("training", "retrieval_eval_vote_strategy"), False),
    ("dtw_window_size", ("retrieval", "dtw_window_size"), False),
    ("fast_dtw_max_len", ("retrieval", "fast_dtw_max_len"), False),
    ("save_dir", ("training", "save_dir"), False),
    ("ckpt_path", ("training", "ckpt_path"), False),
    ("disable_ssl_verify", ("environment", "disable_ssl_verify"), False),
    ("force_offline", ("environment", "force_offline"), False),
    ("use_dummy_tokenizer", ("environment", "use_dummy_tokenizer"), False),
    ("use_api", ("inference", "use_api"), False),
    ("api_key", ("inference", "api_key"), False),
    ("domain_info", ("data", "domain_info"), False),
    ("mode", ("mode",), False),
)

def create_config_from_args(args: Any) -> ExperimentConfig:
    """Create config from command line arguments."""
    config = ExperimentConfig()

    for arg_name, target_path, skip_none in ARG_TO_CONFIG_PATHS:
        _assign_if_present(config, args, arg_name, target_path, skip_none=skip_none)

    if hasattr(args, 'ts_dim') and args.ts_dim is not None:
        config.data.ts_dim = args.ts_dim
        config.model.ts_input_dim = args.ts_dim

    return config