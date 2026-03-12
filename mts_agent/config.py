"""
Configuration management for MTS-O1-Embedder.
Centralizes all configurable parameters.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

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
    llm_path: str = "Qwen/Qwen2.5-0.5B"
    ts_input_dim: Optional[int] = None
    ts_hidden_dim: int = 128
    encoder_base_channels: int = 64
    encoder_target_tokens: int = 16
    encoder_dropout: float = 0.1
    encoder_norm: str = "group"
    embedding_pooling: str = "mean"  # "mean", "last", or "ts_tokens"
    tuning_strategy: str = "lora"  # "full", "freeze", "partial", "lora"
    num_trainable_layers: int = 2

@dataclass
class TrainingConfig:
    """Training configuration."""
    seed: int = 42
    epochs: int = 3
    lr: float = 1e-4
    contrastive_temperature: float = 0.07
    alignment_text_mode: str = "full"
    contrastive_positive_mode: str = "diagonal"
    gradient_accumulation_steps: int = 1
    balanced_sampling: bool = False
    classes_per_batch: Optional[int] = None
    samples_per_class: Optional[int] = None
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 0.0
    training_stage: str = "alignment"  # "alignment" or "reasoning"
    contrastive_weight: float = 0.5
    hard_negative_weight: float = 0.0
    hard_negative_margin: float = 0.05
    neighbor_weight: float = 0.0
    neighbor_margin: float = 0.1
    lm_weight: float = 1.0
    cls_weight: float = 0.0          # Weight for direct CE classification loss on the embedding
    ts_only_embedding: bool = False   # When True, bypass LLM; embed with ts_encoder+projector only
    retrieval_eval_enabled: bool = True
    retrieval_eval_k: Optional[int] = None
    retrieval_eval_alpha: Optional[float] = None
    retrieval_eval_vote_strategy: str = "weighted"
    retrieval_eval_interval: int = 1
    use_amp: bool = True
    save_dir: str = "checkpoints"
    ckpt_path: Optional[str] = None
    memory_bank_size: int = 64

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
    device: str = "cuda"  # "cuda" or "cpu"

@dataclass
class ARMConfig:
    """Adaptive Retrieval Mixer configuration (TS-RAG-inspired post-training)."""
    enabled: bool = False
    k: int = 5                     # neighbors used by ARM at train & test time
    epochs: int = 100              # ARM training epochs (backbone frozen)
    lr: float = 3e-4               # AdamW learning rate for ARM
    num_heads: int = 8             # multi-head attention heads
    dropout: float = 0.1           # dropout inside ARM
    augment_gallery: bool = False  # enrich gallery with augmented embeddings
    augment_factor: int = 5        # augmentations per sample when enabled

@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    arm: ARMConfig = field(default_factory=ARMConfig)
    mode: str = "train"  # "gen_data", "train", "inference", "evaluate"
    experiment_name: str = "default_experiment"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        # Extract nested configs
        data_config = DataConfig(**config_dict.get('data', {}))

        model_values = dict(config_dict.get('model', {}))
        if 'ts_input_dim' not in model_values:
            model_values['ts_input_dim'] = config_dict.get('data', {}).get('ts_dim', None)
        model_config = ModelConfig(**model_values)
        training_config = TrainingConfig(**config_dict.get('training', {}))
        retrieval_config = RetrievalConfig(**config_dict.get('retrieval', {}))
        inference_config = InferenceConfig(**config_dict.get('inference', {}))
        environment_config = EnvironmentConfig(**config_dict.get('environment', {}))
        arm_config = ARMConfig(**config_dict.get('arm', {}))

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
            arm=arm_config,
            mode=mode,
            experiment_name=experiment_name
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'retrieval': self.retrieval.__dict__,
            'inference': self.inference.__dict__,
            'environment': self.environment.__dict__,
            'arm': self.arm.__dict__,
            'mode': self.mode,
            'experiment_name': self.experiment_name
        }

    def save(self, filepath: str):
        """Save config to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
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

def create_config_from_args(args):
    """Create config from command line arguments."""
    config = ExperimentConfig()

    # Update from args
    if hasattr(args, 'data_path'):
        config.data.data_path = args.data_path
    if hasattr(args, 'train_path'):  # For retrieval/evaluation-style entry points
        config.data.data_path = args.train_path
    if hasattr(args, 'test_path'):  # Reserved for evaluation-style entry points
        # Store test path separately if needed
        pass
    if hasattr(args, 'raw_data_path'):
        config.data.raw_data_path = args.raw_data_path
    if hasattr(args, 'dataset_name'):
        config.data.dataset_name = args.dataset_name
    if hasattr(args, 'batch_size'):
        config.data.batch_size = args.batch_size
    if hasattr(args, 'ts_dim') and args.ts_dim is not None:
        config.data.ts_dim = args.ts_dim
        config.model.ts_input_dim = args.ts_dim
    if hasattr(args, 'ts_base_channels'):
        config.model.encoder_base_channels = args.ts_base_channels
    if hasattr(args, 'ts_tokens'):
        config.model.encoder_target_tokens = args.ts_tokens
    if hasattr(args, 'ts_dropout'):
        config.model.encoder_dropout = args.ts_dropout
    if hasattr(args, 'ts_norm'):
        config.model.encoder_norm = args.ts_norm
    if hasattr(args, 'embedding_pooling'):
        config.model.embedding_pooling = args.embedding_pooling
    if hasattr(args, 'llm_path'):
        config.model.llm_path = args.llm_path
    if hasattr(args, 'tuning_strategy'):
        config.model.tuning_strategy = args.tuning_strategy
    if hasattr(args, 'epochs'):
        config.training.epochs = args.epochs
    if hasattr(args, 'seed'):
        config.training.seed = args.seed
    if hasattr(args, 'lr'):
        config.training.lr = args.lr
    if hasattr(args, 'contrastive_temperature'):
        config.training.contrastive_temperature = args.contrastive_temperature
    if hasattr(args, 'alignment_text_mode'):
        config.training.alignment_text_mode = args.alignment_text_mode
    if hasattr(args, 'contrastive_positive_mode'):
        config.training.contrastive_positive_mode = args.contrastive_positive_mode
    if hasattr(args, 'gradient_accumulation_steps'):
        config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    if hasattr(args, 'balanced_sampling'):
        config.training.balanced_sampling = args.balanced_sampling
    if hasattr(args, 'classes_per_batch'):
        config.training.classes_per_batch = args.classes_per_batch
    if hasattr(args, 'samples_per_class'):
        config.training.samples_per_class = args.samples_per_class
    if hasattr(args, 'early_stopping_patience'):
        config.training.early_stopping_patience = args.early_stopping_patience
    if hasattr(args, 'early_stopping_min_delta'):
        config.training.early_stopping_min_delta = args.early_stopping_min_delta
    if hasattr(args, 'training_stage'):
        config.training.training_stage = args.training_stage
    if hasattr(args, 'contrastive_weight'):
        config.training.contrastive_weight = args.contrastive_weight
    if hasattr(args, 'hard_negative_weight'):
        config.training.hard_negative_weight = args.hard_negative_weight
    if hasattr(args, 'hard_negative_margin'):
        config.training.hard_negative_margin = args.hard_negative_margin
    if hasattr(args, 'neighbor_weight'):
        config.training.neighbor_weight = args.neighbor_weight
    if hasattr(args, 'neighbor_margin'):
        config.training.neighbor_margin = args.neighbor_margin
    if hasattr(args, 'lm_weight'):
        config.training.lm_weight = args.lm_weight
    if hasattr(args, 'retrieval_eval_enabled') and args.retrieval_eval_enabled is not None:
        config.training.retrieval_eval_enabled = args.retrieval_eval_enabled
    if hasattr(args, 'retrieval_eval_k'):
        config.training.retrieval_eval_k = args.retrieval_eval_k
    if hasattr(args, 'retrieval_eval_alpha'):
        config.training.retrieval_eval_alpha = args.retrieval_eval_alpha
    if hasattr(args, 'retrieval_eval_vote_strategy'):
        config.training.retrieval_eval_vote_strategy = args.retrieval_eval_vote_strategy
    if hasattr(args, 'dtw_window_size'):
        config.retrieval.dtw_window_size = args.dtw_window_size
    if hasattr(args, 'fast_dtw_max_len'):
        config.retrieval.fast_dtw_max_len = args.fast_dtw_max_len
    if hasattr(args, 'save_dir'):
        config.training.save_dir = args.save_dir
    if hasattr(args, 'ckpt_path'):
        config.training.ckpt_path = args.ckpt_path
    if hasattr(args, 'disable_ssl_verify'):
        config.environment.disable_ssl_verify = args.disable_ssl_verify
    if hasattr(args, 'force_offline'):
        config.environment.force_offline = args.force_offline
    if hasattr(args, 'use_dummy_tokenizer'):
        config.environment.use_dummy_tokenizer = args.use_dummy_tokenizer
    if hasattr(args, 'use_api'):
        config.inference.use_api = args.use_api
    if hasattr(args, 'api_key'):
        config.inference.api_key = args.api_key
    if hasattr(args, 'domain_info'):
        config.data.domain_info = args.domain_info
    if hasattr(args, 'mode'):
        config.mode = args.mode
    else:
        # Default mode when used by evaluation-style entry points
        config.mode = "evaluate"

    return config