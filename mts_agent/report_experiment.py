import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def pick_best_epoch(history: List[Dict[str, Any]], metric_key: str, mode: str = 'max') -> Optional[Dict[str, Any]]:
    if not history:
        return None
    candidates = [item for item in history if metric_key in item and item[metric_key] is not None]
    if not candidates:
        return None
    reverse = mode == 'max'
    return sorted(candidates, key=lambda item: item[metric_key], reverse=reverse)[0]


def build_summary(run_dir: Path) -> Dict[str, Any]:
    run_config = load_json(run_dir / 'run_config.json') or {}
    history = load_json(run_dir / 'training_history.json') or []
    best_meta = load_json(run_dir / 'model_best.pt.meta.json') or {}
    last_meta = load_json(run_dir / 'model_last.pt.meta.json') or {}

    final_epoch = history[-1] if history else {}
    best_val_epoch = pick_best_epoch(history, 'val_loss', mode='min')
    best_retrieval_epoch = pick_best_epoch(history, 'retrieval_accuracy', mode='max')
    best_macro_f1_epoch = pick_best_epoch(history, 'retrieval_macro_f1', mode='max')

    retrieval_cfg = {
        'enabled': run_config.get('retrieval_eval_enabled', False),
        'k': run_config.get('retrieval_eval_k'),
        'alpha': run_config.get('retrieval_eval_alpha'),
        'vote_strategy': run_config.get('retrieval_eval_vote_strategy'),
        'dtw_window_size': run_config.get('retrieval_eval_dtw_window_size'),
        'fast_dtw_max_len': run_config.get('retrieval_eval_fast_dtw_max_len'),
    }

    model_cfg = {
        'training_stage': run_config.get('training_stage'),
        'learning_rate': run_config.get('lr'),
        'epochs': run_config.get('epochs'),
        'batch_size': run_config.get('batch_size'),
        'embedding_pooling': run_config.get('embedding_pooling'),
        'contrastive_weight': run_config.get('contrastive_weight'),
        'hard_negative_weight': run_config.get('hard_negative_weight'),
        'neighbor_weight': run_config.get('neighbor_weight'),
        'lm_weight': run_config.get('lm_weight'),
    }

    summary = {
        'run_dir': str(run_dir),
        'files_present': {
            'run_config': (run_dir / 'run_config.json').exists(),
            'training_history': (run_dir / 'training_history.json').exists(),
            'best_meta': (run_dir / 'model_best.pt.meta.json').exists(),
            'last_meta': (run_dir / 'model_last.pt.meta.json').exists(),
        },
        'model_config': model_cfg,
        'retrieval_config': retrieval_cfg,
        'history_length': len(history),
        'final_epoch': final_epoch,
        'best_val_epoch': best_val_epoch,
        'best_retrieval_epoch': best_retrieval_epoch,
        'best_macro_f1_epoch': best_macro_f1_epoch,
        'best_checkpoint': best_meta,
        'last_checkpoint': last_meta,
    }
    return summary


def format_markdown(summary: Dict[str, Any]) -> str:
    model_cfg = summary.get('model_config', {})
    retrieval_cfg = summary.get('retrieval_config', {})
    final_epoch = summary.get('final_epoch') or {}
    best_val_epoch = summary.get('best_val_epoch') or {}
    best_retrieval_epoch = summary.get('best_retrieval_epoch') or {}
    best_macro_f1_epoch = summary.get('best_macro_f1_epoch') or {}
    best_checkpoint = summary.get('best_checkpoint') or {}

    lines = [
        '# Experiment Report',
        '',
        f"- Run directory: {summary.get('run_dir', '')}",
        f"- Training stage: {model_cfg.get('training_stage')}",
        f"- Epochs configured: {model_cfg.get('epochs')}",
        f"- Batch size: {model_cfg.get('batch_size')}",
        f"- Learning rate: {model_cfg.get('learning_rate')}",
        f"- Embedding pooling: {model_cfg.get('embedding_pooling')}",
        '',
        '## Retrieval Configuration',
        '',
        f"- Enabled: {retrieval_cfg.get('enabled')}",
        f"- k: {retrieval_cfg.get('k')}",
        f"- alpha: {retrieval_cfg.get('alpha')}",
        f"- vote strategy: {retrieval_cfg.get('vote_strategy')}",
        f"- dtw window size: {retrieval_cfg.get('dtw_window_size')}",
        f"- fast dtw max len: {retrieval_cfg.get('fast_dtw_max_len')}",
        '',
        '## Final Epoch',
        '',
        f"- epoch: {final_epoch.get('epoch')}",
        f"- train_loss: {final_epoch.get('train_loss')}",
        f"- val_loss: {final_epoch.get('val_loss')}",
        f"- retrieval_accuracy: {final_epoch.get('retrieval_accuracy')}",
        f"- retrieval_macro_f1: {final_epoch.get('retrieval_macro_f1')}",
        '',
        '## Best Epochs',
        '',
        f"- best val loss epoch: {best_val_epoch.get('epoch')} (val_loss={best_val_epoch.get('val_loss')})",
        f"- best retrieval accuracy epoch: {best_retrieval_epoch.get('epoch')} (retrieval_accuracy={best_retrieval_epoch.get('retrieval_accuracy')})",
        f"- best retrieval macro-f1 epoch: {best_macro_f1_epoch.get('epoch')} (retrieval_macro_f1={best_macro_f1_epoch.get('retrieval_macro_f1')})",
        '',
        '## Best Checkpoint Metadata',
        '',
        f"- checkpoint: {best_checkpoint.get('checkpoint')}",
        f"- epoch: {best_checkpoint.get('epoch')}",
        f"- is_best: {best_checkpoint.get('is_best')}",
        f"- best_val_loss: {best_checkpoint.get('best_val_loss')}",
        f"- best_retrieval_accuracy: {best_checkpoint.get('best_retrieval_accuracy')}",
    ]
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize a training run into JSON or Markdown.')
    parser.add_argument('--run_dir', type=str, required=True, help='Checkpoint directory containing run_config.json and training_history.json')
    parser.add_argument('--output', type=str, default=None, help='Optional output path. Defaults to experiment_report.{json|md} inside run_dir')
    parser.add_argument('--format', type=str, default='md', choices=['md', 'json'], help='Output format')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    summary = build_summary(run_dir)

    if args.format == 'json':
        payload = json.dumps(summary, indent=2, ensure_ascii=False)
        default_output = run_dir / 'experiment_report.json'
    else:
        payload = format_markdown(summary)
        default_output = run_dir / 'experiment_report.md'

    output_path = Path(args.output) if args.output else default_output
    output_path.write_text(payload, encoding='utf-8')
    print(f'Report written to: {output_path}')


if __name__ == '__main__':
    main()
