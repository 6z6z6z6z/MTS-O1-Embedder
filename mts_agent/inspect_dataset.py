"""
Inspect a raw dataset or processed JSONL file and print basic shape metadata.
"""
import argparse
import json
import os
from collections import Counter

from mts_agent.data.adapters import infer_dataset_name, infer_ts_input_dim, load_dataset_by_adapter, ensure_channel_first


def inspect_processed_jsonl(data_path: str):
    sample_count = 0
    labels = Counter()
    first_shape = None

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            sample_count += 1
            labels[str(item.get('label', 'Unknown'))] += 1
            if first_shape is None and 'time_series' in item:
                first_shape = tuple(ensure_channel_first(item['time_series']).shape)

    print(f"dataset_type=processed_jsonl")
    print(f"samples={sample_count}")
    print(f"first_sample_shape={first_shape}")
    print(f"ts_dim={infer_ts_input_dim(data_path=data_path)}")
    print(f"label_distribution={dict(labels)}")


def inspect_raw_dataset(data_path: str, dataset_name: str | None):
    inferred_name = infer_dataset_name(data_path, dataset_name)
    data, labels = load_dataset_by_adapter(data_path, mode='train', dataset_name=inferred_name)

    sample_count = len(data)
    first_shape = tuple(ensure_channel_first(data[0]).shape) if sample_count else None
    label_counter = Counter(map(str, labels[: min(len(labels), 1000)])) if labels is not None else {}

    print(f"dataset_type=raw")
    print(f"dataset_name={inferred_name}")
    print(f"samples={sample_count}")
    print(f"first_sample_shape={first_shape}")
    print(f"ts_dim={infer_ts_input_dim(raw_data_path=data_path, dataset_name=inferred_name)}")
    print(f"label_distribution_preview={dict(label_counter)}")


def main():
    parser = argparse.ArgumentParser(description="Inspect time-series dataset metadata")
    parser.add_argument("--data_path", required=True, help="Path to raw dataset folder/file or processed JSONL")
    parser.add_argument("--dataset_name", default=None, help="Optional dataset name override")
    args = parser.parse_args()

    if args.data_path.endswith('.jsonl') and os.path.isfile(args.data_path):
        inspect_processed_jsonl(args.data_path)
    else:
        inspect_raw_dataset(args.data_path, args.dataset_name)


if __name__ == "__main__":
    main()