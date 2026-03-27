import json
import torch
from torch.utils.data import Dataset, DataLoader
from .augmentations import TimeSeriesAugmentor


def ensure_channel_first_tensor(ts):
    if not torch.is_tensor(ts):
        ts = torch.as_tensor(ts, dtype=torch.float32)
    if ts.ndim == 1:
        return ts.unsqueeze(0)
    if ts.ndim == 2 and ts.shape[0] > ts.shape[1]:
        return ts.transpose(0, 1)
    if ts.ndim != 2:
        raise ValueError(f"time_series must be 1D or 2D, got shape={tuple(ts.shape)}")
    return ts


class MTSDataset(Dataset):
    def __init__(self, data_path, mode='train', augment=False, domain_info=None, samples=None):
        """If `samples` is provided (a list of dicts), data_path is not read from disk.
        This enables in-memory sub-splits without creating new files."""
        self.data_path = data_path
        self.mode = mode
        self.augment = augment
        self.augmentor = TimeSeriesAugmentor(prob=0.5) if augment else None
        # Safe augmentor for view2: no window slicing (preserves temporal pattern / peak position)
        self.safe_augmentor = TimeSeriesAugmentor(
            prob=0.5, sigma=0.03, scale_sigma=0.1, channel_dropout_prob=0.1,
            window_slice_min_ratio=1.0,  # ratio=1.0 disables window slicing
        ) if augment else None
        self.domain_info = domain_info  # Injected Domain Knowledge
        self.samples = []

        if samples is not None:
            # Pre-loaded samples passed directly (e.g. internal train/val split)
            self.samples = list(samples)
            print(f"Dataset created from {len(self.samples)} pre-loaded samples (mode={mode}, augment={augment}).")
            return

        print(f"Loading dataset from {data_path} (mode={mode}, augment={augment})...")
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, start=1):
                    if line.strip():
                        try:
                            self.samples.append(json.loads(line))
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"Invalid JSON on line {line_no} in {data_path}") from exc
            print(f"Loaded {len(self.samples)} samples.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {data_path}")

    def _build_context(self, item):
        base_context = str(item.get('context', ''))
        if self.domain_info and self.domain_info not in base_context:
            return f"Domain Knowledge: {self.domain_info}\n{base_context}".strip()
        return base_context

    def _parse_time_series(self, item):
        if 'time_series' not in item:
            item_id = item.get('id', '<missing-id>')
            raise KeyError(f"Sample {item_id} is missing required field: time_series")
        ts = ensure_channel_first_tensor(torch.as_tensor(item['time_series'], dtype=torch.float32))
        return ts

    def _normalize_item(self, item, idx):
        sample_id = item.get('id', f'sample_{idx}')
        teacher_thought = str(item.get('teacher_thought') or item.get('thought') or "")
        return {
            "id": sample_id,
            "dataset": str(item.get('dataset', '')),
            "time_series": self._parse_time_series(item),
            "context": self._build_context(item),
            "teacher_thought": teacher_thought,
            "thought": teacher_thought,
            "label": item.get('label', ""),
        }

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self._normalize_item(self.samples[idx], idx)
        ts = item['time_series']

        # Apply augmentation if enabled
        if self.augment and self.mode == 'train' and self.augmentor:
            ts_view1 = self.augmentor(ts)       # heavy aug (view1: query)
            ts_view2 = self.safe_augmentor(ts)  # safe aug  (view2: gallery, no window slicing)
        else:
            ts_view1 = ts
            ts_view2 = ts

        return {
            "id": item['id'],
            "dataset": item['dataset'],
            "time_series": ts_view1,
            "time_series_view2": ts_view2,
            "context": item['context'],
            "teacher_thought": item['teacher_thought'],
            "thought": item['thought'],
            "label": item['label']
        }

def create_loader(
    data_path,
    batch_size=32,
    mode='train',
    augment=False,
    domain_info=None,
    samples=None,
    shuffle=None,
    num_workers=0,
    collate_fn=None,
):
    dataset = MTSDataset(
        data_path,
        mode=mode,
        augment=augment,
        domain_info=domain_info,
        samples=samples,
    )
    effective_shuffle = (mode == 'train') if shuffle is None else shuffle
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=effective_shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

