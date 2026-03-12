import json
import torch
from torch.utils.data import Dataset, DataLoader
from .augmentations import TimeSeriesAugmentor


def ensure_channel_first_tensor(ts):
    if ts.ndim == 1:
        return ts.unsqueeze(0)
    if ts.ndim == 2 and ts.shape[0] > ts.shape[1]:
        return ts.transpose(0, 1)
    return ts

class MTSDataset(Dataset):
    def __init__(self, data_path, mode='train', augment=False, domain_info=None, samples=None):
        """If `samples` is provided (a list of dicts), data_path is not read from disk.
        This enables in-memory sub-splits without creating new files."""
        self.data_path = data_path
        self.mode = mode
        self.augment = augment
        self.augmentor = TimeSeriesAugmentor(prob=0.5) if augment else None
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
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line))
            print(f"Loaded {len(self.samples)} samples.")
        except FileNotFoundError:
            print(f"Error: Data file {data_path} not found.")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # Format:
        # time_series: list or numpy array
        # context: str
        # teacher_thought: str
        
        ts = torch.tensor(item['time_series'], dtype=torch.float32)
        ts = ensure_channel_first_tensor(ts)
        
        base_context = str(item.get('context', ''))
        if self.domain_info and self.domain_info not in base_context:
            enhanced_context = f"Domain Knowledge: {self.domain_info}\n{base_context}".strip()
        else:
            enhanced_context = base_context

        teacher_thought = str(item.get('teacher_thought') or item.get('thought') or "")


        
        # Apply augmentation if enabled
        if self.augment and self.mode == 'train' and self.augmentor:
            ts = self.augmentor(ts)

        return {
            "id": item['id'],
            "time_series": ts,
            "context": enhanced_context, # Use enhanced context
            "teacher_thought": teacher_thought,
            "thought": teacher_thought,
            "label": item.get('label', "")
        }

def create_loader(data_path, batch_size=32, mode='train', augment=False, domain_info=None):
    dataset = MTSDataset(data_path, mode=mode, augment=augment, domain_info=domain_info)
    shuffle = (mode == 'train')
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

