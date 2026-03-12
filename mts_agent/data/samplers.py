"""Sampling utilities for metric-learning friendly batches."""
import math
import random
from collections import defaultdict

from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    """Yield batches with a fixed number of classes and samples per class."""

    def __init__(self, labels, classes_per_batch, samples_per_class, drop_last=False):
        self.labels = ["" if label is None else str(label) for label in labels]
        self.classes_per_batch = max(1, int(classes_per_batch))
        self.samples_per_class = max(1, int(samples_per_class))
        self.drop_last = drop_last

        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            if label == "":
                continue
            self.label_to_indices[label].append(idx)

        self.unique_labels = sorted(self.label_to_indices.keys())
        self.batch_size = self.classes_per_batch * self.samples_per_class
        self.num_samples = len(self.labels)
        self.num_batches = self.num_samples // self.batch_size if drop_last else math.ceil(self.num_samples / self.batch_size)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        if not self.unique_labels:
            return

        label_pools = {
            label: random.sample(indices, len(indices))
            for label, indices in self.label_to_indices.items()
        }
        label_offsets = {label: 0 for label in self.unique_labels}

        for _ in range(self.num_batches):
            if len(self.unique_labels) >= self.classes_per_batch:
                chosen_labels = random.sample(self.unique_labels, self.classes_per_batch)
            else:
                chosen_labels = [random.choice(self.unique_labels) for _ in range(self.classes_per_batch)]

            batch = []
            for label in chosen_labels:
                indices = label_pools[label]
                offset = label_offsets[label]

                needed = self.samples_per_class
                selected = []
                while needed > 0:
                    remaining = len(indices) - offset
                    if remaining <= 0:
                        indices = random.sample(self.label_to_indices[label], len(self.label_to_indices[label]))
                        label_pools[label] = indices
                        offset = 0
                        remaining = len(indices)

                    take = min(needed, remaining)
                    selected.extend(indices[offset:offset + take])
                    offset += take
                    needed -= take

                label_offsets[label] = offset
                batch.extend(selected)

            if len(batch) == self.batch_size or not self.drop_last:
                yield batch[:self.batch_size]