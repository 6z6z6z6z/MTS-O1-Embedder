"""Sampling utilities for metric-learning friendly batches."""
import math
import random
import warnings
from collections import defaultdict

from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    """Yield batches with a fixed number of classes and samples per class."""

    def __init__(
        self,
        labels,
        classes_per_batch,
        samples_per_class,
        drop_last=False,
        seed=None,
        allow_repeated_classes=True,
    ):
        self.labels = ["" if label is None else str(label) for label in labels]
        self.classes_per_batch = max(1, int(classes_per_batch))
        self.samples_per_class = max(1, int(samples_per_class))
        self.drop_last = drop_last
        self.seed = seed
        self.allow_repeated_classes = allow_repeated_classes
        self._rng = random.Random(seed)

        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            if label == "":
                continue
            self.label_to_indices[label].append(idx)

        self.unique_labels = sorted(self.label_to_indices.keys())
        self.batch_size = self.classes_per_batch * self.samples_per_class
        self.num_samples = len(self.labels)
        self.num_labeled_samples = sum(len(indices) for indices in self.label_to_indices.values())

        dropped_unlabeled = self.num_samples - self.num_labeled_samples
        if dropped_unlabeled > 0:
            warnings.warn(f"BalancedBatchSampler ignored {dropped_unlabeled} unlabeled sample(s)")

        if self.num_labeled_samples == 0:
            raise ValueError("BalancedBatchSampler requires at least one labeled sample")

        if not self.allow_repeated_classes and len(self.unique_labels) < self.classes_per_batch:
            raise ValueError(
                "BalancedBatchSampler does not have enough unique labels for the requested classes_per_batch: "
                f"have {len(self.unique_labels)}, need {self.classes_per_batch}"
            )

        self.num_batches = (
            self.num_labeled_samples // self.batch_size
            if drop_last else
            math.ceil(self.num_labeled_samples / self.batch_size)
        )

    def __len__(self):
        return self.num_batches

    def _reshuffle_indices(self, label):
        indices = list(self.label_to_indices[label])
        self._rng.shuffle(indices)
        return indices

    def _choose_labels_for_batch(self):
        if len(self.unique_labels) >= self.classes_per_batch:
            return self._rng.sample(self.unique_labels, self.classes_per_batch)
        if self.allow_repeated_classes:
            return [self._rng.choice(self.unique_labels) for _ in range(self.classes_per_batch)]
        raise ValueError(
            "BalancedBatchSampler cannot form a batch without repeating classes: "
            f"have {len(self.unique_labels)}, need {self.classes_per_batch}"
        )

    def __iter__(self):
        label_pools = {
            label: self._reshuffle_indices(label)
            for label, indices in self.label_to_indices.items()
        }
        label_offsets = {label: 0 for label in self.unique_labels}

        for _ in range(self.num_batches):
            chosen_labels = self._choose_labels_for_batch()

            batch = []
            for label in chosen_labels:
                indices = label_pools[label]
                offset = label_offsets[label]

                needed = self.samples_per_class
                selected = []
                while needed > 0:
                    remaining = len(indices) - offset
                    if remaining <= 0:
                        indices = self._reshuffle_indices(label)
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


class MultiDatasetBatchSampler(Sampler):
    """Balanced batch sampler for multi-dataset training.

    Ensures every batch contains samples from exactly ONE dataset,
    with balanced class sampling within that dataset.

    Parameters
    ----------
    dataset_labels : list of (dataset_name, class_label) tuples, one per sample.
    classes_per_batch : number of classes per batch.
    samples_per_class : samples drawn per class per batch.
    seed : random seed.
    allow_repeated_classes : allow repeating classes when a dataset has fewer
        classes than classes_per_batch.
    """

    def __init__(
        self,
        dataset_labels,
        classes_per_batch,
        samples_per_class,
        seed=None,
        allow_repeated_classes=True,
    ):
        self.classes_per_batch = max(1, int(classes_per_batch))
        self.samples_per_class = max(1, int(samples_per_class))
        self.seed = seed
        self.allow_repeated_classes = allow_repeated_classes
        self._rng = random.Random(seed)
        self.batch_size = self.classes_per_batch * self.samples_per_class

        # Build: dataset → {class → [indices]}
        self.dataset_class_indices: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for idx, (ds, cls) in enumerate(dataset_labels):
            self.dataset_class_indices[ds][str(cls)].append(idx)

        self.datasets = sorted(self.dataset_class_indices.keys())

        # Count batches: each dataset contributes ceil(total_samples / batch_size) batches
        self.batches_per_dataset: dict[str, int] = {}
        for ds in self.datasets:
            n = sum(len(v) for v in self.dataset_class_indices[ds].values())
            self.batches_per_dataset[ds] = max(1, math.ceil(n / self.batch_size))

        self._total_batches = sum(self.batches_per_dataset.values())

    def __len__(self):
        return self._total_batches

    def _sample_batch_for_dataset(self, ds: str, pools: dict, offsets: dict) -> list:
        """Draw one balanced batch from a single dataset."""
        cls_to_idx = self.dataset_class_indices[ds]
        unique_classes = sorted(cls_to_idx.keys())

        if len(unique_classes) >= self.classes_per_batch:
            chosen = self._rng.sample(unique_classes, self.classes_per_batch)
        elif self.allow_repeated_classes:
            chosen = [self._rng.choice(unique_classes) for _ in range(self.classes_per_batch)]
        else:
            chosen = unique_classes  # use all if fewer than requested

        batch = []
        for cls in chosen:
            pool_key = (ds, cls)
            if pool_key not in pools:
                pools[pool_key] = list(cls_to_idx[cls])
                self._rng.shuffle(pools[pool_key])
                offsets[pool_key] = 0

            needed = self.samples_per_class
            while needed > 0:
                pool = pools[pool_key]
                offset = offsets[pool_key]
                remaining = len(pool) - offset
                if remaining <= 0:
                    pools[pool_key] = list(cls_to_idx[cls])
                    self._rng.shuffle(pools[pool_key])
                    offsets[pool_key] = 0
                    pool = pools[pool_key]
                    offset = 0
                    remaining = len(pool)
                take = min(needed, remaining)
                batch.extend(pool[offset:offset + take])
                offsets[pool_key] = offset + take
                needed -= take

        return batch[:self.batch_size]

    def __iter__(self):
        pools: dict = {}
        offsets: dict = {}

        # Build a schedule: repeat each dataset for its allotted number of batches,
        # then shuffle the full schedule so datasets are interleaved.
        schedule = []
        for ds in self.datasets:
            schedule.extend([ds] * self.batches_per_dataset[ds])
        self._rng.shuffle(schedule)

        for ds in schedule:
            yield self._sample_batch_for_dataset(ds, pools, offsets)
