"""Select one reusable support set per incremental class."""
import random


class SupportSampler:
    def __init__(self, seed):
        self._rng = random.Random(seed)
        self.indices = {}

    def select(self, class_id, population, count):
        if count <= 0:
            raise ValueError('Support size must be positive')
        candidates = list(population)
        if len(candidates) < count:
            raise ValueError(f'Class {class_id} has fewer than {count} training samples')
        if class_id not in self.indices:
            self.indices[class_id] = self._rng.sample(candidates, count)
        selected = self.indices[class_id]
        if len(selected) != count or not set(selected).issubset(candidates):
            raise ValueError(f'Class {class_id}: support pool changed during the run')
        return selected.copy()
