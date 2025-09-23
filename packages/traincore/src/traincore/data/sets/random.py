from .protocol import MapDatasetProtocol

import torch
from torch import Tensor

__all__ = ["RandomAudioDataset", "RandomAudioWithClassifierDataset"]


class RandomAudioDataset(MapDatasetProtocol):
    def __init__(self, n_examples: int, n_samples: int):
        self.n_examples = n_examples
        self.n_samples = n_samples

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, index: int) -> dict[str, str | Tensor]:
        return {"audio": torch.randn((1, self.n_samples))}


class RandomAudioWithClassifierDataset(RandomAudioDataset):
    def __init__(self, n_examples: int, n_samples: int, n_classes: int = 2):
        super().__init__(n_examples, n_samples)
        self.n_classes = n_classes

    def __getitem__(self, index: int) -> dict[str, str | Tensor]:
        class_index = torch.randint(0, high=self.n_classes, size=(1,))
        return {"class": class_index, **super().__getitem__(index)}
