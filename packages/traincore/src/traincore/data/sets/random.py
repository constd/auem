import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from traincore.config_stores.datasets import dataset_store

__all__ = ["RandomAudioDataset", "RandomAudioWithClassifierDataset"]


@dataset_store(name="random_audio")
class RandomAudioDataset(torch.utils.data.Dataset):
    def __init__(self, n_examples: int, n_samples: int, n_channels: int = 1):
        self.n_examples = n_examples
        self.n_channels = n_channels
        self.n_samples = n_samples

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(
        self, index: int
    ) -> dict[str, str | Tensor | Float[Tensor, "channel time"]]:
        return {"audio": torch.randn(self.n_channels, self.n_samples)}


@dataset_store(name="random_class")
class RandomAudioWithClassifierDataset(RandomAudioDataset):
    def __init__(
        self, n_examples: int, n_samples: int, n_channels: int = 1, n_classes: int = 2
    ):
        super().__init__(n_examples, n_samples, n_channels)
        self.n_classes = n_classes

    def __getitem__(
        self, index: int
    ) -> dict[str, str | Tensor] | Float[Tensor, "channel time"]:
        class_index = torch.randint(0, high=self.n_classes, size=(1,))
        return {
            "class": F.one_hot(class_index, num_classes=self.n_classes),
            **super().__getitem__(index),
        }
