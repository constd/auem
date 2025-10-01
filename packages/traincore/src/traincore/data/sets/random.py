import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from traincore.config_stores.datasets import dataset_store

__all__ = ["RandomAudioDataset", "RandomAudioWithClassifierDataset"]


@dataset_store(name="random_audio")
class RandomAudioDataset(torch.utils.data.Dataset):
    def __init__(
        self, n_examples: int = 1, n_samples: int = 22050, n_channels: int = 1
    ):
        self.n_examples = n_examples
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.data = None

    def __len__(self) -> int:
        return self.n_examples

    def setup(self, stage: str | None = None) -> None:
        self.data = torch.randn(self.n_examples, self.n_channels, self.n_samples)

    def prepare_data(self) -> None: ...

    def __getitem__(
        self, index: int
    ) -> dict[str, str | Tensor | Float[Tensor, "channel time"]]:
        if self.data is not None:
            return {"audio": self.data[index]}
        return {}


@dataset_store(name="random_class")
class RandomAudioWithClassifierDataset(RandomAudioDataset):
    def __init__(
        self,
        n_examples: int = 1,
        n_samples: int = 22050,
        n_channels: int = 1,
        num_classes: int = 2,
    ):
        super().__init__(n_examples, n_samples, n_channels)
        self.num_classes = num_classes
        self.labels = None

    def setup(self, stage: str | None = None) -> None:
        super().setup()
        self.labels = torch.randint(0, self.num_classes, size=(self.n_examples,))

    def __getitem__(
        self, index: int
    ) -> dict[str, str | Tensor] | Float[Tensor, "channel time"]:
        if self.labels is not None:
            return {
                "class": F.one_hot(self.labels[index], num_classes=self.num_classes),
                **super().__getitem__(index),
            }
        return {}
