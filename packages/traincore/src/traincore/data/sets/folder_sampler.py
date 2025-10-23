from pathlib import Path
import random

from omegaconf import II
from torch import Tensor

from traincore.config_stores.datasets import dataset_store

from traincore.data.sets.folder_dataset import FolderDataset

__all__ = ["FolderSamplerDataset"]


@dataset_store(
    name="foldersampler",
    target_sample_rate=II("recipe.model.sample_rate"),
    num_channels=II("recipe.model.num_channels"),
    max_frames=II("recipe.model.max_frames"),
)
class FolderSamplerDataset:
    """Dataset for sampling subdirectories of audio files from a folder.

    Each subdirectory is sampled uniformly.
    """

    def __init__(
        self,
        target_sample_rate: float = 44100.0,
        num_channels: int = 1,
        max_frames: int = 44100,
        name: str = "AFolderSamplerDataset",
        data_dir: str | Path = "/data/dataset_dir",
        glob_str: str | None = None,
        suffix: str = ".wav",
    ):
        self.target_sample_rate = target_sample_rate
        self.num_channels = num_channels
        self.max_frames = max_frames
        self.name = name
        self.data_dir = Path(data_dir)
        self.glob_str = glob_str
        self.suffix = suffix

    def prepare_data(self) -> None:
        for _, v in self.subdatasets.items():
            v.prepare_data()

    def setup(self, stage: str | None = None) -> None:
        self.child_keys = [
            x.name
            for x in self.data_dir.iterdir()
            if x.is_dir() and not x.name.startswith(".")
        ]
        self.subdatasets = {
            child_folder: FolderDataset(
                target_sample_rate=self.target_sample_rate,
                num_channels=self.num_channels,
                max_frames=self.max_frames,
                data_dir=self.data_dir / child_folder,
                glob_str=self.glob_str,
                suffix=self.suffix,
            )
            for child_folder in self.child_keys
        }
        for _, v in self.subdatasets.items():
            v.setup(stage)

    def __len__(self) -> int:
        return len(self.child_keys)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        # Select a folder from the index.
        child_folder = self.child_keys[index]

        # Select a file from the folder.
        selected_ds = self.subdatasets[child_folder]
        index = random.randint(0, len(selected_ds) - 1)
        item = selected_ds[index]
        item["id"] = f"{child_folder}/{item['id']}"

        return item
