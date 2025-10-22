from pathlib import Path

from omegaconf import II
from torch import Tensor

from traincore.config_stores.datasets import dataset_store
from traincore.functional.audio import pad_sources
from traincore.io.audio import load_audio

__all__ = ["FolderDataset"]


@dataset_store(
    name="folder",
    target_sample_rate=II("recipe.model.sample_rate"),
    num_channels=II("recipe.model.num_channels"),
    max_frames=II("recipe.model.max_frames"),
)
class FolderDataset:
    def __init__(
        self,
        target_sample_rate: float = 44100,
        num_channels: int = 1,
        max_frames: int | None = 44100,
        name: str = "AFolderDataset",
        data_dir: str | Path = "/data/dataset_dir",
        suffix: str = ".wav",
    ):
        self.target_sample_rate = target_sample_rate
        self.num_channels = num_channels
        self.max_frames = max_frames
        self.name = name
        self.data_dir = Path(data_dir)
        self.suffix = suffix
        self.data: list = []

    def prepare_data(self) -> None: ...

    def setup(self, stage: str | None = None) -> None:
        self.data = sorted(list(self.data_dir.rglob(f"*{self.suffix}")))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        datum = self.data[index]

        # returns (channels, samples)
        audio, _ = load_audio(
            datum,
            target_sample_rate=self.target_sample_rate,
            max_frames=self.max_frames,
        )
        # (channels, samples)
        audio = pad_sources(audio, self.max_frames)
        # TODO
        # audio = adjust_channels_to_target(audio, self.num_channels)
        audio = audio.mean(0, keepdim=True)
        # (source, channels, samples)
        audio = audio.unsqueeze(0)

        return {
            "mix": audio,
            "mix_augmented": audio,
            "target": audio,
            "id": f"{datum.stem}",
        }
