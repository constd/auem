from pathlib import Path

from omegaconf import II
from torch import Tensor

from traincore.config_stores.datasets import dataset_store
from traincore.io.audio import load_audio
from traincore.functional.audio import pad_sources


@dataset_store(
    name="folder",
    target_sample_rate=II("recipe.model.sample_rate"),
    num_channels=II("recipe.model.num_channels"),
    max_frames=II("recipe.model.max_frames"),
)
class FolderDataset:
    def __init__(
        self,
        target_sample_rate: int = 44100,
        num_channels: int = 1,
        max_frames: int = 44100,
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

    def prepare_data(self) -> None: ...

    def setup(self, stage: str | None) -> None:
        self.data = list(self.data_dir.glob(f"**/*{self.suffix}"))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        datum = self.data[index]

        audio, _ = load_audio(datum, target_sample_rate=self.target_sample_rate)

        audio = pad_sources(audio, self.max_frames)
        # TODO
        # audio = adjust_channels_to_target(audiol, self.num_channels)

        return {"mix": audio, "id": f"{datum.stem}"}
