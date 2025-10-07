import numpy as np
import pytest
import soundfile as sf
from torch import Tensor

from traincore.data.sets.folder_dataset import FolderDataset


@pytest.fixture(scope="session")
def folder_of_audio_files(tmp_path_factory):
    sr = 44100
    tmp_path = tmp_path_factory.mktemp("audio_files")
    for i in range(5):
        path = tmp_path / f"audio_{i}.wav"
        sf.write(str(path), np.random.randn(sr, 2) * 2 - 1, sr)
    return tmp_path


def test_folder_dataset_should_return_datums_of_max_frame_length(folder_of_audio_files):
    n_samples = 48000
    ds = FolderDataset(
        data_dir=folder_of_audio_files,
        suffix=".wav",
        target_sample_rate=44100,
        max_frames=n_samples,
    )
    ds.setup()

    item = ds[0]
    audio = item["mix"]

    assert isinstance(audio, (Tensor, np.typing.NDArray))
    assert audio.dim() == 3
    assert audio.shape[-1] == n_samples
