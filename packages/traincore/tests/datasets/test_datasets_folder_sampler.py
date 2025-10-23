import numpy as np
import pytest
import soundfile as sf

from traincore.data.sets.folder_sampler import FolderSamplerDataset


@pytest.fixture(scope="session")
def folder_of_folders_music_voiceof_audio_file(tmp_path_factory):
    sr = 44100
    tmp_path = tmp_path_factory.mktemp("audio_files")
    for f in range(2):
        subfolder = tmp_path / f"SP{f:03}"
        subfolder.mkdir()
        for dname in ["VOICE", "MUSIC"]:
            subfolder = subfolder / dname
            subfolder.mkdir()
            for i in range(5):
                path = subfolder / f"audio_{i}.wav"
            sf.write(str(path), np.random.randn(sr, 2) * 2 - 1, sr)
    return tmp_path


def test_folder_sampler_should_have_len_equal_number_of_sub_folders(
    folder_of_folders_music_voiceof_audio_file,
):
    dataset_root = folder_of_folders_music_voiceof_audio_file
    ds = FolderSamplerDataset(
        name="AFolderDataset",
        data_dir=dataset_root,
        suffix=".wav",
    )
    ds.setup()

    assert len(ds) == len(list(dataset_root.iterdir()))


def test_folder_sampler_should_return_a_audio_sample_for_each_sub_folder(
    folder_of_folders_music_voiceof_audio_file,
):
    dataset_root = folder_of_folders_music_voiceof_audio_file
    n_frames = 2000
    ds = FolderSamplerDataset(
        name="AFolderDataset", data_dir=dataset_root, suffix=".wav", max_frames=n_frames
    )
    ds.setup()

    for i in range(len(ds)):
        item = ds[i]
        assert isinstance(item, dict)
        assert item["mix"].shape[-1] == n_frames


def test_folder_sampler_should_handle_custom_glob_to_filter_desired_files(
    folder_of_folders_music_voiceof_audio_file,
):
    dataset_root = folder_of_folders_music_voiceof_audio_file
    ds = FolderSamplerDataset(
        name="AFolderDataset",
        data_dir=dataset_root,
        suffix=".wav",
        glob_str="**/VOICE/*{suffix}",
    )
    ds.setup()

    for i in range(len(ds)):
        item = ds[i]
        assert isinstance(item, dict)
        assert "MUSIC" not in item["id"]
