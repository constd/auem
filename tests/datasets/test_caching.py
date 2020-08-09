import librosa
import numpy as np
import pytest
import soundfile as sf

import auem.data.caching


@pytest.mark.xfail(reason="TODO")
class TestCachingDataset:
    @pytest.fixture(scope="session")
    def example_dataset(self, tmp_path_factory):
        def _generate_wav(rate):
            return np.random.uniform(-1, 1, size=(int(rate * 10), 2))

        dataset_root = tmp_path_factory.mktemp("dataset")
        sr = (44_100) // 2

        files = []
        for i in range(5):
            audio_path = dataset_root / f"audio{i}.wav"

            sf.write(str(audio_path), _generate_wav(sr), sr, subtype="PCM_16")
            files.append(audio_path)

        return files

    def test_basic(self, example_dataset, tmpdir):
        def data_loader(audio_file):
            y, sr = librosa.load(audio_file)

            # Return the first 1s
            return y[:sr]

        dataset = auem.data.caching.CachingDataset(
            example_dataset, data_loader, cache_dir=str(tmpdir)
        )
        for fp, cache in dataset._file_cache_lookup.items():
            assert fp.exists()
            assert cache is None

        # make sure the cache loads
        for i in range(len(dataset)):
            assert dataset._file_cache_lookup[dataset.x_files[i]] is None
            audio = dataset[i]
            assert isinstance(audio, np.ndarray)
            assert len(audio) == 22050
            assert dataset._file_cache_lookup[dataset.x_files[i]] is not None

        # do it again; should use the cache this time.
        for i in range(len(dataset)):
            assert dataset._file_cache_lookup[dataset.x_files[i]] is not None
            audio = dataset[i]
            assert isinstance(audio, np.ndarray)
            assert len(audio) == 22050
            assert dataset._file_cache_lookup[dataset.x_files[i]] is not None
