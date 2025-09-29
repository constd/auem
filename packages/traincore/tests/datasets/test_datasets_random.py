from torch import Tensor

from traincore.data.sets.random import (
    RandomAudioDataset,
    RandomAudioWithClassifierDataset,
)


def test_RandomAudioDataset_should_return_audio_like_data():
    sr = 22050
    n_examples = 10
    example_length = 1.0
    n_samples = round(sr * example_length)
    ds = RandomAudioDataset(n_examples, n_samples=n_samples)

    assert len(ds) == n_examples
    item: dict[str, str | Tensor] = ds[0]
    assert item["audio"].shape == (1, n_samples)


def test_RandomAudioWithClassifierDataset_should_return_audio_and_class():
    sr = 22050
    n_examples = 10
    example_length = 1.0
    n_samples = round(sr * example_length)
    n_classes = 2
    ds = RandomAudioWithClassifierDataset(
        n_examples, n_samples=n_samples, n_classes=n_classes
    )

    assert len(ds) == n_examples
    item: dict[str, str | Tensor] = ds[0]
    assert item["audio"].shape == (1, n_samples)
    assert item["class"].size() == (1, n_classes) and (
        0 <= item["class"][0][0] < n_classes
    )
