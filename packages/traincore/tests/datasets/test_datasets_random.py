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
    ds.setup()

    assert len(ds) == n_examples
    item: dict[str, str | Tensor] = ds[0]
    assert item["audio"].shape == (1, 1, n_samples)  # ty: ignore[possibly-unbound-attribute]


def test_RandomAudioWithClassifierDataset_should_return_audio_and_class():
    sr = 22050
    n_examples = 10
    example_length = 1.0
    n_samples = round(sr * example_length)
    num_classes = 2
    ds = RandomAudioWithClassifierDataset(
        n_examples, n_samples=n_samples, num_classes=num_classes
    )
    ds.setup()

    assert len(ds) == n_examples
    item: dict[str, str | Tensor] | Tensor = ds[0]
    assert item["audio"].shape == (1, 1, n_samples)  # ty: ignore[possibly-unbound-attribute]
    assert item["class"].size() == (num_classes,)  # ty: ignore[possibly-unbound-attribute]
