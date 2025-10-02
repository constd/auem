import pytest
import torch

from generation.models.generators import MelGanGenerator


@pytest.fixture(
    params=[
        # (batch, source, channel, mel_band/freq, frames)
        (torch.randn(1, 1, 1, 120, 100), (1, 1, 1, 25600)),
        (torch.randn(1, 1, 1, 80, 100), (1, 1, 1, 25600)),
        (torch.randn(1, 1, 1, 20, 10), (1, 1, 1, 2560)),
    ]
)
def random_mel_spectrograms(request):
    # param.shape = (batch, source, channels, n_bins, time) <- time in frames
    return request.param


@pytest.mark.parametrize("gan_cls", [MelGanGenerator])
def test_should_have_a_valid_audio_output(
    gan_cls, random_mel_spectrograms: tuple[torch.Tensor, int]
):
    random_mel_spectrogram, expected_audio_shape = random_mel_spectrograms
    generator = gan_cls(n_mels=random_mel_spectrogram.shape[-2])

    # y.shape = (batch, source, channels, time) <- time in samples
    y = generator(random_mel_spectrogram)

    assert y.shape == expected_audio_shape
    assert y.min() >= -1.0
    assert y.max() <= 1.0
