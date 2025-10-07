import pytest
import torch
from generation.models.generators import MelGanGenerator
from traincore.models.encoders.melspec import MelEncoder


@pytest.fixture(
    params=[
        # (batch, source, channel, mel_band/freq, frames)
        (120, 25600),
        (80, 25600),
        (20, 2560),
    ]
)
def mels_samples(request):
    # param.shape = (batch, source, channels, n_bins, time) <- time in frames
    return request.param


@pytest.mark.parametrize("gan_cls", [MelGanGenerator])
def test_should_have_a_valid_audio_output(
    gan_cls, mels_samples: tuple[torch.Tensor, int]
):
    n_mels, n_frames = mels_samples
    samples = torch.randn(1, 1, 1, n_frames)
    n_residual_layers = 2
    generator = gan_cls(
        n_mels=n_mels,
        n_residual_layers=n_residual_layers,
        encoder=MelEncoder(n_mels=n_mels.item()),
    )

    # y.shape = (batch, source, channels, time) <- time in samples
    y = generator(samples)

    assert 0 < y.shape[-1] < n_frames
    assert y.min() >= -1.0
    assert y.max() <= 1.0
