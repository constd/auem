import pytest
import torch
from generation.losses.reconstruction_loss import (
    MultiMelSpecReconstructionLoss,
    MultiSTFTReconstructionLoss,
)


@pytest.mark.parametrize(
    "criterion_cls,kwargs",
    [
        (MultiMelSpecReconstructionLoss, {}),
        (MultiSTFTReconstructionLoss, {}),
        (
            MultiSTFTReconstructionLoss,
            {
                "hop_length": [512, 1024, 2048],
                "n_fft": [512, 1024, 2048],
            },
        ),
    ],
)
@pytest.mark.parametrize("batch_size", [1, 2])
def test_multi_spectral_reconstruction_loss(batch_size, criterion_cls, kwargs):
    sr = 22050
    samples = round(sr * 0.5)
    audio1 = torch.randn(batch_size, 1, samples) * 2 - 1
    audio2 = torch.randn(batch_size, 1, samples) * 2 - 1

    loss_fn = criterion_cls(sample_rate=sr, **kwargs)
    loss = loss_fn(audio1, audio2)

    assert len(loss_fn.spec_modules) == 3

    # loss is just a single value!
    assert loss.ndim == 1
    assert loss.shape[0] == 1
