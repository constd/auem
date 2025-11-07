import pytest
import torch

from traincore.models.encoders.cqt import CQTEncoder


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_bins": 84},
        {
            "sample_rate": 44100.0,
            "fmax": 10000.0,
            "n_bins": 48,
            "bins_per_octave": 12,
            "cqt_cls": "CQT2010v2",
        },
    ],
)
def test_cqt_basic(kwargs):
    cqt = CQTEncoder(**kwargs)
    x = torch.randn(1, 1, 22050)
    y = cqt(x)
    assert y.shape[0:3] == (1, 1, kwargs["n_bins"])
