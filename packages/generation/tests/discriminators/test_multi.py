import pytest
import torch
from generation.models.discriminators.multi import MultiDiscriminator
from generation.models.discriminators.scale import ScaleDiscriminator


@pytest.mark.parametrize("disc_def", (0, [0, 1, 2]))
def test_multi_scale(disc_def):
    n_discriminators = len(disc_def) if isinstance(disc_def, list) else 1
    configs = {
        "downsampling_factor": 4,
        "num_prefix_downsamples": disc_def,
        "num_filters": 16,
        "n_layers": 4,
    }
    multi_discriminator = MultiDiscriminator(ScaleDiscriminator, configs)

    seq_len = 8192
    x = torch.randn(1, 1, 1, seq_len)
    y_hat = multi_discriminator(x)

    assert isinstance(y_hat["estimates"], list)
    assert len(y_hat["estimates"]) == n_discriminators
    assert isinstance(y_hat["feature_maps"], list)
    assert len(y_hat["feature_maps"]) == n_discriminators
