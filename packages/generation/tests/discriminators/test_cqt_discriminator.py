import torch
from generation.models.discriminators.cqt import (
    CQTDiscriminator,
    MultiScaleSubbandCQTDiscriminator,
)


def test_cqt_discriminator_basic_configuration_runs_forward_pass():
    hop_length = 512
    sr = 22050
    n_octaves = 5
    bins_per_octave = 24
    n_samples = 8192

    # Default configurations from BigVGAN
    cfg = {
        "cqtd_filters": 32,
        "cqtd_max_filters": 1024,
        "cqtd_filters_scale": 1.0,
        "cqtd_dilations": [1, 2, 4],
        "cqtd_in_channels": 1,
        "cqtd_out_channels": 1,
        "sampling_rate": sr,
    }

    discriminator = CQTDiscriminator(cfg, hop_length, n_octaves, bins_per_octave)

    input_data = torch.randn(1, n_samples)

    output = discriminator(input_data)

    # (estimation, feature maps)
    assert len(output) == 2
    # Check batch dimensions
    assert output[0].shape[0] == input_data.shape[0]
    for fm in output[1]:
        assert fm.shape[0] == input_data.shape[0]

    # Check "out channels" dimensions.
    assert output[0].shape[1] == cfg["cqtd_out_channels"]
    for fm in output[1]:
        assert fm.shape[1] == cfg["cqtd_filters"]

    # The true effective hop length is actually effectively hop_length // 2
    # because the CQT is resampled to 2 * sr internally, but the hop length
    # is not modified.
    approx_expected_n_frames = (n_samples // (hop_length // 2)) + 1
    assert output[0].shape[2] == approx_expected_n_frames
    for fm in output[1]:
        assert fm.shape[2] == approx_expected_n_frames

    assert output[0].shape[3] == 15
    assert output[1][0].shape[3] == (n_octaves * bins_per_octave)


def test_multicqt_discriminator_basic_configuration_runs_forward_pass():
    n_samples = 8192
    sr = 22050
    cfg = {"sampling_rate": sr}

    discriminator = MultiScaleSubbandCQTDiscriminator(cfg)

    input_data = torch.randn(1, n_samples)
    input_data2 = torch.randn(1, n_samples)

    discriminator(input_data, input_data2)
