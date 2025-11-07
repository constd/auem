import torch
from generation.models.discriminators.cqt import (
    CQTDiscriminator,
)


def test_cqt_discriminator_basic_configuration_runs_forward_pass():
    hop_length = 512
    sr = 22050
    n_octaves = 5
    bins_per_octave = 24
    n_samples = 8192
    num_filters = 32

    discriminator = CQTDiscriminator(
        hop_length,
        n_octaves,
        bins_per_octave,
        num_filters=num_filters,
        sample_rate=sr,
    )

    input_data = torch.randn(1, n_samples)

    output = discriminator(input_data)

    # (estimation, feature maps)
    assert len(output) == 2
    # Check batch dimensions
    assert output["estimate"].shape[0] == input_data.shape[0]
    for fm in output["feature_map"]:
        assert fm.shape[0] == input_data.shape[0]

    # Check "out channels" dimensions.
    assert output["estimate"].shape[1] == 1
    for fm in output["feature_map"]:
        assert fm.shape[1] == num_filters

    # The true effective hop length is actually effectively hop_length // 2
    # because the CQT is resampled to 2 * sr internally, but the hop length
    # is not modified.
    approx_expected_n_frames = (n_samples // (hop_length // 2)) + 1
    assert output["estimate"].shape[2] == approx_expected_n_frames
    for fm in output["feature_map"]:
        assert fm.shape[2] == approx_expected_n_frames

    assert output["estimate"].shape[3] == 15
    assert output["feature_map"][0].shape[3] == (n_octaves * bins_per_octave)


# def test_multicqt_discriminator_basic_configuration_runs_forward_pass():
#     n_samples = 8192
#     sr = 22050
#     cfg = {"sampling_rate": sr}

#     discriminator = MultiScaleSubbandCQTDiscriminator(cfg)

#     input_data = torch.randn(1, n_samples)
#     input_data2 = torch.randn(1, n_samples)

#     discriminator(input_data, input_data2)
