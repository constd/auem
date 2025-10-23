import pytest
import numpy as np
import soundfile as sf

from traincore.io.audio import load_audio


@pytest.fixture(params=[1, 2])
def existing_audio_file_and_samples_channels(request, tmp_path):
    num_channels = request.param
    num_samples = 1000
    audio_data = np.random.rand(num_samples, num_channels) * 2 - 1
    sample_rate = 44100
    audio_path = tmp_path / "audio.wav"
    sf.write(str(audio_path), audio_data, sample_rate)
    return audio_path, num_samples, num_channels


def test_load_audio_should_return_samples_from_existing_wav_file(
    existing_audio_file_and_samples_channels,
):
    audio_path, expected_samples, expected_channels = (
        existing_audio_file_and_samples_channels
    )

    audio_data, sr = load_audio(audio_path)
    assert audio_data.shape[0] == expected_channels
    assert audio_data.shape[1] == expected_samples
    assert sr == 44100


def test_requesting_start_end_sample_should_return_samples_from_existing_wav_file(
    existing_audio_file_and_samples_channels,
):
    audio_path, _, expected_channels = existing_audio_file_and_samples_channels

    start_sample = 100
    num_samples = 100
    audio_data, sr = load_audio(audio_path, start=start_sample, frames=num_samples)
    assert audio_data.shape[0] == expected_channels
    assert audio_data.shape[1] == num_samples
    assert sr == 44100


@pytest.mark.xfail
def test_loading_audio_at_requested_sample_rate_should_resample(
    existing_audio_file_and_samples_channels,
):
    audio_path, expected_samples, expected_channels = (
        existing_audio_file_and_samples_channels
    )

    sr = 22050
    audio_data, sr_returned = load_audio(audio_path, sr)
    assert audio_data.shape[0] == expected_channels
    assert audio_data.shape[1] == expected_samples // 2
    assert sr_returned == sr
