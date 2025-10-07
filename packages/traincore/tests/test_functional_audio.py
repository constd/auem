import pytest
import torch

from traincore.functional.audio import pad_sources


@pytest.mark.parametrize("ndim", [2, 3, pytest.param(4, marks=pytest.mark.xfail)])
def test_pad_sources_should_pad_the_last_dimension_to_the_specified_length(ndim: int):
    match ndim:
        case 2:
            test_audio = torch.rand(1, 66)
        case 3:
            test_audio = torch.rand(1, 1, 66)
        case _:
            assert False, f"Invalid number of dimensions {ndim}"

    output = pad_sources(test_audio, 100)
    assert output.shape[-1] == 100
    assert output.ndim == ndim
