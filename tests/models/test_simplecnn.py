"""Tests simplecnn model."""
import pytest
import torch

from auem.models.simplecnn import SimpleCNNBase


@pytest.mark.parametrize("feature_shape", [(None, 1, 128, 2206), (None, 1, 128, 1102)])
@pytest.mark.parametrize("n_classes", [2, 10])
@pytest.mark.parametrize(
    "conv_def",
    [
        [  # this is the default
            (8, 7, (2,)),
            (16, 5, (1, 2)),
            (32, 3, (1, 2)),
            (32, 4, (2, 3)),
        ],
        [(8, 7, (2,))],
        [(8, 7, (2,)), (16, 3, (2, 3))],
    ],
)
def test_create_model(feature_shape, n_classes, conv_def):
    """Try creating the cnn model with various parameters."""
    model = SimpleCNNBase(feature_shape, num_classes=n_classes, conv_layer_def=conv_def)

    batch_shape = (8,) + feature_shape[1:]
    batch = torch.rand(*batch_shape)

    y_hat = model(batch)
    assert len(y_hat) == len(batch)
    assert y_hat.shape[1] == n_classes
