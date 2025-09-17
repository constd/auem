# """Tests simplecnn model."""

# import pytest
# import torch

# from auem.models.simplenn import SimpleNN


# @pytest.mark.parametrize("feature_shape", [(None, 1, 128, 2206), (None, 1, 128, 1102)])
# @pytest.mark.parametrize("n_classes", [2, 10])
# @pytest.mark.parametrize(
#     "layer_def",
#     [(100,), (10, 5), (1024, 512, 256, 128)],
# )
# def test_create_model(feature_shape, n_classes, layer_def):
#     """Try creating the cnn model with various parameters."""
#     model = SimpleNN(feature_shape, num_classes=n_classes, dense_layer_def=layer_def)

#     batch_shape = (8,) + feature_shape[1:]
#     batch = torch.rand(*batch_shape)

#     y_hat = model(batch)
#     assert len(y_hat) == len(batch)
#     assert y_hat.shape[1] == n_classes