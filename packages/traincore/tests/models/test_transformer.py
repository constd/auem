# """Tests transformer classifier model."""

# import pytest
# import torch

# from auem.models.audiotransformer import SimpleTransformerEncoderClassifier


# @pytest.mark.parametrize("feature_shape", [(None, 256, 128), (None, 64, 128)])
# @pytest.mark.parametrize("n_classes", [2, 10])
# @pytest.mark.parametrize("out_nonlinearity", ["none"])
# def test_create_model(feature_shape, n_classes, out_nonlinearity):
#     """Try creating the cnn model with various parameters."""
#     model = SimpleTransformerEncoderClassifier(
#         feature_shape[-1], num_classes=n_classes, out_nonlinearity=out_nonlinearity
#     )
#     criterion = torch.nn.BCEWithLogitsLoss()

#     # import pdb; pdb.set_trace()
#     batch_shape = (8,) + feature_shape[1:]
#     targets = torch.randint(n_classes, (8,))
#     y = torch.zeros(8, n_classes)
#     y[range(8), targets] = 1
#     batch = torch.rand(*batch_shape)

#     model.train()
#     y_hat = model(batch)
#     assert y_hat.shape[0] == batch.shape[0]
#     assert y_hat.shape[-1] == n_classes

#     # Make sure we can update the weights successfully.
#     # loss = torch.nn.NLLLoss(model.parameters())
#     loss = criterion(y_hat, y)
#     loss.backward()