"""Test resnet models."""

# noqa: D400, D103
import torch

import traincore.models.resnet as auem_resnet


def test_resent_basicblock():
    """Test building a resnet block without building the whole model."""
    batch_size, freq_bins, time_steps = (8, 128, 256)
    in_planes = 1
    features = torch.rand((batch_size, 1, freq_bins, time_steps))
    n_planes = 16
    n_blocks = 3

    resnet_block = auem_resnet.resnet_block(
        auem_resnet.BasicBlock,
        n_planes,
        n_blocks,
        in_planes,
    )
    assert len(resnet_block) == n_blocks

    output = resnet_block(features)
    assert output.shape == (batch_size, n_planes, freq_bins, time_steps)


# @pytest.mark.parametrize("feature_shape", [(None, 1, 128, 256), (None, 1, 128, 64)])
# @pytest.mark.parametrize("n_classes", [2, 10])
# @pytest.mark.parametrize("arch", ["resnet18", "resnet34"])
# def test_create_model(feature_shape, n_classes, arch):
#     """Test creating resnet model."""
#     model = auem_resnet.CQTResNet(feature_shape, num_classes=n_classes, arch=arch)

#     criterion = torch.nn.BCEWithLogitsLoss()

#     batch_shape = (8,) + feature_shape[1:]
#     targets = torch.randint(n_classes, (8,))
#     y = torch.zeros(8, n_classes)
#     y[range(8), targets] = 1
#     batch = torch.rand(*batch_shape)

#     model.train()
#     y_hat = model(batch)
#     assert len(y_hat) == len(batch)
#     assert y_hat.shape[1] == n_classes

#     # Make sure we can update the weights successfully.
#     # loss = torch.nn.NLLLoss(model.parameters())
#     loss = criterion(y_hat, y)
#     loss.backward()
