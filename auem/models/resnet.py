"""Resnet models."""
import torch.nn as nn
import torchvision.models.resnet as torch_resnet
from torchvision.models.resnet import Bottleneck  # noqa: F401
from torchvision.models.resnet import BasicBlock, conv1x1, conv3x3


def resnet_block(
    block_type: nn.Module,
    planes,
    blocks,
    inplanes,
    stride=1,
    dilate=False,
    previous_dilation=1,
    norm_layer=None,
    groups=1,
    base_width=64,
):
    """Build a single ResNet block."""
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    downsample = None
    dilation = previous_dilation
    if dilate:
        dilation *= stride
        stride = 1
    if stride != 1 or inplanes != planes * block_type.expansion:
        downsample = nn.Sequential(
            conv1x1(inplanes, planes * block_type.expansion, stride),
            norm_layer(planes * block_type.expansion),
        )

    layers = []
    layers.append(
        block_type(
            inplanes,
            planes,
            stride,
            downsample,
            groups,
            base_width,
            previous_dilation,
            norm_layer,
        )
    )
    inplanes = planes * block_type.expansion
    for _ in range(1, blocks):
        layers.append(
            block_type(
                inplanes,
                planes,
                groups=groups,
                base_width=base_width,
                dilation=dilation,
                norm_layer=norm_layer,
            )
        )

    return nn.Sequential(*layers)


class SpectrogramResNet(torch_resnet.ResNet):
    """Copied with slight variation from the original pytorch ResNet.

    The original assumes 3 input planes, and provides no way to configure it.
    """

    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        """Build a ResNet for spectograms.

        Identical to ResNet init with the input plane change.

        But... can't use the parent class's because it would overwrite the parameter
        we want to change ;(
        """
        nn.Module.__init__(self)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        # so that the residual branch starts with zeros
        # and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677  # noqa: E501
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, torch_resnet.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, torch_resnet.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


# Had to copy the below to override with SpectrogramResNet
# removed pretrained and progress.
def _resnet(arch, block, layers, **kwargs):
    """Replacementl helper for creating resnet modules from pytorch."""
    model = SpectrogramResNet(block, layers, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def resnet18(**kwargs):
    """Build a resnet18."""
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    """Build a resnet34."""
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    """Build a resnet50."""
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    """Build a resnet101."""
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    """Build a resnet152."""
    return _resnet("resnet152", Bottleneck, [3, 8, 36, 3], **kwargs)


arch_map = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
}


def from_arch(arch_name, **kwargs):
    """Get a model from it's arch name."""
    model_fn = arch_map[arch_name]
    return model_fn(**kwargs)


def CQTResNet(input_shape, num_classes, arch):
    """Build a spectrogram resnet, including setting the num classes."""
    return from_arch(arch, num_classes=num_classes)
