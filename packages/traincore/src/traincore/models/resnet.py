from typing import Callable

import torch.nn as nn
import torchvision.models.resnet as torch_resnet
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1

from traincore.config_stores.models import model_store
from traincore.models.decoders.protocol import DecoderProtocol
from traincore.models.encoders.protocol import EncoderProtocol

__all__ = ["resnet_block", "BasicBlock", "Bottleneck", "SpectrogramResNet"]


def resnet_block(
    block_type: Callable[..., BasicBlock | Bottleneck | nn.Module],
    planes: int,
    blocks: int,
    inplanes: int,
    stride: int = 1,
    dilate: bool = False,
    previous_dilation: int = 1,
    norm_layer: Callable[..., nn.Module] | None = None,
    groups: int = 1,
    base_width: int = 64,
) -> nn.Sequential:
    """Build a single ResNet block."""
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    downsample = None
    dilation = previous_dilation
    if dilate:
        dilation *= stride
        stride = 1
    if getattr(block_type, "expansion", None) is not None:
        if stride != 1 or inplanes != planes * block_type.expansion:  #  ty: ignore[unresolved-attribute]
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block_type.expansion, stride),  #  ty: ignore[unresolved-attribute]
                norm_layer(planes * block_type.expansion),  #  ty: ignore[unresolved-attribute]
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
    inplanes = planes * block_type.expansion  #  ty: ignore[unresolved-attribute]
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


@model_store(name="resent18", block=BasicBlock, layers=[2, 2, 2, 2])
@model_store(name="resent34", block=BasicBlock, layers=[3, 4, 6, 3])
@model_store(name="resent50", block=Bottleneck, layers=[3, 4, 6, 3])
@model_store(name="resent101", block=Bottleneck, layers=[3, 4, 23, 3])
@model_store(name="resent152", block=Bottleneck, layers=[3, 8, 36, 3])
class SpectrogramResNet(torch_resnet.ResNet):
    """Copied with slight variation from the original pytorch ResNet.

    The original assumes 3 input planes, and provides no way to configure it.
    """

    mtype: str = "e2a"

    def __init__(
        self,
        encoder: EncoderProtocol | None = None,
        decoder: DecoderProtocol | None = None,
        block: type[BasicBlock] | type[Bottleneck] = BasicBlock,
        layers: tuple[int, ...] = (2, 2, 2, 2),
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: list[bool] | None = None,
        norm_layer: Callable[..., nn.Module] | None = None,
    ):
        """Build a ResNet for spectograms.

        Identical to ResNet init with the input plane change.

        But... can't use the parent class's because it would overwrite the parameter
        we want to change ;(
        """
        nn.Module.__init__(self)
        self.encoder: EncoderProtocol | None = encoder
        self.decoder: EncoderProtocol | None = decoder

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer: Callable[..., nn.Module] = norm_layer

        self.inplanes: int = 64
        self.dilation: int = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups: int = groups
        self.base_width: int = width_per_group
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
