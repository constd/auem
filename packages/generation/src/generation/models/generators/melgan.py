from typing import ClassVar
from omegaconf import II

import numpy as np
from einops import rearrange
from torch import nn, Tensor
from torch.nn.utils.parametrizations import weight_norm

from traincore.models.encoders.protocol import EncoderProtocol
from traincore.config_stores.models import model_store

__all__ = ["MelGanGenerator"]


class ResnetBlock(nn.Module):
    """The exact resnet block used by MelGan."""

    def __init__(self, dim: int, dilation: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            weight_norm(nn.Conv1d(dim, dim, kernel_size=3, dilation=dilation)),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv1d(dim, dim, kernel_size=1)),
        )
        self.shortcut = weight_norm(nn.Conv1d(dim, dim, kernel_size=1))

    def forward(self, x: Tensor) -> Tensor:
        return self.shortcut(x) + self.block(x)


@model_store(name="melgan", n_mels=II(".encoder.n_mels"))
class MelGanGenerator(nn.Module):
    ratios: ClassVar[list[int]] = [8, 8, 2, 2, 2]

    # TODO: should we have a base class?
    def __init__(
        self,
        n_mels: int = 80,
        ratios: list[int] = [8, 8, 2, 2, 2],
        pad_input: bool = True,
        n_residual_layers: int = -1,
        output_channels: int = 1,
        encoder: EncoderProtocol | None = None,
        sample_rate: float = 44100.0,
        max_frames: int = -1,
    ) -> None:
        super().__init__()
        self.encoder = encoder

        self.n_mels = n_mels
        self.pad_input = pad_input
        self.n_residual_layers = n_residual_layers
        self.output_channels = output_channels
        self.sample_rate = sample_rate
        self.max_frames = max_frames
        self.in_out_kernel_size = 7
        self.hop_length = int(np.prod(self.ratios))
        mult = int(
            2 ** len(self.ratios)
        )  # mult implies how quickly the change of channels happens in the upscaling
        ngf = 32  # growth factor

        model = nn.ModuleList()

        if self.pad_input:
            # 3 because the kernel size is 7
            model.append(nn.ReflectionPad1d(self.in_out_kernel_size // 2))

        # in Conv
        model.append(
            nn.Conv1d(
                self.n_mels,
                mult * ngf,
                kernel_size=self.in_out_kernel_size,
                stride=1,
                padding=0,
            )
        )

        # upsample to raw audio scale
        for _, r in enumerate(self.ratios):
            model += [
                nn.LeakyReLU(0.2),
                weight_norm(
                    nn.ConvTranspose1d(
                        mult * ngf,
                        mult * ngf // 2,
                        kernel_size=r * 2,
                        stride=r,
                        padding=r // 2 + r % 2,
                        output_padding=r % 2,
                    )
                ),
            ]

            for j in range(self.n_residual_layers):
                model += [
                    ResnetBlock(mult * ngf // 2, dilation=3**j),
                ]

            mult //= 2

        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            weight_norm(nn.Conv1d(ngf, self.output_channels, kernel_size=7, padding=0)),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x: Tensor) -> Tensor:
        if self.encoder is not None:
            X = self.encoder(x)
        else:
            X = x

        # x: (batch source channels frequency time)
        # TODO: reshape
        match X.dim():
            case 4:
                b, c, f, t = X.shape
                s = 1
                X = rearrange(X, "b c f t -> (b c) f t")
            case 5:
                b, s, c, f, t = X.shape
                X = rearrange(X, "b s c f t -> (b s c) f t")
            case _:
                raise ValueError(f"Unsupported input dimension: {X.dim()}")
        X_heart = self.model(X)
        return rearrange(X_heart, "(b s c) 1 t -> b s c t", b=b, s=s, c=c)
