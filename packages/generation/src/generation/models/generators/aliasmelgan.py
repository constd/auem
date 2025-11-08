import numpy as np
from einops import rearrange
from jaxtyping import Float
from omegaconf import II
from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm
from traincore.config_stores.models import model_store
from traincore.models.activations.afa import Activation1d
from traincore.models.activations.snake import Snake, SnakeBeta
from traincore.models.encoders.protocol import EncoderProtocol

__all__ = ["AliasFreMelGanGenerator"]


class ResnetBlock(nn.Module):
    """The exact resnet block used by MelGan."""

    def __init__(self, dim: int, dilation: int = 1, activation: str = "snake") -> None:
        super().__init__()
        match activation:
            case "Snake":
                activation_cls = Snake
            case "SnakeBeta":
                activation_cls = SnakeBeta
            case _:
                activation_cls = nn.LeakyReLU
        if activation_cls is None:
            raise ValueError(
                f"unknown activation: '{activation}'. Possible choices are: Snake, SnakeBeta"
            )
        self.block = nn.Sequential(
            Activation1d(activation_cls(dim)),
            nn.ReflectionPad1d(dilation),
            weight_norm(nn.Conv1d(dim, dim, kernel_size=3, dilation=dilation)),
            Activation1d(activation_cls(dim)),
            weight_norm(nn.Conv1d(dim, dim, kernel_size=1)),
        )
        self.shortcut = weight_norm(nn.Conv1d(dim, dim, kernel_size=1))

    def forward(self, x: Tensor) -> Tensor:
        return self.shortcut(x) + self.block(x)


@model_store(name="melganaf", n_mels=II(".encoder.n_mels"))
class AliasFreMelGanGenerator(nn.Module):
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
        num_channels: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = encoder

        self.n_mels = n_mels
        self.ratios = ratios
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
            # nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)

    def forward(
        self,
        x: Float[Tensor, "batch channel time"],
    ) -> Float[Tensor, "batch source channel generated_time"]:
        X = self.encoder(x)

        # x: (batch source channels frequency time)
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
