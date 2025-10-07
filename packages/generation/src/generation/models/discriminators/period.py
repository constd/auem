from jaxtyping import Float
from torch import Tensor, flatten, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm, weight_norm
from traincore.config_stores.models import model_store
from traincore.models.decoders.protocol import DecoderProtocol
from traincore.models.encoders.protocol import EncoderProtocol

from generation.models.discriminators.protocol import DiscriminatorReturnType

__all__ = ["PeriodDiscriminator"]


@model_store(name="model/discriminator/period")
class PeriodDiscriminator(nn.Module):
    def __init__(
        self,
        period: int = 2,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
        discriminator_channel_multiplier: int = 4,
        encoder: EncoderProtocol | None = None,
        decoder: DecoderProtocol | None = None,
        sample_rate: float = 44100.0,
        num_samples: int = -1,
    ) -> None:
        super().__init__()
        self.mtype = "a2e"
        self.encoder = encoder
        self.decoder = decoder

        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.period = period
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_spectral_norm = use_spectral_norm
        self.d_mult = 4  # discriminator_channel_multiplier

        dilation = 1
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.model = nn.ModuleList([
            norm(
                nn.Conv2d(
                    1,
                    32 * self.d_mult,
                    kernel_size=(self.kernel_size, 1),
                    stride=(self.stride, 1),
                    padding=((self.kernel_size * dilation - dilation) // 2, 0),
                )
            ),
            norm(
                nn.Conv2d(
                    32 * self.d_mult,
                    128 * self.d_mult,
                    kernel_size=(self.kernel_size, 1),
                    stride=(self.stride, 1),
                    padding=((self.kernel_size * dilation - dilation) // 2, 0),
                )
            ),
            norm(
                nn.Conv2d(
                    128 * self.d_mult,
                    512 * self.d_mult,
                    kernel_size=(self.kernel_size, 1),
                    stride=(self.stride, 1),
                    padding=((self.kernel_size * dilation - dilation) // 2, 0),
                )
            ),
            norm(
                nn.Conv2d(
                    512 * self.d_mult,
                    1024 * self.d_mult,
                    kernel_size=(self.kernel_size, 1),
                    stride=(self.stride, 1),
                    padding=((self.kernel_size * dilation - dilation) // 2, 0),
                )
            ),
            norm(
                nn.Conv2d(
                    1024 * self.d_mult,
                    1024 * self.d_mult,
                    kernel_size=(self.kernel_size, 1),
                    stride=1,
                    padding=((2, 0),),
                )
            ),
        ])
        self.conv_post = norm(
            nn.Conv2d(int(1024 * self.d_mult), 1, (3, 1), 1, padding=(1, 0))
        )

    def forward(
        self,
        x: Float[Tensor, "batch channel time"],
        x_hat: Float[Tensor, "batch channel time"],
    ) -> DiscriminatorReturnType:
        fmap = []

        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.model:
            x = layer(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = flatten(x, 1, -1)
        return {"x": x, "fmap": fmap}
