from einops import rearrange
from jaxtyping import Float
from torch import Tensor, flatten, nn
from torch.nn.utils.parametrizations import spectral_norm, weight_norm
from traincore.config_stores.models import model_store
from traincore.models.decoders.protocol import DecoderProtocol
from traincore.models.encoders.protocol import EncoderProtocol

from generation.models.discriminators.protocol import DiscriminatorReturnType

__all__ = ["ScaleDiscriminator"]


@model_store(name="scale", group="model/discriminator")
class ScaleDiscriminator(nn.Module):
    def __init__(
        self,
        num_filters: int,
        n_layers: int,
        downsampling_factor: int,
        num_prefix_downsamples: int = 0,
        use_spectral_norm: bool = True,
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

        self.use_spectral_norm = use_spectral_norm

        self.num_filters = num_filters
        self.n_layers = n_layers
        self.downsampling_factor = downsampling_factor
        self.num_prefix_downsamples = num_prefix_downsamples

        self.n_layers = n_layers
        self.downsampling_factor = downsampling_factor

        self.model = nn.ModuleList([])
        norm = spectral_norm if use_spectral_norm else weight_norm

        self.downsample = (
            nn.AvgPool1d(
                4,
                stride=2**self.num_prefix_downsamples,
                padding=1,
                count_include_pad=False,
            )
            if self.num_prefix_downsamples > 0
            else None
        )

        # "In conv"
        self.model.append(
            nn.Sequential(
                nn.ReflectionPad1d(7),
                norm(nn.Conv1d(1, self.num_filters, kernel_size=15)),
                nn.LeakyReLU(0.2, True),
            )
        )

        nf = self.num_filters
        stride = self.downsampling_factor
        for _ in range(self.n_layers):
            nf_prev = nf
            nf = min(nf * stride, 1024)

            self.model.append(
                nn.Sequential(
                    norm(
                        nn.Conv1d(
                            nf_prev,
                            nf,
                            kernel_size=stride * 10 + 1,
                            stride=stride,
                            padding=stride * 5,
                            groups=nf_prev // 4,
                        )
                    ),
                    nn.LeakyReLU(0.2, True),
                )
            )

        nf = min(nf * 2, 1024)
        self.model.append(
            nn.Sequential(
                norm(nn.Conv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2)),
                nn.LeakyReLU(0.2, True),
            )
        )

        self.model.append(norm(nn.Conv1d(nf, 1, kernel_size=3, stride=1, padding=1)))

    def forward(
        self,
        x: Float[Tensor, "batch source channel time"],
    ) -> DiscriminatorReturnType:
        fmap: list[Float[Tensor, "..."]] = []
        b, s, c, t = x.size()
        x = rearrange(x, "b s c t -> (b s) c t")
        if self.downsample:
            x = self.downsample(x)

        for layer in self.model:
            x = layer(x)
            fmap.append(x)

        fmap.append(x)
        x: Float[Tensor, "..."] = flatten(x, 1, -1)
        return {"estimate": x, "feature_map": fmap}
