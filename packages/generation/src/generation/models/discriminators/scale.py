from jaxtyping import Float
from torch import Tensor, flatten, nn
from torch.nn import functional as F
from traincore.config_stores.models import model_store
from traincore.models.decoders.protocol import DecoderProtocol
from traincore.models.encoders.protocol import EncoderProtocol

from generation.models.discriminators.protocol import DiscriminatorReturnType

__all__ = ["ScaleDiscriminator"]


@model_store(name="scale", group="model/discriminator")
class ScaleDiscriminator(nn.Module):
    def __init__(
        self,
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

        self.model = nn.ModuleList([])

    def forward(
        self,
        x: Float[Tensor, "batch source channel time"],
    ) -> DiscriminatorReturnType:
        fmap: list[Float[Tensor, "..."]] = []

        b, *_, c, t = x.shape
        x = x.squeeze(dim=1)
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
        x: Float[Tensor, "..."] = flatten(x, 1, -1)
        return {"estimate": x, "feature_map": fmap}
