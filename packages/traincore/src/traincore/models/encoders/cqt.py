from einops import rearrange
from jaxtyping import Float
from nnAudio import features
from torch import Tensor
from torch.nn import Module

from traincore.config_stores.model_encoders import model_encoders_store

__all__ = ["CQTEncoder"]


@model_encoders_store(name="cqt")
class CQTEncoder(Module):
    """nnAudio CQT calculates the Constant-Q Transform of an audio signal.

    Input signal should be in one of the following formats:
    - (batch, time)
    - (batch, channel, time)
    - (batch, source, channel, time)

    nnAudio's CQT operates only on (1, time), so if batch or channel are provided, they will be
    collaped and re-exploded to the original shape.
    """

    def __init__(
        self,
        sr: float = 22050,
        hop_length: int = 512,
        fmin: float = 32.7,
        fmax: float | None = None,
        n_bins: int = 84,
        bins_per_octave: int = 12,
        filter_scale: float = 1.0,
        norm: int = 1,
        window: str | float | tuple = "hann",
        cqt_cls: str = "CQT2010v2",
        **kwargs,
    ):
        super(CQTEncoder, self).__init__()
        self.cqt = getattr(features, cqt_cls)(
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            filter_scale=filter_scale,
            norm=norm,
            window=window,
            **kwargs,
        )

    def forward(
        self, x: Float[Tensor, "batch ... time"]
    ) -> Float[Tensor, "batch ... freq timeframes"]:
        try:
            match x.dim():
                case 3:
                    b, c, _ = x.size()
                    x_ = rearrange(x, "b c t -> (b c) t")

                    X = self.cqt(x_)

                    return rearrange(X, "(b c) f t -> b c f t", b=b, c=c)

                case 4:
                    b, s, c, _ = x.size()
                    x_ = rearrange(x, "b s c t -> (b s c) t")

                    X = self.cqt(x_)

                    return rearrange(X, "(b s c) f t -> b s c f t", b=b, s=s, c=c)
                case _:
                    raise ValueError(
                        f"Invalid input shape: {x.dim()}. Expected shapes are 'batch channel time' (3) or 'batch source channel time' (4)"
                    )
        except RuntimeError as e:
            raise RuntimeError(
                f"Error in CQT forward pass: {e} -- min len for cqt kernel_width is: {(self.cqt.kernel_width // 2) + 1}"
            )
