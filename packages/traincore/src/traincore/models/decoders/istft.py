import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, istft
from torch.nn import Module, Parameter

from traincore.config_stores.model_decoders import model_decoders_store


@model_decoders_store(name="istft")
class iSTFTDecoder(Module):
    def __init__(
        self,
        n_fft=2048,
        hop_length=512,
        win_length=2048,
        normalized: bool = False,
        center: bool = True,
        window: str = "hann_window",
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.normalized = normalized
        self.center = center
        self.window = window

        self.window = Parameter(
            getattr(torch, self.window, "hann_window")(n_fft),  # ty: ignore[call-non-callable]
            requires_grad=False,
        )

    def forward(
        self,
        X: Float[Tensor, "batch channel frequency time"],
        length: int | None = None,
    ) -> Float[Tensor, "batch channel time"]:
        b, c, f, t = X.size()
        X_ = rearrange(
            X, "batch channel frequency time -> (batch channel) frequency time"
        )
        x_: Tensor = istft(
            X_,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=self.center,
            normalized=self.normalized,
            onesided=True,
            length=length,
        )
        x = rearrange(
            x_, "(batch channel) time -> batch channel time", batch=b, channels=c
        )
        return x
