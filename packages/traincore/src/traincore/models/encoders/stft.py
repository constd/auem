import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, stft
from torch.nn import Module

from traincore.config_stores.model_encoders import model_encoders_store


@model_encoders_store(name="stft")
class STFTEncoder(Module):
    def __init__(
        self,
        window_size: int = 2048,
        n_fft: int = 2048,
        n_hop: int = 512,
        sample_rate: float = 44100.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.sample_rate = int(sample_rate)
        self.normalized: bool = False
        self.window = getattr(torch, "hann_window")(self.n_fft)

    def forward(
        self, x: Float[Tensor, "batch channel time"]
    ) -> Float[Tensor, "batch channel frequency time"]:
        b, c, t = x.size()
        # simplify shape
        x_ = rearrange(x, "b c t -> (b c) t")
        X_ = stft(
            x_,
            n_fft=self.n_fft,
            hop_length=self.n_hop,
            window=self.window,
            center=True,
            return_complex=True,
            normalized=False,
            onesided=True,
            pad_mode="reflect",
        )
        # get back to the original batch/channels
        X = rearrange(X_, "(b c) f t -> b c f t")
        return torch.view_as_real(X)
