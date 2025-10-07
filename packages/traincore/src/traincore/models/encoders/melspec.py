from typing import overload

from einops import rearrange
from jaxtyping import Float
from torch import Tensor
from torch.nn import Module
from torchaudio.transforms import MelSpectrogram

from traincore.config_stores.model_encoders import model_encoders_store


@model_encoders_store(name="mel")
class MelEncoder(Module):
    def __init__(
        self,
        n_mels: int = 80,
        window_size: int = 2048,
        n_fft: int = 2048,
        n_hop: int = 512,
        sample_rate: float = 44100.0,
    ) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.window_size: int = window_size
        self.n_fft: int = n_fft
        self.n_hop: int = n_hop
        self.sample_rate: int = int(sample_rate)

        self.mel = MelSpectrogram(
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.n_hop,
            sample_rate=self.sample_rate,
        )

    @overload
    def forward(
        self, x: Float[Tensor, "batch channel time"]
    ) -> Float[Tensor, "batch channel frequency timeframes"]: ...

    @overload
    def forward(
        self, x: Float[Tensor, "batch source channel time"]
    ) -> Float[Tensor, "batch source channel frequency timeframes"]: ...

    def forward(
        self,
        x: Float[Tensor, "batch channel time"]
        | Float[Tensor, "batch source channel time"],
    ) -> (
        Float[Tensor, "batch channel frequency timeframes"]
        | Float[Tensor, "batch source channel frequency timeframes"]
    ):
        # Mel spectrogram only suports two dimensions.
        match x.dim():
            case 3:
                b, c, _ = x.size()
                x_ = rearrange(x, "b c t -> (b c) t")

                X = self.mel(x_)

                return rearrange(X, "(b c) f t -> b c f t")

            case 4:
                b, s, c, _ = x.size()
                x_ = rearrange(x, "b s c t -> (b s c) t")

                X = self.mel(x_)

                return rearrange(X, "(b s c) f t -> b s c f t", b=b, s=s, c=c)
            case _:
                raise ValueError(
                    f"Invalid input shape: {x.dim()}. Expected shapes are 'batch channel time' (3) or 'batch source channel time' (4)"
                )
