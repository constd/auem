from einops import rearrange
from jaxtyping import Float
from torch import Tensor, log10, clamp
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
        f_min: float = 0.,
        f_max: float | None = None,
        sample_rate: float = 44100.0,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.window_size: int = window_size
        self.n_fft: int = n_fft
        self.n_hop: int = n_hop
        self.sample_rate: int = int(sample_rate)
        self.epsilon: float = epsilon
        self.f_min: float = f_min
        self.f_max: float = f_max

        self.mel = MelSpectrogram(
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.n_hop,
            sample_rate=self.sample_rate,
            f_min=self.f_min,
            f_max=self.f_max,
            mel_scale="slaney",
            normalized=False,
            center=False,
        )

    def log_clamp(
        self, mel: Float[Tensor, "batch frequency timeframes"]
    ) -> Float[Tensor, "batch frequency timeframes"]:
        return log10(clamp(mel, min=self.epsilon))

    def forward(
        self, x: Float[Tensor, "batch ... channel time"]
    ) -> Float[Tensor, "batch ... channel frequency timeframes"]:
        # Mel spectrogram only suports two dimensions.
        match x.dim():
            case 3:
                b, c, _ = x.size()
                x_ = rearrange(x, "b c t -> (b c) t")

                X = self.log_clamp(self.mel(x_))

                return rearrange(X, "(b c) f t -> b c f t", b=b, c=c)

            case 4:
                b, s, c, _ = x.size()
                x_ = rearrange(x, "b s c t -> (b s c) t")

                X = self.log_clamp(self.mel(x_))

                return rearrange(X, "(b s c) f t -> b s c f t", b=b, s=s, c=c)
            case _:
                raise ValueError(
                    f"Invalid input shape: {x.dim()}. Expected shapes are 'batch channel time' (3) or 'batch source channel time' (4)"
                )
