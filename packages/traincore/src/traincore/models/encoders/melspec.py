from einops import reshape
from jaxtyping import Array, Float
from torch.nn import Module
from torchaudio.transforms import MelSpectrogram

from traincore.config_stores.model_encoders import model_encoders_store


@model_encoders_store.register("mel")
class MelEncoder(Module):
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

        self.mel = MelSpectrogram(
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.n_hop,
            sample_rate=self.sample_rate,
        )

    def forward(
        self, x: Float[Array, "batch channel time"]
    ) -> Float[Array, "batch channel frequency time"]:
        b, c, t = x.size()
        x_ = reshape(x, "b c t -> (b c) t")
        X = self.mel(x_)
        return reshape(X, "(b c) f t -> b c f t")
