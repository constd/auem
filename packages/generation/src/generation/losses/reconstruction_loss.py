import torch
import torchaudio
from omegaconf import II
from torch import Tensor, nn
from traincore.config_stores.criterions import criterion_store

__all__ = [
    "MelSpecReconstructionLoss",
    "MultiMelSpecReconstructionLoss",
    "MultiSTFTReconstructionLoss",
]


def safe_log(x: Tensor, epsilon: float = 1e-7) -> Tensor:
    return torch.log(torch.clip(x, min=epsilon))


class SpectralReconstructionLoss(nn.Module):
    def __init__(
        self,
        spec_loss_cls: type[nn.Module]
        | type[torchaudio.transforms.Spectrogram]
        | type[torchaudio.transforms.MelSpectrogram],
        spec_loss_kwargs: dict = {},
        sample_rate: int | float = 48000.0,
        hop_length: int = 256,
        apply_log: bool = True,
        epsilon: float = 1e-7,
        weight: float = 1.0,
    ):
        super().__init__()
        self.apply_log_ = apply_log
        self.epsilon = epsilon

        self.spec = spec_loss_cls(**spec_loss_kwargs)
        self.weight_ = weight

    def log(self, t: Tensor) -> Tensor:
        return safe_log(t, epsilon=self.epsilon) if self.apply_log_ else t

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        stft_hat = self.log(self.spec(y_hat))
        stft = self.log(self.spec(y))

        loss = torch.nn.functional.l1_loss(stft, stft_hat)

        return loss * self.weight_


@criterion_store(name="stft")
class STFTReconstructionLoss(SpectralReconstructionLoss):
    def __init__(
        self,
        sample_rate: int | float = 48000.0,
        hop_length: int = 256,
        n_fft: int = 1024,
        weight: float = 1.0,
    ):
        super().__init__(
            sample_rate=sample_rate,
            hop_length=hop_length,
            spec_loss_cls=torchaudio.transforms.Spectrogram,
            spec_loss_kwargs={
                "n_fft": n_fft,
                "center": True,
            },
        )


@criterion_store(name="mel")
class MelSpecReconstructionLoss(SpectralReconstructionLoss):
    def __init__(
        self,
        sample_rate: int | float = 48000.0,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 100,
        weight: float = 1.0,
    ):
        super().__init__(
            sample_rate=sample_rate,
            hop_length=hop_length,
            spec_loss_cls=torchaudio.transforms.MelSpectrogram,
            spec_loss_kwargs={
                "n_fft": n_fft,
                "n_mels": n_mels,
                "center": True,
                "power": 1,
            },
        )


class MultiSpectralReconstructionLoss(nn.Module):
    """A base class for any multi-spectral reconstruction loss, creatable via config."""

    # class MultiMelSpecReconstructionLoss(nn.Module):
    def __init__(
        self,
        spec_loss_cls: type[nn.Module]
        | type[torchaudio.transforms.Spectrogram]
        | type[torchaudio.transforms.MelSpectrogram],
        spec_loss_kwargs: dict = dict(),
        sample_rate: int | float = 48000.0,
        hop_length: list[int] = [256, 512, 1024],
        weight: float = 1.0,
    ):
        super().__init__()
        self.weight_ = weight
        aggregate_kwargs = spec_loss_kwargs

        # Hop length is the one guaranteed list of len[n_modules]
        # So we use that to determine the number of modules.
        spec_loss_kwargs["hop_length"] = hop_length
        self.spec_modules = nn.ModuleList()
        for i in range(len(hop_length)):
            module_kwargs = {}
            for k, v in aggregate_kwargs.items():
                if isinstance(v, (tuple, list)) and len(v) == len(hop_length):
                    module_kwargs[k] = v[i]

            self.spec_modules.append(spec_loss_cls(sample_rate, **module_kwargs))

        # self.spec_modules = nn.ModuleList([
        #     MelSpecReconstructionLoss(sample_rate, n_f, h_l, n_m)
        #     for n_f, h_l, n_m in zip(n_fft, hop_length, n_mels)
        # ])

    def forward(self, y_hat, y) -> Tensor:
        loss = torch.zeros(1, device=y_hat.device, dtype=y_hat.dtype)
        for spec_ in self.spec_modules:
            loss = loss + spec_(y_hat, y)
        loss = loss / len(self.spec_modules)
        return loss * self.weight_


@criterion_store(
    name="multimel",
    sample_rate=II("recipe.model.generator.sample_rate"),
)
class MultiMelSpecReconstructionLoss(MultiSpectralReconstructionLoss):
    def __init__(
        self,
        sample_rate: int | float = 48000.0,
        hop_length: list[int] = [256, 512, 1024],
        n_fft: list[int] = [1024, 2048, 4096],
        n_mels: list[int] = [80, 160, 320],
        weight: float = 1.0,
    ):
        super().__init__(
            spec_loss_cls=MelSpecReconstructionLoss,
            spec_loss_kwargs={"n_fft": n_fft, "n_mels": n_mels},
            sample_rate=sample_rate,
            hop_length=hop_length,
            weight=weight,
        )
        assert len(n_fft) == len(hop_length) == len(n_mels), (
            "n_fft, hop_length, and n_mels must have the same length"
        )


@criterion_store(
    name="multistft",
    sample_rate=II("recipe.model.generator.sample_rate"),
)
class MultiSTFTReconstructionLoss(MultiSpectralReconstructionLoss):
    def __init__(
        self,
        sample_rate: int | float = 48000.0,
        hop_length: list[int] = [256, 512, 1024],
        n_fft: list[int] = [1024, 2048, 4096],
        weight: float = 1.0,
    ):
        super().__init__(
            sample_rate=sample_rate,
            hop_length=hop_length,
            weight=weight,
            spec_loss_cls=STFTReconstructionLoss,
            spec_loss_kwargs={
                "n_fft": n_fft,
            },
        )
