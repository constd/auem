import torch
import torchaudio
from omegaconf import II
from torch import Tensor, nn
from traincore.config_stores.criterions import criterion_store


def safe_log(x: Tensor) -> Tensor:
    return torch.log(torch.clip(x, min=1e-7))


@criterion_store(name="mel")
class MelSpecReconstructionLoss(nn.Module):
    def __init__(
        self,
        sample_rate: float = 48000.0,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 100,
        weight: float = 1.0,
    ):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=int(sample_rate),
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=1,
        )
        self.weight_ = weight

    def forward(self, y_hat, y) -> Tensor:
        mel_hat = safe_log(self.mel_spec(y_hat))
        mel = safe_log(self.mel_spec(y))

        loss = torch.nn.functional.l1_loss(mel, mel_hat)

        return loss * self.weight_


@criterion_store(name="multimel", sample_rate=II("recipe.model.generator.sample_rate"))
class MultiMelSpecReconstructionLoss(nn.Module):
    def __init__(
        self,
        sample_rate: float = 48000.0,
        n_fft: list[int] = [1024, 2048, 4096],
        hop_length: list[int] = [256, 512, 1024],
        n_mels: list[int] = [80, 160, 320],
        weight: float = 1.0,
    ):
        super().__init__()
        assert len(n_fft) == len(hop_length) == len(n_mels), (
            "n_fft, hop_length, and n_mels must have the same length"
        )
        self.mel_specs = nn.ModuleList([
            MelSpecReconstructionLoss(sample_rate, n_f, h_l, n_m)
            for n_f, h_l, n_m in zip(n_fft, hop_length, n_mels)
        ])
        self.weight_ = weight

    def forward(self, y_hat, y) -> Tensor:
        loss = torch.zeros(1, device=y_hat.device, dtype=y_hat.dtype)
        for mel_spec in self.mel_specs:
            loss = loss + mel_spec(y_hat, y)
        loss = loss / len(self.mel_specs)
        return loss * self.weight_
