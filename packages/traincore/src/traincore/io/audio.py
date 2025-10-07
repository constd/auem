from pathlib import Path
from jaxtyping import Float
import soundfile as sf
import torch
from torch import Tensor


def load_audio(
    audio_path: str | Path, target_sample_rate: float | int | None = None
) -> tuple[Float[Tensor, "channel frame"], float | int]:
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    audio_data, sample_rate = sf.read(audio_path, always_2d=True)
    audio_data = torch.as_tensor(audio_data).transpose(0, 1).float()
    return audio_data, sample_rate
