import torch
from torch import Tensor
from torch.nn import functional as F


def pad_sources(audio: Tensor, max_frames: int | None) -> Tensor:
    """Pads the final dimension of a time domain audio signal to the specified length."""
    if max_frames is not None and audio.shape[-1] < max_frames:
        audio = F.pad(
            audio,
            (0, max_frames - audio.shape[-1]),
            mode="constant",
            value=torch.finfo(audio.dtype).eps,
        )
    return audio
