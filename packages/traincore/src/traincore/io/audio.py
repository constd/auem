from pathlib import Path
from jaxtyping import Float
from pedalboard.io import AudioFile
from random import randint
import torch
from torch import Tensor


def load_audio(
    audio_path: str | Path,
    target_sample_rate: float | int | None = None,
    max_frames: int | None = None,
) -> tuple[Float[Tensor, "channel time"], int | float]:
    start, stop = 0, None
    with AudioFile(str(audio_path)).resampled_to(float(target_sample_rate)) as af:
        if max_frames is not None:
            start = randint(0, af.frames) if af.frames > max_frames else 0

        if 0 < start < af.frames:
            af.seek(start)
        chunk_size = max_frames if max_frames is not None else af.frames - start
        audio = af.read(chunk_size)
        sample_rate = (
            af.sample_rate if target_sample_rate is None else target_sample_rate
        )
    return torch.from_numpy(audio), sample_rate
