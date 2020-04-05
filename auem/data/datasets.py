import pathlib
from typing import Union

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class DCASE2020Task1a(Dataset):
    def __init__(
        self,
        split_df_path: Union[str, pathlib.Path],
        metadata_df_path: pd.DataFrame,
        data_root: pathlib.Path,
        transform: Compose = None,
    ):
        self.data = pd.read_csv(split_df_path, sep="\t", header=0)
        self.metadata = pd.read_csv(metadata_df_path)
        self.data_root = pathlib.Path(data_root)
        self.transform = transform
        self.label_to_idx = {
            v: k
            for k, v in enumerate(sorted(self.data["scene_label"].unique().tolist()))
        }

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.data.iloc[idx]
        example_fn = row["filename"]
        scene_label = self.label_to_idx[row["scene_label"]]
        # row_meta = self.metadata[self.metadata["filename"] == example_fn]
        y, sr = torchaudio.load(self.data_root / example_fn, normalization=True)
        out = torchaudio.transforms.MelSpectrogram(sr)(y)
        if self.transform:
            out = self.transform(y)
        sample = {
            "filename": [row["filename"]],
            "raw": y,
            "X": out,
            "sr": sr,
            "label": torch.tensor(scene_label)
        }
        return sample

    def __len__(self):
        return len(self.data)
