# import pathlib
# from copy import deepcopy
# from typing import Union

# import pandas as pd
# import torch
# import torchaudio
# import torchvision

# __all__ = ["DCASE2020Task1aDataset"]


# class DCASE2020Task1aDataset(torch.utils.data.Dataset):
#     SR = 44100
#     TRAIN_SEC = 3

#     def __init__(
#         self,
#         split_df_path: Union[str, pathlib.Path],
#         metadata_df_path: pd.DataFrame,
#         data_root: pathlib.Path,
#         fold: str = "fold1_train.csv",
#         transforms: torchvision.transforms.Compose | torch.nn.Module | None = None,
#     ):
#         self.data = pd.read_csv(f"{split_df_path}{fold}", sep="\t", header=0)
#         if "train" in fold:
#             df = pd.concat([self.data] * 3, ignore_index=True)
#             df["chunk"] = df.groupby(["filename", "scene_label"]).cumcount() + 1
#             self.data = df
#         self.fold = fold
#         self.metadata = pd.read_csv(metadata_df_path)
#         self.data_root = pathlib.Path(data_root)
#         self.transforms = transforms

#         # convenience convertor from class index to class label
#         self.c2l = {
#             k: v
#             for k, v in enumerate(sorted(self.data["scene_label"].unique().tolist()))
#         }
#         self.l2c = {v: k for k, v in self.c2l.items()}
#         self.classes = list(self.c2l.values())
#         print(self.classes)
#         self.c = len(self.classes)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         row = self.data.iloc[idx]
#         example_fn = row["filename"]
#         scene_label = self.l2c[row["scene_label"]]
#         # row_meta = self.metadata[self.metadata["filename"] == example_fn]
#         audio, sr = torchaudio.load(self.data_root / example_fn, normalization=True)
#         if "train" in self.fold:
#             audio = audio[:, row["chunk"] * sr : (row["chunk"] + self.TRAIN_SEC) * sr]
#         data = deepcopy(audio)
#         data = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.SR)(data)
#         if self.transforms:
#             data = self.transforms(data)
#         sample = {
#             "filename": [row["filename"]],
#             "raw": audio,
#             "X": data,
#             "sr": sr,
#             "label": torch.tensor(scene_label),
#         }
#         return sample

#     def __len__(self) -> int:
#         return len(self.data)
