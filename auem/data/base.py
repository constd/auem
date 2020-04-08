import pathlib
import numpy as np
import tempfile
from typing import Callable, List, Union
import torch


class CachingDataset():
    """You can use this CachingDataset to provide feature caching to a large
    dataset.

    Parameters
    ----------
    x_files : list of filepaths
        List of source data files

    cache_dir : path-like
        Directory to cache the outputs to. If none, will use a temporary directory.

    save_format : str
        npy
        bcolz
    """
    def __init__(
        self,
        x_files: List[Union[str, pathlib.Path]],
        data_loader_fn: Callable,
        cache_dir: Union[str, pathlib.Path] = None,
        save_format: str = "npy"
    ):
        # The
        self.x_files = x_files
        self.data_loader_fn = data_loader_fn
        self.cache_dir = cache_dir if cache_dir else tempfile.TemporaryDirectory()
        self.cache_dir = pathlib.Path(self.cache_dir)
        self.save_format = save_format

        self._file_cache_lookup = {x: None for x in x_files}

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, index : int) -> dict:
        # Lookup the original
        x_file = self.x_files[index]
        x_file_path = pathlib.Path(x_file)
        
        # If not in the index, create the index
        if self._file_cache_lookup[x_file] is None:
            data = self.data_loader_fn(x_file)
            
            # write the cache
            output_file = self.cache_dir / f"{x_file_path.stem}-cache.npy"
            np.save(output_file, data)

            self._file_cache_lookup[x_file] = output_file
        
            return data

        else:
            return np.load(self._file_cache_lookup[x_file])
