"""Utilities for caching."""
import pathlib
import tempfile
from collections import defaultdict
from typing import Any, List, Union

import numpy as np


class FeatureCache:
    """You can use this FeatureCache to provide feature caching for a large dataset.

    Parameters
    ----------
    cache_dir : path-like
        Directory to cache the outputs to. If none, will use a temporary directory.
    """

    def __init__(
        self, cache_dir: Union[str, pathlib.Path] = None,
    ):
        self.cache_dir = cache_dir if cache_dir else tempfile.TemporaryDirectory()
        self.cache_dir = pathlib.Path(self.cache_dir)

        self._stats = defaultdict(int)

    def clean(self) -> None:
        """Clean the cache."""
        raise NotImplementedError("TODO")

    def files(self) -> List[pathlib.Path]:
        """List all the files in the cache."""
        return [x for x in self.cache_dir.glob("*.npy")]

    def __len__(self) -> int:
        """Return the size of the cache."""
        return len(self.files())

    def load(
        self, loader_fn, filepath_key: Union[str, pathlib.Path], *args, **kwargs
    ) -> Any:
        """Return the data defined by the args/kwargs.

        If the data exists in the cache, returns the data instead of performing
        the load.

        The first argument *must* be the path to the file to load; the file will be used
        as the key in the cache.
        """
        filepath_key = pathlib.Path(filepath_key)
        filepath_stem = filepath_key.stem
        cache_filepath = self.cache_dir / f"{filepath_stem}-cache.npy"

        # If the data exists, just load it.
        if cache_filepath.exists():
            self._stats["cache_hit"] += 1
            return np.load(cache_filepath, allow_pickle=True)

        # Otherwise, save it.
        else:
            self._stats["cache_miss"] += 1
            if not self.cache_dir.exists():
                self.cache_dir.mkdir(parents=True, exist_ok=True)

            data = loader_fn(filepath_key, *args, **kwargs)

            np.save(cache_filepath, data, allow_pickle=True)
            return data
