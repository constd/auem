"""Dataset classes for Google's AudioSet dataset."""
import csv
import json
import logging
import math
import time
import warnings
from copy import deepcopy
from csv import DictReader
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import librosa
import numpy as np
import pescador
import torch
import torchvision

from auem.data.caching import FeatureCache

logger = logging.getLogger(__name__)

__all__ = [
    "AudiosetAnnotationReaderV1",
    "AudiosetAnnotationReaderV2",
    "AudiosetDataset",
    "IterableAudiosetDataset",
]


class AudiosetAnnotationReaderV1(DictReader):
    """Annotation Reader which wraps csv's DictReader."""

    def __init__(
        self,
        f,
        ontology=None,
        fieldnames=None,
        fieldtypes=None,
        restkey=None,
        restval=None,
        dialect="excel",
        *args,
        **kwds,
    ):
        self.ontology = self.__load_ontology(ontology)
        self.annotation_stats = None
        self.annotation_creation_date = None
        if fieldtypes is None:
            fieldtypes = {"YTID": str, "end_seconds": float, "start_seconds": float}
        self.fieldtypes = fieldtypes
        super().__init__(
            f,
            fieldnames=fieldnames,
            restkey=restkey,
            restval=restval,
            dialect=dialect,
            *args,
            **kwds,
        )
        self.init_properties()

    @staticmethod
    def __load_ontology(ontology):
        """Return the ontology as a map."""
        if ontology is not None:
            return {x["id"]: x for x in ontology}
        return ontology

    def init_properties(self):
        """Load the properties from the .csv file, which are contained in comments."""
        row = next(self.reader)
        self.annotation_creation_date = [x.strip("#").rstrip().lstrip() for x in row]
        row = next(self.reader)
        self.annotation_stats = {
            x.split("=")[0].strip("""#|\' \'|\n"""): int(x.split("=")[1]) for x in row
        }
        row = next(self.reader)
        row = [x.strip("""#|\' \'|\n""") for x in row]
        self.fieldnames = row[:-1]
        if self.restkey is None:
            self.restkey = row[-1]
        self.line_num = self.reader.line_num

    def __next__(self):
        """Process one row of the csv at a time."""
        row = next(self.reader)
        self.line_num = self.reader.line_num

        while row == []:
            row = next(self.reader)
        if self.fieldtypes is None:
            d = dict(zip(self.fieldnames, row))
        else:
            # converts fields to a specific datatype (e.g. strings to floats)
            d = {f: self.fieldtypes[f](r) for f, r in zip(self.fieldnames, row)}

        lf = len(self.fieldnames)
        lr = len(row)
        if lf < lr:
            classes = [x.strip("' '|\"") for x in row[lf:]]
            if self.ontology is not None:
                classes = {x: self.ontology[x] for x in set(classes)}
            d[self.restkey] = classes
        elif lf > lr:
            for key in self.fieldnames[lr:]:
                d[key] = self.restval
        return d


class AudiosetAnnotationReaderV2:
    """Process the annotation file using the csv reader.

    Loads the entire file at one time, instead of processing line-by-line.

    Parameters
    ----------
    annotation_path
        Path to an Audioset annotations file, which is a csv with four columns:
        - youtube ID
        - Start offset (s)
        - End offset (s)
        - List of classes
    """

    def __init__(self, annotation_path: Union[Path, str], classes: List):
        self.annotation_path = annotation_path
        self.classes = classes

        self._annotations = None
        self._idx = None

    def _load_annotations(self):
        """Load the entire annotations file, skipping any "comment" lines."""
        self._annotations = []
        with open(self.annotation_path, "r") as fh:
            for l in csv.reader(
                fh, quotechar='"', delimiter=",", skipinitialspace=True
            ):
                if not l[0].startswith("#"):
                    self._annotations.append(l)

    def __len__(self):
        """Return how many annotations were provided in this file."""
        if not self._annotations:
            self._load_annotations()

        return len(self._annotations)

    def __getitem__(self, idx):
        """Return a single line from the file, by index."""
        if not self._annotations:
            self._load_annotations()

        item = self._annotations[idx]

        return {
            "ytid": item[0],
            "start_seconds": float(item[1].strip()),
            "end_seconds": float(item[2].strip()),
            "classes": [self.classes.index(k) for k in item[3].split(",")],
        }

    def __next__(self):
        """Iterate over the annotations, one by one."""
        if not self._annotations:
            self._load_annotations()
        if self._idx is None:
            self._idx = 0

        yield self[self._idx]
        self._idx += 1

    def clear_non_existing(
        self, audioset_path: Path, datapath_map: Dict[str, Path]
    ) -> "AudiosetAnnotationReaderV2":
        """Remove annotations for files which do not exist.

        Returns a new copy of the dataset.
        """
        if not self._annotations:
            self._load_annotations()

        copy = self.__class__(self.annotation_path, self.classes)
        copy._annotations = [x for x in self._annotations if x[0] in datapath_map]

        return copy


def _load_audio(
    audio_path: Path, start_seconds: float, end_seconds: float
) -> Tuple[np.ndarray, float]:
    duration = end_seconds - start_seconds

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        return librosa.load(audio_path, offset=start_seconds, duration=duration)


class AudiosetDataset(torch.utils.data.Dataset):
    """Torch-style dataset class for loading AudioSet."""

    SR = 44100

    def __init__(
        self,
        audioset_path: Union[str, Path],
        ontology: Union[str, Path],
        audioset_annotations: Union[str, Path],
        transforms: Union[torch.nn.Module, torchvision.transforms.Compose] = None,
        audio_cache_dir: Union[str, Path] = None,
    ):
        self.transforms = transforms
        with open(ontology, "r") as f:
            self.ontology = json.load(f)

        self.c = len(self.ontology)
        self.classes = [x["id"] for x in sorted(self.ontology, key=lambda x: x["id"])]
        self.c2l = {
            x["id"]: x["name"] for x in sorted(self.ontology, key=lambda x: x["id"])
        }
        self.l2c = {v: k for k, v in self.c2l.items()}

        audioset_path = Path(audioset_path)
        self.datapaths = {x.stem: x for x in audioset_path.glob("*[m4a|webm]")}

        self.annotations = AudiosetAnnotationReaderV2(
            audioset_annotations, classes=self.classes
        ).clear_non_existing(audioset_path, self.datapaths)

        self.audio_cache_dir = Path(audio_cache_dir) if audio_cache_dir else None
        self._init_cache()

        self._init_duration_log()

    def _init_cache(self):
        self._cache = None
        if self.audio_cache_dir is not None:
            self._cache = FeatureCache(self.audio_cache_dir)

    def _init_duration_log(self):
        self._load_durations = []

    def load_audio(
        self, ytid: str, start_seconds: float, end_seconds: float
    ) -> Tuple[np.ndarray, float]:
        """Load the audio, optionally with caching."""
        t0 = time.time()

        try:
            audio_path = self.datapaths[ytid]
        except KeyError:
            logger.error(f"**** Missing key: {ytid} ***")
            raise

        if self._cache is not None:
            audio, sr = self._cache.load(
                _load_audio, audio_path, start_seconds, end_seconds
            )

        else:
            audio, sr = _load_audio(audio_path, start_seconds, end_seconds)

        self._load_durations.append(time.time() - t0)

        return audio, sr

    def __getitem__(self, idx: int):
        """Get a sample from the dataset."""
        sample = self.annotations[idx]

        audio, sr = self.load_audio(
            sample["ytid"], sample["start_seconds"], sample["end_seconds"]
        )

        max_length = sr * 10
        temp = torch.zeros(max_length).unsqueeze(0)
        temp[0, : min(audio.shape[0], max_length)] = torch.tensor(audio)

        audio = temp
        data = deepcopy(audio)
        if self.transforms:
            data = self.transforms(data)

        labels = torch.zeros(self.c)
        for c in sample["classes"]:
            labels[c] = 1
        return {
            "raw": audio,
            "X": data,
            "label": labels,
            "class": sample["classes"],
            "class_name": [self.c2l[self.classes[x]] for x in sample["classes"]],
        }

    def __len__(self):
        """Size of the dataset, in annotations."""
        return len(self.annotations)

    def sampling_report(self):
        """Generate a report as a string."""
        last_5_durations = np.mean(self._load_durations[-5:])

        cache_report = None
        if self._cache:
            cache_report = (
                f"Cache [hits={self._cache._stats['cache_hit']}|"
                f"misses={self._cache._stats['cache_miss']}]"
            )
        return (
            f"Dataset time log (n=5): {last_5_durations} " + cache_report
            if cache_report
            else ""
        )


@pescador.streamable
def _gen_frames(
    audioset_dataset: AudiosetDataset,
    annotation_idx: int,
    n_frames: int,
    n_target_frames: int,
) -> Iterable[dict]:
    """Given an annotation index, load the audio file and generate frames from it."""
    sample_data = audioset_dataset[annotation_idx]
    # sample['X'] is the mel spectrum
    _, n_features, n_available_frames = sample_data["X"].shape

    # number of frames available
    frame_index = np.arange(n_available_frames - (n_frames + n_target_frames))

    while True:
        np.random.shuffle(frame_index)

        for i in frame_index:
            sample_frames = sample_data.copy()
            del sample_frames["raw"]
            sample_frames["X"] = sample_data["X"][:, :, i : i + n_frames]
            sample_frames["Y"] = sample_data["X"][
                :, :, i + n_frames : i + n_frames + n_target_frames
            ]
            yield sample_frames


class IterableAudiosetDataset(torch.utils.data.IterableDataset):
    """Iterable-style Torch dataset for sampling from AudioSet data."""

    STREAMER_DEFAULTS = {"n_frames": 10, "n_target_frames": 1}

    def __init__(
        self,
        audioset_path: Union[str, Path],
        ontology: Union[str, Path],
        audioset_annotations: Union[str, Path],
        streamer_settings: Optional[dict] = None,
        evaluate: bool = False,
        **audioset_kwargs,
    ):
        self.audioset_dataset = AudiosetDataset(
            audioset_path, ontology, audioset_annotations, **audioset_kwargs
        )

        self.streamer_settings = streamer_settings
        self.evaluate = evaluate

        self.start = 0
        self.end = len(self.audioset_dataset)

    def __len__(self):
        """If evaluate is true, return the length of the validation set.

        Otherwise, length is not defined.
        """
        if self.evaluate:
            return len(self.audioset_dataset)

        else:
            return float("inf")

    def _build_streamer(self, start_index: int, end_index: int) -> pescador.Streamer:
        """Create a pescador streamer for the provided indecies into the dataset."""
        audiofile_streamers = [
            _gen_frames(
                self.audioset_dataset,
                index,
                self.streamer_settings["n_frames"],
                self.streamer_settings["n_target_frames"],
            )
            for index in range(start_index, end_index)
        ]

        if self.evaluate:
            audiofile_mux = pescador.RoundRobinMux(audiofile_streamers)
        else:
            audiofile_mux = pescador.StochasticMux(
                audiofile_streamers,
                # todo: eventually, this should probably be a function of
                #   <batch size> & <# workers>
                # should probably be (batch_size / num_workers)
                n_active=6,
                # on average how many samples are generated from a stream before it dies
                rate=5,
            )

        return audiofile_mux

    def __iter__(self):
        """Iterate over the dataset, or the fraction provided to this worker."""
        # Worker handling taken from https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset # noqa E501
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process: return the full iterator.
            iter_start = self.start
            iter_end = self.end

        else:  # in a worker process.
            # split workload
            per_worker = int(
                math.ceil((self.end - self.start) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id

            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        if self.evaluate:
            return self._build_streamer(iter_start, iter_end).iterate(
                len(self.audioset_dataset)
            )
        else:
            return self._build_streamer(iter_start, iter_end).cycle()

    def sampling_report(self):
        """Placeholder, #TODO."""
        return ""
