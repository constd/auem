import json
import sys, os
import pickle
from csv import reader, DictReader
from tqdm import tqdm
import torch, torchaudio, torchvision
from copy import deepcopy
from glob import glob
from pathlib import Path
from typing import Union
import librosa


class AudiosetAnnotationReaderV1(DictReader):
    def __init__(self, f, ontology=None, fieldnames=None, fieldtypes=None,
                 restkey=None, restval=None, dialect="excel", *args, **kwds):
        self.ontology = self.__load_ontology(ontology)
        self.annotation_stats = None
        self.annotation_creation_date = None
        if fieldtypes is None:
            fieldtypes = {'YTID': str, "end_seconds": float, "start_seconds": float}
        self.fieldtypes = fieldtypes
        super().__init__(f, fieldnames=fieldnames, restkey=restkey, restval=restval,
                         dialect=dialect, *args, **kwds)
        self.init_properties()

    @staticmethod
    def __load_ontology(ontology):
        if ontology is not None:
            return {x["id"]: x for x in ontology}
        return ontology

    def init_properties(self):
        row = next(self.reader)
        self.annotation_creation_date = [x.strip("#").rstrip().lstrip() for x in row]
        row = next(self.reader)
        self.annotation_stats = {x.split("=")[0].strip("#|\' \'|\n"): int(x.split("=")[1]) for x in row}
        row = next(self.reader)
        row = [x.strip("#|\' \'|\n") for x in row]
        self.fieldnames = row[:-1]
        if self.restkey is None:
            self.restkey = row[-1]
        self.line_num = self.reader.line_num

    def __next__(self):
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
            classes = [x.strip("\' \'|\"") for x in row[lf:]]
            if self.ontology is not None:
                classes = {x: self.ontology[x] for x in set(classes)}
            d[self.restkey] = classes
        elif lf > lr:
            for key in self.fieldnames[lr:]:
                d[key] = self.restval
        return d


class AudiosetDataset(torch.utils.data.Dataset):
    SR = 44100
    def __init__(self, audioset_path: Union[str, Path],
                       ontology: Union[str, Path],
                       audioset_annotations: Union[str, Path],
                       transforms: Union[torch.nn.Module, torchvision.transforms.Compose]=None):
        self.transforms = transforms
        with open(ontology, 'r') as f:
            self.ontology = json.load(f)

        self.c = len(self.ontology)
        self.classes = [x["id"] for x in sorted(self.ontology, key=lambda x: x["id"])]
        self.c2l = {x["id"]: x["name"] for x in sorted(self.ontology, key=lambda x: x["id"])}
        self.l2c = {v: k for k, v in self.c2l.items()}
        

        datapaths = glob(audioset_path + "*[m4a|webm]")
        datapaths = {os.path.splitext(os.path.basename(x))[0]: x for x in datapaths}

        self.annotations = []
        with open(audioset_annotations, "r") as f:
            for row in AudiosetAnnotationReaderV1(f, ontology=self.ontology):
                if row["YTID"] in datapaths.keys():
                    self.annotations.append({
                        "ytid": row["YTID"],
                        "start_seconds": row["start_seconds"],
                        "end_seconds": row["end_seconds"],
                        "classes": [self.classes.index(k) for k in row["positive_labels"].keys()],
                        "filepath": datapaths[row["YTID"]]
                    })
    
    def __getitem__(self, idx: int):
        sample = self.annotations[idx]
        start, duration = sample["start_seconds"], sample["end_seconds"] - sample["start_seconds"]
        audio, sr = librosa.load(sample["filepath"], offset=start, duration=duration)
        

        max_length = sr*10
        temp = torch.zeros(sr*10).unsqueeze(0)
        temp[0, :min(audio.shape[0], max_length)] = torch.tensor(audio)

        # resample here so that all subsequent transforms can assume a SR
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.SR)(temp)
        data = deepcopy(audio)
        if self.transforms:
            data = self.transforms(data)

        labels = torch.zeros(self.c)
        for c in sample['classes']:
            labels[c] = 1
        return {
            "raw": audio,
            "X": data,
            "label": labels,
            "class": sample["classes"],
            "class_name": [self.c2l[self.classes[x]] for x in sample["classes"]]
        }

    def __len__(self):
        return len(self.annotations)
