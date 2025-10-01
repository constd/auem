# r"""Splits a .csv dataset into one or more .csv subsets.

# Reads a .csv file, such as the audioset datasplits .csv files,
# and produces new .csv files in the same directory with lines randomly
# sampled from the csv file. Lines beginning with '#' are ignored.

# You may enter multiple subsets to be included, by passing a list of names
# and floats. The sum of all the named subsets must be <= 1.0

# If you pass in an integer > 1, it will select exactly that number of records (up to the
# maximum number)

# Example:
#     python auem/data/split.py \
#         /audioset/datasplits/balanced_train_segments.csv train=0.7 val=0.3

# This will split balanced_train_segments.csv into two new csv files,
#  - `balanced_train_segments-train.csv`, containing 70% of the original, sampled randomly
#  - `balanced_train_segments-val.csv`, containing 30% of the original, sampled randomly

# Example:
#     python auem/data/split.py \
#         /audioset/datasplits/balanced_train_segments.csv train-dev=5000

# This will create `balanced_train_segments-train-dev.csv` with 5000 records.

# """

# import csv
# import logging
# from pathlib import Path
# from typing import List

# import click
# import torch

# logger = logging.getLogger(__name__)


# @click.command()
# @click.argument("source_dataset_path", type=click.Path(exists=True))
# @click.argument("dest_datasets", nargs=-1)
# @click.option("--writedir")
# def split(source_dataset_path: Path | str, dest_datasets: List[str], writedir: str):
#     """Split a .csv dataset into one or more .csv subsets."""
#     source_dataset_path = Path(source_dataset_path)
#     writedir: Path = source_dataset_path.parent if not writedir else Path(writedir)

#     logger.info("Loading source_dataset_path")
#     source_dataset = []
#     with open(source_dataset_path, "r") as fh:
#         for line in csv.reader(fh, quotechar='"', delimiter=",", skipinitialspace=True):
#             if not line[0].startswith("#"):
#                 source_dataset.append(line)
#     source_len = len(source_dataset)
#     logger.info(f"Loading source_dataset_path: len={source_len}")

#     parsed_dest_datasets = [x.split("=") for x in dest_datasets]
#     # Conver the second value to a number
#     parsed_dest_datasets = [(k, float(v)) for k, v in parsed_dest_datasets]
#     logger.info(f"Parsed destinations: {parsed_dest_datasets}")

#     # All floats < 1
#     if all([x[1] < 1 for x in parsed_dest_datasets]):
#         dest_splits = [(k, int(source_len * v)) for k, v in parsed_dest_datasets]

#     # All numbers
#     elif all([x[1] > 1 for x in parsed_dest_datasets]):
#         dest_splits = [(k, int(v)) for k, v in parsed_dest_datasets]

#     else:
#         logger.error("Malformed input: splits must be all < 1 or > 1")

#     # Validate the dest_datasets
#     subset_def = [x[1] for x in dest_splits]
#     # random_split requires all your splits to match the dataset size
#     extra_records = len(source_dataset) - sum(subset_def)
#     subsets = torch.utils.data.random_split(
#         source_dataset, subset_def + [extra_records]
#     )

#     csv_stem = source_dataset_path.stem
#     for i, subset in enumerate(subsets[: len(subset_def)]):
#         subset_name = dest_splits[i][0]
#         subset_write_path = writedir / f"{csv_stem}-{subset_name}.csv"

#         with open(subset_write_path, "w") as fh:
#             writer = csv.writer(fh, quotechar='"', delimiter=",", skipinitialspace=True)

#             for i in range(len(subset)):
#                 writer.writerow(subset[i])
#         logger.info(f"Wrote {subset_write_path}, with {len(subset)} records")


# if __name__ == "__main__":
#     split()
