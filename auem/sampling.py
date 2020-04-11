"""Script to test and benchmark the Dataset sampling used in `train.py.`
"""
import logging
import time
from collections import defaultdict

import hydra
import numpy as np
from omegaconf import DictConfig

logger = logging.getLogger("auem.sampling")


def benchmark_sampling(cfg: DictConfig) -> None:
    """Benchmark the Dataset configured in the config."""
    transforms = hydra.utils.instantiate(cfg.transform)

    ds_train = hydra.utils.get_class(cfg.dataset["class"])(
        audioset_annotations=cfg.dataset["folds"]["train"],
        transforms=transforms,
        **cfg.dataset.params
    )

    dl_train = hydra.utils.get_class(cfg.dataloader["class"])(
        ds_train, **cfg.dataloader.params
    )

    time_log = defaultdict(list)

    for epoch_idx in range(3):
        time_log["epoch_start"].append(time.time())

        i = 0
        batch_len = len(dl_train)
        batch_iter = iter(dl_train)
        for i in range(batch_len):
            time_log["batch_start"].append(time.time())
            try:
                next(batch_iter)
            except KeyError:
                print("Batch failed, but continuing anyway")
            time_log["batch_end"].append(time.time())

            durations = np.array(time_log["batch_end"][-5:]) - np.array(
                time_log["batch_start"][-5:]
            )
            print("Rolling Batch Load Avg (n=5):", durations.mean())
        time_log["epoch_end"].append(time.time())
        print()
        print(
            "Epoch Load duraiton:",
            time_log["epoch_end"][-1] - time_log["epoch_start"][-1],
        )


@hydra.main(config_path="config/config.yaml")
def main(cfg: DictConfig) -> None:
    benchmark_sampling(cfg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
