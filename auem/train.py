import logging
from math import ceil

import hydra
from omegaconf import DictConfig
from torch.cuda import is_available as is_cuda_available
from torch.utils.data import DataLoader

from tqdm import tqdm

logger = logging.getLogger(__name__)


def train(cfg: DictConfig) -> None:
    device = cfg.cuda.device if cfg.cuda.enable and is_cuda_available() else "cpu"

    # transforms = hydra.utils.instantiate(cfg.transforms)
    dataset = hydra.utils.instantiate(cfg.dataset)

    # hydra doesn't work with non primitives like the dataset class
    # TODO: file a bug with hydra to allow non-promitive pass-through of non-primitives
    dataloader = hydra.utils.get_class(cfg.dataloader["class"])(
        dataset, **cfg.dataloader.params
    )
    # dataloader = hydra.utils.instantiate(cfg.dataloader, **{"dataset": dataset})

    model = hydra.utils.instantiate(cfg.model).to(device)

    # hydra doesn't work with non primitives like the model.parameters() generator in the following
    # TODO: file a bug with hydra to allow non-promitive pass-through of non-primitives
    optimizer = hydra.utils.get_class(cfg.optim["class"])(
        model.parameters(), **cfg.optim.params
    )

    criterion = hydra.utils.instantiate(cfg.criterion)

    for epoch in tqdm(range(cfg.epochs), position=0, desc="Epoch"):
        for batch in tqdm(
            dataloader,
            total=ceil(len(dataset) / cfg.dataloader.params.batch_size),
            position=1,
            desc="Batch",
        ):
            X, y = batch["sample"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

    logger.debug(f"in train loop {cfg.intrain}")


@hydra.main(config_path="config/config.yaml")
def main(cfg: DictConfig) -> None:
    logger.info(f"{cfg.inmain}")
    train(cfg)


if __name__ == "__main__":
    main()
