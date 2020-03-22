import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from torch.utils.data import DataLoader

import logging

logger = logging.getLogger(__name__)


def train(cfg: DictConfig) -> None:
    # transforms = hydra.utils.instantiate(cfg.transforms)
    dataset = hydra.utils.instantiate(cfg.dataset)

    # hydra doesn't work with non primitives like the dataset class
    # TODO: file a bug with hydra to allow non-promitive pass-through of non-primitives
    dataloader = DataLoader(dataset, **cfg.dataloader.params)
    # dataloader = hydra.utils.instantiate(cfg.dataloader, dataset=dataset)
    
    model = hydra.utils.instantiate(cfg.model)

    # hydra doesn't work with non primitives like the model.parameters() generator in the following
    # TODO: file a bug with hydra to allow non-promitive pass-through of non-primitives
    optimizer = hydra.utils.get_class(cfg.optim["class"])(model.parameters(), **cfg.optim.params)

    criterion = hydra.utils.instantiate(cfg.criterion)

    for epoch in tqdm(range(cfg.epochs)):
        for batch in tqdm(dataloader, total=len(dataset)):
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out)
            loss.backward()
            optimizer.step()

    logger.debug(f"in train loop {cfg.intrain}")


@hydra.main(config_path="config/config.yaml")
def main(cfg: DictConfig) -> None:
    logger.info(f"{cfg.inmain}")
    train(cfg)


if __name__ == "__main__":
    main()
