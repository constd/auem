# given a config, set up all the necessary parts, and run training
from hydra_zen import instantiate
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from torch.nn import Module


def train(config: DictConfig) -> None:
    loggers: Logger = instantiate(config.logger)

    data: LightningDataModule = instantiate(config.data)

    model: Module = instantiate(config.model)

    recipe: LightningModule = instantiate(config.recipe, model=model)

    trainer: Trainer = instantiate(config.trainer, logger=loggers)

    trainer.fit(recipe, data)
