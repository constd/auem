# given a config, set up all the necessary parts, and run training
from hydra import compose, initialize
from hydra_zen import instantiate
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from torch.nn import Module

from auem.configs.mainconfig import train_store  # noqa


def train(config: DictConfig) -> None:
    loggers: Logger | None = (
        instantiate(config.logger) if config.get("logger", None) else None
    )

    data: LightningDataModule = instantiate(config.data)

    model: Module = instantiate(config.model)

    recipe: LightningModule = instantiate(config.recipe, model=model)

    trainer: Trainer = instantiate(config.trainer, logger=loggers)

    trainer.fit(recipe, data)


if __name__ == "__main__":
    initialize(
        config_path="configs",
        job_name="train",
        version_base="1.3",
    )
    config = compose(config_name="train_config")
    train(config)
