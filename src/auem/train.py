# given a config, set up all the necessary parts, and run training
from hydra import main
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


@main(config_path="configs", config_name="train_config")
def get_config(config: DictConfig) -> DictConfig:
    return config


if __name__ == "__main__":
    # initialize(
    #     job_name="train",
    #     config_path="configs",
    #     version_base="1.3",
    # )
    # config = compose(config_name="train_config")
    config = get_config()
    train(config)
