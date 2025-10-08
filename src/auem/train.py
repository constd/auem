# given a config, set up all the necessary parts, and run training
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
    # the following becomes the entrypoint to training
    # this allows us to decouple hydra from train and
    # still enable cli overrides when running the script directly
    # https://stackoverflow.com/questions/60674012/how-to-get-a-hydra-config-without-using-hydra-main
    import hydra
    from omegaconf import OmegaConf

    @hydra.main(config_path="configs", config_name="train_config")
    def get_config(cfg: DictConfig) -> DictConfig:
        # If you need resolving, it needs to be done here
        # TODO: is this the place where we add more things to the config?
        OmegaConf.resolve(cfg)
        return cfg

    config = get_config()
    train(config)
