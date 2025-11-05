# given a config, set up all the necessary parts, and run training
from hydra_zen import instantiate
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig
import os
import random
from auem.configs.mainconfig import train_store  # noqa


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    from lightning.pytorch import seed_everything as LSE

    LSE(seed, workers=True)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train(config: DictConfig) -> None:
    seed_everything(config.seed + int(os.getenv("LOCAL_RANK", "0")) * int(config.data.num_workers))
    
    callbacks: list[Callback] | None = None
    if config.get("callbacks", None):
        callbacks = [instantiate(v) for k, v in config.callbacks.items()]

    data: LightningDataModule = instantiate(config.data)

    recipe: LightningModule = instantiate(config.recipe)

    trainer: Trainer = instantiate(config.trainer, callbacks=callbacks)

    trainer.fit(recipe, data)


if __name__ == "__main__":
    # the following becomes the entrypoint to training
    # this allows us to decouple hydra from train and
    # still enable cli overrides when running the script directly
    # https://stackoverflow.com/questions/60674012/how-to-get-a-hydra-config-without-using-hydra-main
    import hydra
    from omegaconf import OmegaConf

    @hydra.main(config_path="configs", config_name="train_config")
    def run_train_with_config(cfg: DictConfig):
        # If you need resolving, it needs to be done here
        # TODO: is this the place where we add more things to the config?
        OmegaConf.resolve(cfg)

        train(cfg)

    run_train_with_config()
