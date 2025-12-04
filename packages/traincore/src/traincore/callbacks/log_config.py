from pathlib import Path
from typing import TYPE_CHECKING

from comet_ml import (
    Experiment,
)
from lightning import (
    Callback,
    LightningModule,
    Trainer,
)
from lightning.pytorch.utilities import (
    rank_zero_only,
)
from omegaconf import (
    DictConfig,
    OmegaConf,
)

from traincore.config_stores.callbacks import (
    callback_store,
)

if TYPE_CHECKING:
    from lightning.pytorch.loggers import (
        Logger,
    )

__all__ = ["ConfigLogger"]


def resolve_configuration(
    config: DictConfig,
) -> DictConfig:
    """This function helps resolve all interpolations eagerly, so that we can see the final values of the entire config immediately.

    Args:
        config (DictConfig): the hydra configuration, usually the entire config of an experiment.

    Returns:
        DictConfig: the resolved configuration
    """
    OmegaConf.resolve(config)
    return config


def save_configuration(config: DictConfig, path: Path | str = ".") -> None:
    """This function helps resolve all interpolations eagerly, so that we can see the final values of the entire config immediately.

    Args:
        config (DictConfig): the hydra configuration, usually the entire config of an experiment.
        path (Path | str): the path to write the configuration to.

    Returns:
        DictConfig: the resolved configuration
    """
    config_resolved = resolve_configuration(config=config)
    OmegaConf.save(config=config_resolved, f=path)


@callback_store(name="config_log")
class ConfigLogger(Callback):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config

    @rank_zero_only
    def on_fit_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """This method attempts to resolve and log the experiment's configuration.

        This occurs right _before_ training actually starts.

        Args:
            trainer (Trainer): a reference to the current trainer.
            pl_module (LightningModule): an instance of the recipe.
        """
        logger: Logger | None

        # first, force-resolve the config and save it locally.

        config_path: str = self.config.paths.output_dir + "/config_resolved.yaml"
        save_configuration(self.config, config_path)

        logger = trainer.logger
        if logger is not None:
            experiment = logger.experiment  # ty: ignore[unresolved-attribute]
            if isinstance(experiment, Experiment):
                asset_name = config_path.split("/")[-1]
                _ = experiment.log_asset(
                    config_path,
                    file_name=asset_name,
                )
