from hydra_zen import ZenStore
from hydra_zen.third_party.beartype import (
    validates_with_beartype,
)
from lightning.pytorch import loggers as llog  # noqa

__all__ = ["logger_store"]

logger_store: ZenStore = ZenStore()(
    group="logger",
    populate_full_signature=True,
    zen_wrappers=validates_with_beartype,
    hydra_convert="all",
)

# TBD: do we need this? these should be coming from other libraries
# from traincore.loggers import *  # noqa

logger_store(llog.CSVLogger, name="csv", populate_full_signature=True)
logger_store(llog.CometLogger, name="comet", populate_full_signature=True)
logger_store(llog.MLFlowLogger, name="mlflow", populate_full_signature=True)
logger_store(llog.TensorBoardLogger, name="tensorboard", populate_full_signature=True)
logger_store(llog.WandbLogger, name="wandb", populate_full_signature=True)
