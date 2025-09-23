from typing import Literal

from lightning.pytorch.loggers import (
    CometLogger,
    CSVLogger,
    Logger,
    MLFlowLogger,
    NeptuneLogger,
    TensorBoardLogger,
    WandbLogger,
)

from traincore.callbacks.confusion import ConfusionCallback

__all__ = ["ConfusionCallback"]


def get_logger(
    logger: Logger | None,
) -> Literal["comet", "csv", "neptune", "wandb", "mlflow", "tensorboard"] | None:
    match logger:
        case WandbLogger():
            return "wandb"
        case TensorBoardLogger():
            return "tensorboard"
        case CometLogger():
            return "comet"
        case CSVLogger():
            return "csv"
        case NeptuneLogger():
            return "neptune"
        case MLFlowLogger():
            return "mlflow"
        case _:
            return None
