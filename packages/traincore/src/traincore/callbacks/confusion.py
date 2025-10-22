import itertools
from typing import Any

import numpy as np
from jaxtyping import Float
from lightning import LightningModule
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only
from sklearn.metrics import confusion_matrix
from torch.utils.data.dataset import Dataset
from torch import Tensor
# from traincore.callbacks import get_logger

try:
    from matplotlib import pyplot as plt
    from PIL import Image
except ImportError:
    plt = None  # ty: ignore[invalid-assignment]
    Image = None  # ty: ignore[invalid-assignment]


# TODO: this entire class is almost pseudocode, it needs some work and _testing_
class ConfusionCallback(Callback):
    def __init__(self, dataset: Dataset[Any]) -> None:
        super().__init__()
        self.dataset = dataset

    @staticmethod
    def plot_confusion_matrix(
        y_true: Float[Tensor, "batch class"] | None = None,
        y_pred: Float[Tensor, "batch class"] | None = None,
        cm: Float[Tensor, "true pred"] | None = None,
        class_names: list[str] = [],
        figsize: tuple[float, float] = (8, 8),
    ) -> Image.Image:
        """Returns a matplotlib figure containing the plotted confusion matrix.

        Parameters
        ----------
        y_true (array, shape = [n,])

        y_pred (array, shape = [n,])

        class_names : list of str

        Returns
        -------
        plt.Figure
        """
        if cm is None:
            cm = confusion_matrix(y_true, y_pred, labels=class_names)
        elif not y_true and not y_pred:
            raise ValueError(
                "Must provide either (y_true and y_pred) or the confusion matrix directly by keyword arguments."
            )
        figure = plt.figure(figsize=figsize)
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)  # ty: ignore[unresolved-attribute]
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Normalize the confusion matrix.
        cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)  # ty: ignore

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

        figure.canvas.draw()

        return Image.frombytes(
            "RGB",
            figure.canvas.get_width_height(),
            figure.canvas.tostring_rgb(),  # ty: ignore[unresolved-attribute]
        )

    @rank_zero_only
    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        logger: Logger | None = trainer.logger
        if logger and plt and Image:
            # get predictions
            y_true, y_pred = [], []
            for datum in self.dataset:
                y_true.append(datum.true_label)
                y_pred.append(pl_module.model(datum))  # ty: ignore[call-non-callable]
            self.plot_confusion_matrix(y_true, y_pred)
            match logger:
                case "wandb":
                    pass
                case "comet":
                    pass
                case _:
                    pass
        else:
            pass
