import datetime
import itertools
import pathlib
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter


def plot_confusion_matrix(
    *,  # prevent positional arguments
    y_true: np.ndarray = None,
    y_pred: np.ndarray = None,
    cm: np.ndarray = None,
    class_names: Optional[str] = None,
    figsize=(8, 8)
):
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
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


def log_confusion_matrix(
    summary_writer: SummaryWriter,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    **kwargs
) -> None:
    """Compute the image of a confusion matrix, and save it to disk for tensorboard

    Use as a callback.
    """
    figure = plot_confusion_matrix(y_true, y_pred, class_names)
    summary_writer.add_figure("cm", figure, **kwargs)
