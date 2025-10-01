from hydra_zen import ZenStore, builds
from hydra_zen.third_party.beartype import (
    validates_with_beartype,
)
from torch import nn

__all__ = ["criterion_store"]

criterion_store: ZenStore = ZenStore()(
    group="criterion",
    populate_full_signature=True,
    hydra_convert="all",
    zen_wrappers=validates_with_beartype,
)

available_criterions = [
    "AdaptiveLogSoftmaxWithLoss",
    "L1Loss",
    "MSELoss",
    "CrossEntropyLoss",
    "CTCLoss",
    "NLLLoss",
    "PoissonNLLLoss",
    "GaussianNLLLoss",
    "KLDivLoss",
    "BCELoss",
    "BCEWithLogitsLoss",
    "MarginRankingLoss",
    "HingeEmbeddingLoss",
    "MultiLabelMarginLoss",
    "HuberLoss",
    "SmoothL1Loss",
    "SoftMarginLoss",
    "MultiLabelSoftMarginLoss",
    "CosineEmbeddingLoss",
    "MultiMarginLoss",
    "TripletMarginLoss",
    "TripletMarginWithDistanceLoss",
]

from traincore.criterions import *  # noqa

for criterion in available_criterions:
    criterion_store(
        builds(
            getattr(nn, criterion),
            populate_full_signature=True,
            zen_partial=True,
        ),
        name=criterion,
    )
