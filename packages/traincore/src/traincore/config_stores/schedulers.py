from hydra_zen import ZenStore, builds
from hydra_zen.third_party.beartype import (
    validates_with_beartype,
)
from torch.optim import lr_scheduler

__all__ = ["scheduler_store"]

scheduler_store: ZenStore = ZenStore(name="scheduler")(
    group="scheduler",
    populate_full_signature=True,
    zen_wrappers=validates_with_beartype,
    hydra_convert="all",
)

available_schedulers = [
    "LambdaLR",
    "MultiplicativeLR",
    "StepLR",
    "MultiStepLR",
    "ConstantLR",
    "LinearLR",
    "ExponentialLR",
    "SequentialLR",
    "CosineAnnealingLR",
    "ChainedScheduler",
    "ReduceLROnPlateau",
    "CyclicLR",
    "CosineAnnealingWarmRestarts",
    "OneCycleLR",
    "PolynomialLR",
    "LRScheduler",
]


for scheduler in available_schedulers:
    scheduler_store(
        builds(
            getattr(lr_scheduler, scheduler),
            populate_full_signature=True,
            zen_partial=True,
        ),
        name=scheduler,
    )
