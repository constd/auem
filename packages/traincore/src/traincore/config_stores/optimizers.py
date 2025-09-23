from hydra_zen import ZenStore, builds
from hydra_zen.third_party.beartype import (
    validates_with_beartype,
)
from torch import optim

__all__ = ["optimizer_store"]

optimizer_store: ZenStore = ZenStore(name="optimizer")(
    group="optimizer",
    populate_full_signature=True,
    zen_wrappers=validates_with_beartype,
    hydra_convert="all",
)

available_optimizers = [
    "Adam",
    "AdamW",
    "Adafactor",
    "Adadelta",
    "Adagrad",
    "Adam",
    "Adamax",
    "AdamW",
    "ASGD",
    "LBFGS",
    "NAdam",
    "RAdam",
    "RMSprop",
    "Rprop",
    "SGD",
    "SparseAdam",
]


for optimizer in available_optimizers:
    optimizer_store(
        builds(
            getattr(optim, optimizer), populate_full_signature=True, zen_partial=True
        )
    )

optimizer_store.add_to_hydra_store()
