import lightning as L
from hydra_zen import ZenStore, builds
from hydra_zen.third_party.beartype import (
    validates_with_beartype,
)
from lightning import Trainer  # noqa

__all__ = ["trainer_store"]

trainer_store: ZenStore = ZenStore()(
    group="trainer",
    populate_full_signature=True,
    zen_wrappers=validates_with_beartype,
    hydra_convert="all",
)


trainer_store(
    builds(
        L.Trainer,
        populate_full_signature=True,
        zen_partial=True,
    )
)
