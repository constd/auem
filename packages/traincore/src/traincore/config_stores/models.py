from hydra_zen import ZenStore
from hydra_zen.third_party.beartype import (
    validates_with_beartype,
)

__all__ = ["model_store"]

model_store: ZenStore = ZenStore()(
    group="model",
    populate_full_signature=True,
    zen_wrappers=validates_with_beartype,
    hydra_convert="all",
)

from traincore.models import *  # noqa
