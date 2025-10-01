from hydra_zen import ZenStore
from hydra_zen.third_party.beartype import (
    validates_with_beartype,
)

__all__ = ["datamodule_store"]

datamodule_store: ZenStore = ZenStore()(
    group="datamodule",
    populate_full_signature=True,
    hydra_convert="all",
    zen_wrappers=validates_with_beartype,
)

from traincore.data.modules import *  # noqa
