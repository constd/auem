from hydra_zen import ZenStore
from hydra_zen.third_party.beartype import (
    validates_with_beartype,
)

__all__ = ["dataset_store"]

dataset_store: ZenStore = ZenStore()(
    group="dataset",
    populate_full_signature=True,
    hydra_convert="all",
    zen_wrappers=validates_with_beartype,
)

from traincore.data.sets import *  # noqa

dataset_store.add_to_hydra_store()
