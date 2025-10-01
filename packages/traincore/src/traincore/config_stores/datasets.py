from hydra_zen import ZenStore
from hydra_zen.third_party.beartype import (
    validates_with_beartype,
)

__all__ = ["dataset_store"]

dataset_store: ZenStore = ZenStore(
    name="dataset_store",
    deferred_to_config=True,
    deferred_hydra_store=True,
)(
    group="dataset",
    populate_full_signature=True,
    hydra_convert="all",
    zen_wrappers=validates_with_beartype,
)

from traincore.data.sets import *  # noqa
