from hydra_zen import ZenStore

__all__ = ["dataset_store"]

dataset_store: ZenStore = ZenStore()(
    group="dataset",
    populate_full_signature=True,
    hydra_convert="all",
)

from traincore.data.datasets import *  # noqa

dataset_store.add_to_hydra_store()
