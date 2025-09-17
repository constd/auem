from hydra_zen import ZenStore

__all__ = ["datamodule_store"]

datamodule_store: ZenStore = ZenStore()(
    group="datamodule",
    populate_full_signature=True,
    hydra_convert="all",
)

from traincore.data.datamodules import *  # noqa

datamodule_store.add_to_hydra_store()
