from hydra_zen import ZenStore

__all__ = ["model_store"]

model_store: ZenStore = ZenStore()(
    group="model",
    populate_full_signature=True,
    hydra_convert="all",
)

from traincore.models import *  # noqa

model_store.add_to_hydra_store()
