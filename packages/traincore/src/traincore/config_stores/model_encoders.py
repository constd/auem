from hydra_zen import ZenStore

__all__ = ["model_encoders_store"]

model_encoders_store: ZenStore = ZenStore()(
    group="model/encoder",
    populate_full_signature=True,
    hydra_convert="all",
)

from traincore.models.encoders import *  # noqa

model_encoders_store.add_to_hydra_store()
