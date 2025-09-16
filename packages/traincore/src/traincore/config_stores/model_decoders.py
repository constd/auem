from hydra_zen import ZenStore

__all__ = ["model_decoders_store"]

model_decoders_store: ZenStore = ZenStore()(
    group="model/decoder",
    populate_full_signature=True,
    hydra_convert="all",
)

from traincore.models.decoders import *  # noqa

model_decoders_store.add_to_hydra_store()
