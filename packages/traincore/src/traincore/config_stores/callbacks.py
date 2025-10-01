import lightning.pytorch.callbacks as lcall
from hydra_zen import ZenStore, builds
from hydra_zen.third_party.beartype import (
    validates_with_beartype,
)

__all__ = ["callback_store"]

callback_store: ZenStore = ZenStore()(
    group="callback",
    populate_full_signature=True,
    hydra_convert="all",
    zen_wrappers=validates_with_beartype,
)

from traincore.callbacks import *  # noqa

callback_store(
    builds(lcall.EarlyStopping), name="earlystop", populate_full_signature=True
)
callback_store(
    builds(lcall.ModelSummary), name="modelsummary", populate_full_signature=True
)
callback_store(builds(lcall.RichProgressBar), name="rich", populate_full_signature=True)
callback_store(
    builds(lcall.ModelCheckpoint, name="checkpoint", save_top_k=10, save_last=True),
    populate_full_signature=True,
)
