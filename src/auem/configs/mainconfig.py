from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from traincore.config_stores.criterions import criterion_store
from traincore.config_stores.datamodules import datamodule_store
from traincore.config_stores.datasets import dataset_store
from traincore.config_stores.loggers import logger_store
from traincore.config_stores.model_decoders import model_decoders_store
from traincore.config_stores.model_encoders import model_encoders_store
from traincore.config_stores.models import model_store
from traincore.config_stores.optimizers import optimizer_store
from traincore.config_stores.recipes import recipe_store
from traincore.config_stores.schedulers import scheduler_store
from traincore.config_stores.trainers import trainer_store

from generation.recipe.gan_recipe import *  # noqa
from generation.models import *  # noqa
from generation.models.generators import *  # noqa
from generation.models.discriminators import *  # noqa
from generation.losses import *  # noqa

dataset_store.add_to_hydra_store()
datamodule_store.add_to_hydra_store()
logger_store.add_to_hydra_store()
model_decoders_store.add_to_hydra_store()
model_encoders_store.add_to_hydra_store()
model_store.add_to_hydra_store()
optimizer_store.add_to_hydra_store()
scheduler_store.add_to_hydra_store()
trainer_store.add_to_hydra_store()
criterion_store.add_to_hydra_store()
recipe_store.add_to_hydra_store()


@dataclass
class MainConfigStore:
    name: str = "here is a name for the training run"
    # run_id: str = II("hash_my_config:${},${},${}")
    data: Any = MISSING
    model: Any = MISSING
    recipe: Any = MISSING
    trainer: Any = MISSING
    logger: Any = MISSING


train_store = ConfigStore.instance()

train_store.store(name="train_config", node=MainConfigStore)
