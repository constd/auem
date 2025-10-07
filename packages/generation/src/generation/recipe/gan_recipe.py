from functools import partial
from typing import Any

from lightning.pytorch import LightningModule
from torch import Tensor
from torch.nn import Module, ModuleDict
from torch.optim import Optimizer
from traincore.config_stores.recipes import recipe_store


@recipe_store(name="gan")
class GanTrainRecipe(LightningModule):
    def __init__(
        self,
        model: ModuleDict,
        loss: Module,
        optimizer: dict[str, Optimizer | partial],
        scheduler: dict[str, Any] | None = {},
        ema: None | Module | partial = None,
        metrics: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = ema
        self.metrics = metrics

        self.automatic_optimization = False

    def training_step(
        self,
        batch: dict[str, Any],
        batch_idx: int | None = None,
        dataloader_idx: int | None = None,
    ):
        augmented_mix = batch["mix_augmented"]
        clean_mix = batch["mix"]

        generator_optimizer, discriminator_optimizer = self.optimizers()

        generated_mix = self.model.generator(augmented_mix)

        discriminator_optimizer.zero_grad()

        discriminator_output: dict[list[Tensor], list[Tensor]] = (
            self.model.discriminator(clean_mix, generated_mix)
        )

        discriminator_loss = self.loss.discriminator(discriminator_output)

        discriminator_loss.backward(retain_graph=True)
        discriminator_optimizer.step()

        # generator
        generator_optimizer.zero_grad()

        discriminator_output = self.model.discriminator(clean_mix, generated_mix)
        generator_loss = self.loss.generator(
            discriminator_output, clean_mix, generated_mix
        )

        generator_loss.backward()
        generator_optimizer.step()

        self.log("train/generator_loss", generator_loss)
        self.log("train/discriminator_loss", discriminator_loss)

        return {"loss": generator_loss + discriminator_loss}

    def configure_optimizers(self):
        ret = []
        for m in ["generator", "discriminator"]:
            opt_lrsch = {}
            if isinstance(self.optimizer, dict) and isinstance(self.model, ModuleDict):
                optimizer = self.optimizer[m](self.model[m].parameters())
                opt_lrsch["optimizer"] = optimizer
                if self.scheduler.get(m, None) is not None:
                    scheduler = self.scheduler[m].__dict__
                    scheduler["scheduler"] = scheduler["scheduler"](optimizer)
                    opt_lrsch["lr_scheduler"] = scheduler
                ret.append(opt_lrsch)
        return ret
