from functools import partial
from typing import Any

from lightning.pytorch import LightningModule
import torch
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
        batch: dict[str, dict[str, str | Tensor] | Tensor],
        batch_idx: int | None = None,
        dataloader_idx: int | None = None,
    ):
        generator_optimizer, discriminator_optimizer = self.optimizers()

        total_generator_loss = torch.tensor(0.0, device=self.device)
        total_discriminator_loss = torch.tensor(0.0, device=self.device)

        for dataset_name, dataset in batch.items():
            augmented_mix = dataset["mix_augmented"]
            clean_mix = dataset["mix"]

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

            total_generator_loss += generator_loss
            total_discriminator_loss += discriminator_loss

            self.log(f"train/generator_loss/{dataset_name}", generator_loss.item())
            self.log(
                f"train/discriminator_loss/{dataset_name}", discriminator_loss.item()
            )

        self.log("train/generator_loss/total", total_generator_loss)
        self.log("train/discriminator_loss/total", total_discriminator_loss.item())

        # return {"loss": total_generator_loss + total_discriminator_loss}

    def configure_optimizers(self):
        ret = []
        for m in ["generator", "discriminator"]:
            model_ = getattr(self.model, m)

            opt_lrsch = {}
            if isinstance(
                self.optimizer.get(m, None), (Optimizer, partial)
            ) and isinstance(model_, Module):
                optimizer = self.optimizer[m](model_.parameters())
                opt_lrsch["optimizer"] = optimizer
                if self.scheduler.get(m, None) is not None:
                    scheduler = self.scheduler[m].__dict__
                    scheduler["scheduler"] = scheduler["scheduler"](optimizer)
                    opt_lrsch["lr_scheduler"] = scheduler
                ret.append(opt_lrsch)
            else:
                raise ValueError(f"Invalid optimizer or model for {m}")
        return ret
