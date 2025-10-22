from functools import partial
from typing import Any

import torch
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.nn import Module, ModuleDict
from torch.optim import Optimizer
from traincore.config_stores.recipes import recipe_store


@recipe_store(name="gan")
class GanTrainRecipe(LightningModule):
    def __init__(
        self,
        model: Module | ModuleDict,
        loss: dict[str, Module],
        optimizer: dict[str, Optimizer | partial],
        scheduler: dict[str, Any] | None = {},
        ema: None | Module | partial = None,
        metrics: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.model = model
        self.loss = ModuleDict(loss)
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
            # get current batch
            augmented_mix = dataset["mix_augmented"]
            clean_mix = dataset["target"]

            generated_mix = self.model.generator(augmented_mix)
            generated_mix = generated_mix[..., : clean_mix.shape[-1]]  # ty: ignore[possibly-missing-attribute]

            # discriminator
            discriminator_output: dict[str, list[Tensor]] = self.model.discriminator(
                clean_mix, generated_mix
            )

            discriminator_loss = self.loss.discriminator(discriminator_output)

            discriminator_optimizer.zero_grad()
            self.manual_backward(discriminator_loss["loss"], retain_graph=True)
            discriminator_optimizer.step()

            # generator
            discriminator_output = self.model.discriminator(clean_mix, generated_mix)
            generator_loss = self.loss.generator(discriminator_output)

            feature_matching_loss = self.loss.feature(discriminator_output)

            reconstruction_loss = self.loss.reconstruction(
                generated_mix[..., : clean_mix.shape[-1]],  # ty: ignore[possibly-missing-attribute]
                clean_mix,
            )

            total_loss = (
                generator_loss["loss"] + feature_matching_loss + reconstruction_loss
            )

            generator_optimizer.zero_grad()
            self.manual_backward(total_loss)
            generator_optimizer.step()

            total_generator_loss += generator_loss["loss"].item()
            total_discriminator_loss += discriminator_loss["loss"].item()

            self.log(
                f"train/loss/generator/{dataset_name}", generator_loss["loss"].item()
            )
            self.log(
                f"train/loss/discriminator/{dataset_name}",
                discriminator_loss["loss"].item(),
            )
            self.log(
                f"train/loss/reconstruction/{dataset_name}", reconstruction_loss.item()
            )
            self.log(
                f"train/loss/feature_matching/{dataset_name}",
                reconstruction_loss.item(),
            )

        self.log("train/loss/generator/total", total_generator_loss)
        self.log("train/loss/discriminator/total", total_discriminator_loss.item())

        # # Step schedulers
        # scheduler_gen, scheduler_dis = self.lr_schedulers()
        # if scheduler_gen:
        #     scheduler_gen.step()
        # if scheduler_dis:
        #     scheduler_dis.step()

    def validation_step(
        self,
        batch: dict[str, dict[str, str | Tensor] | Tensor],
        batch_idx: int | None = None,
        dataloader_idx: int | None = None,
    ):
        return {"loss": 1.0}

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
