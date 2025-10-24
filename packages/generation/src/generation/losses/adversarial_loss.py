from torch import nn, Tensor
from jaxtyping import Float
import torch

from typing import overload
from traincore.config_stores.criterions import criterion_store


@criterion_store(name="hinge")
class HingeLoss(nn.Module):
    @overload
    def forward(self, x: Tensor, x_hat: Tensor) -> tuple[Tensor, Tensor]: ...
    @overload
    def forward(self, x: Tensor, x_hat: None) -> tuple[Tensor, None]: ...

    def forward(self, x, x_hat=None):
        loss_x = torch.mean(torch.clamp(1 - x, min=0))
        match x_hat:
            case None:
                loss_y = None
            case _:
                loss_y = torch.mean(torch.clamp(1 + x_hat, min=0))  # ty: ignore[no-matching-overload]

        return loss_x, loss_y


@criterion_store(name="lsgan")
class LSGANLoss(nn.Module):
    @overload
    def forward(self, x: Tensor, x_hat: Tensor) -> tuple[Tensor, Tensor]: ...
    @overload
    def forward(self, x: Tensor, x_hat: None) -> tuple[Tensor, None]: ...

    def forward(self, x, x_hat=None):
        loss_x = torch.mean((x - 1) ** 2)
        match x_hat:
            case Tensor():
                loss_y = torch.mean((x_hat) ** 2)
            case _:
                loss_y = None
        return loss_x, loss_y


@criterion_store(name="generator")
class GeneratorLoss(nn.Module):
    def __init__(self, loss: nn.Module, weight: float = 1.0):
        super().__init__()
        self.loss_fn = loss
        self.weight_ = weight

    def forward(
        self, dicsriminator_output
    ) -> dict[str, Tensor | list[Float[Tensor, "..."]]]:
        loss = torch.zeros(
            1,
            device=dicsriminator_output["estimates_generated"][0].device,
            dtype=dicsriminator_output["estimates_generated"][0].dtype,
        )
        losses = []
        for fmg in dicsriminator_output["estimates_generated"]:
            curr_loss, _ = self.loss_fn(fmg)

            losses.append(curr_loss)
            loss += curr_loss

        return {"loss": self.weight_ * loss, "losses_generator": losses}


@criterion_store(name="discriminator")
class DiscriminatorLoss(nn.Module):
    def __init__(self, loss: nn.Module, weight: float = 1.0):
        super().__init__()
        self.loss_fn = loss
        self.weight_ = weight

    def forward(
        self, dicsriminator_output
    ) -> dict[str, Tensor | list[Float[Tensor, "..."]]]:
        loss = torch.zeros(
            1,
            device=dicsriminator_output["estimates_generated"][0].device,
            dtype=dicsriminator_output["estimates_generated"][0].dtype,
        )
        losses_real = []
        losses_generated = []

        for fmr, fmg in zip(
            dicsriminator_output["estimates_real"],
            dicsriminator_output["estimates_generated"],
        ):
            loss_real, loss_generated = self.loss_fn(fmr, fmg)
            loss += loss_real + loss_generated
            losses_real.append(loss_real)
            losses_generated.append(loss_generated)

        return {
            "loss": self.weight_ * loss,
            "losses_discriminator_real": losses_real,
            "losses_discriminator_generated": losses_generated,
        }


@criterion_store(name="feature_matching")
class FeatureMatchingLoss(nn.Module):
    def __init__(self, scale: float = 10.0, weight: float = 1.0):
        """
        Paramters
        =========
        scale : float
            A multiplier which scales the feature matching loss to a
            similar range of the other losses. In melgan, this is 10.

        weight: float
            A configurable value to scale this loss against the other
            losses.
        """
        super().__init__()
        self.scale = scale
        self.weight_ = weight

    def forward(self, dicsriminator_output) -> Tensor:
        loss = torch.zeros(
            1,
            device=dicsriminator_output["feature_maps_real"][0][0].device,
            dtype=dicsriminator_output["feature_maps_real"][0][0].dtype,
        )
        for dr, dg in zip(
            dicsriminator_output["feature_maps_real"],
            dicsriminator_output["feature_maps_generated"],
        ):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return self.weight_ * self.scale * loss
