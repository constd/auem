from typing import Mapping

from jaxtyping import Float
from torch import Tensor, nn
from traincore.config_stores.models import model_store

from generation.models.discriminators.protocol import CombinerDiscriminatorReturnType


@model_store(name="combiner", group="model/discriminator")
class CombinerDiscriminator(nn.Module):
    """A wrapper to combine multiple discriminators into a single module."""

    def __init__(self, discriminators: Mapping[str, nn.Module]):
        super().__init__()
        self.discriminators = nn.ModuleDict(discriminators)

    def forward(
        self, real: Float[Tensor, "..."], generated: Float[Tensor, "..."]
    ) -> CombinerDiscriminatorReturnType:
        output = {
            "estimates_real": [],
            "estimates_generated": [],
            "feature_maps_real": [],
            "feature_maps_generated": [],
        }
        for x, input_type in zip([real, generated], ["real", "generated"]):
            for discriminator_name, discriminator in self.discriminators.items():
                current_discriminator_output = discriminator(x)
                output[f"estimates_{input_type}"].extend(
                    current_discriminator_output["estimates"]
                )
                output[f"feature_maps_{input_type}"].extend(
                    current_discriminator_output["feature_maps"]
                )

        return output
