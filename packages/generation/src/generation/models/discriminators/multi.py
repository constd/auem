from functools import partial
from typing import Any

from torch import Tensor, nn

from generation.models.discriminators.protocol import (
    DiscriminatorProtocol,
    MultiDiscriminatorReturnType,
)
from generation.models.discriminators.period import PeriodDiscriminator
from traincore.config_stores.models import model_store


@model_store(name="multidiscriminator", group="model/discriminator")
@model_store(
    name="multiperiod",
    group="model/discriminator",
    discriminator=PeriodDiscriminator,
    configs={
        "period": [2, 3, 5, 7, 11],
    },
)
class MultiDiscriminator(nn.Module):
    """
    A multi-discriminator module that applies multiple discriminators in parallel.

    This class manages a collection of discriminators and applies them to both real and generated data,
    collecting their outputs and feature maps for training purposes (typically in adversarial setups).

    Args:
        discriminator: Either a non-instantiated discriminator class that follows the DiscriminatorProtocol
                      or a partially instantiated class created with functools.partial
        configs: A list of configuration dictionaries, where each dictionary contains the parameters
                needed to instantiate one discriminator instance

    Example:
        >>> from functools import partial
        >>> # Using a non-instantiated class
        >>> multi_disc = MultiDiscriminator(SomeDiscriminator, [{'param1': 'value1'}, {'param1': 'value2'}])
        >>>
        >>> # Using a partially instantiated class
        >>> partial_disc = partial(SomeDiscriminator, shared_param='shared_value')
        >>> multi_disc = MultiDiscriminator(partial_disc, [{'param1': 'value1'}, {'param1': 'value2'}])
    """

    def __init__(
        self,
        # either get a non instantiated class or a half-instantiated class
        discriminator: type[DiscriminatorProtocol] | partial,
        configs: dict[str, Any],
    ):
        super().__init__()
        self.configs = configs
        list_of_configs = [dict(zip(configs, t)) for t in zip(*configs.values())]
        self.discriminators = nn.ModuleList([
            discriminator(**config) for config in list_of_configs
        ])

    def forward(self, y: Tensor, *args, **kwargs) -> MultiDiscriminatorReturnType:
        output: MultiDiscriminatorReturnType = {
            "estimates": [],
            "feature_maps": [],
        }

        for i, d in enumerate(self.discriminators):
            current_output = d(y)
            output["estimates"].append(current_output["estimate"])
            output["feature_maps"].append(current_output["feature_map"])

        return output
