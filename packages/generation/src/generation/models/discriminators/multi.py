from functools import partial
from typing import Any

from torch import Tensor, nn
from traincore.config_stores.models import model_store

from generation.models.discriminators.period import PeriodDiscriminator
from generation.models.discriminators.protocol import (
    DiscriminatorProtocol,
    MultiDiscriminatorReturnType,
)
from generation.models.discriminators.scale import ScaleDiscriminator


@model_store(name="multidiscriminator", group="model/discriminator")
@model_store(
    name="multiperiod",
    group="model/discriminator",
    discriminator=PeriodDiscriminator,
    configs={
        "period": [2, 3, 5, 7, 11],
    },
)
@model_store(
    name="multiscale",
    group="model/discriminator",
    discriminator=ScaleDiscriminator,
    configs={
        "downsampling_factor": 4,
        "num_prefix_downsamples": [0, 1, 2],
        "num_filters": 16,
        "n_layers": 4,
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
        shared_value_keys = [
            k for k in self.configs if isinstance(self.configs[k], (int, float))
        ]

        # a key with a list of values determins how many discriminators you get.
        # any lists have to be the same length.
        list_value_keys = [k for k in self.configs if isinstance(self.configs[k], list)]
        if list_value_keys:
            n_discriminators = len(self.configs[list_value_keys[0]])
            if not all(
                [n_discriminators == len(self.configs[k]) for k in list_value_keys]
            ):
                raise ValueError("All list values must have the same length.")
        else:
            n_discriminators = 1

        config_template = {k: self.configs[k] for k in shared_value_keys}

        list_of_configs = []
        for i in range(n_discriminators):
            c = config_template.copy()
            for k in list_value_keys:
                c[k] = self.configs[k][i]
            list_of_configs.append(c)

        self.discriminators = nn.ModuleList(
            [discriminator(**config) for config in list_of_configs]
        )

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
