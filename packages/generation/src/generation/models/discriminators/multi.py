from functools import partial
from typing import Any

from torch import Tensor, nn

from generation.models.discriminators.protocol import (
    DiscriminatorProtocol,
    MultiDiscriminatorReturnType,
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
        configs: list[dict[str, Any]],
    ):
        super().__init__()
        self.configs = configs
        self.discriminators = nn.ModuleList([
            discriminator(**config) for config in self.configs
        ])

    def forward(self, y: Tensor, y_hat: Tensor) -> MultiDiscriminatorReturnType:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return {
            "disc_rs": y_d_rs,
            "disc_gs": y_d_gs,
            "fmap_rs": fmap_rs,
            "fmap_gs": fmap_gs,
        }
