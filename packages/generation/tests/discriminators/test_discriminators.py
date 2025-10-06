"""Test if all exported models are instantiated and follow the protocol."""

import pytest
from generation.models import discriminators
from generation.models.discriminators import __all__ as all_registered_discriminators
from generation.models.discriminators.protocol import DiscriminatorProtocol


@pytest.fixture(params=all_registered_discriminators)
def discriminator_cls(request) -> type[DiscriminatorProtocol]:
    """Fixture that yields each registered model class for parametrized testing.

    Args:
        request: Pytest request object containing the current parameter value.

    Returns:
        type: A model class from the generation.models module.
    """

    return getattr(discriminators, request.param)


def test_metrics_should_instantiate_and_follow_protocol(
    discriminator_cls: type[DiscriminatorProtocol],
):
    discriminator = discriminator_cls()

    assert isinstance(discriminator, DiscriminatorProtocol)
