"""Test if all exported generators are instantiated and follow the generator protocol."""

import pytest

from generation.models import generators
from generation.models.generators import __all__ as all_registered_generators
from generation.models.generators.protocol import AuemGeneratorProtocol

# from traincore import models
# from traincore.models import __all__ as all_registered_models


@pytest.fixture(params=all_registered_generators)
def generator_cls(request) -> type[AuemGeneratorProtocol]:
    """Fixture that yields each registered model class for parametrized testing.

    Args:
        request: Pytest request object containing the current parameter value.

    Returns:
        type: A model class from the traincore.models module.
    """

    return getattr(generators, request.param)


def test_metrics_should_instantiate_and_follow_protocol(
    generator_cls: type[AuemGeneratorProtocol],
):
    model = generator_cls()

    assert isinstance(model, AuemGeneratorProtocol)
