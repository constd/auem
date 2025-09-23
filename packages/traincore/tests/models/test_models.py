"""Test if all exported models are instantiated and follow the protocol."""

# noqa: D400, D103
import pytest

from traincore import models
from traincore.models import __all__ as all_registered_models
from traincore.models.protocol import AuemModelProtocol


@pytest.fixture(params=all_registered_models)
def model_cls(request) -> AuemModelProtocol:
    """Fixture that yields each registered model class for parametrized testing.

    Args:
        request: Pytest request object containing the current parameter value.

    Returns:
        type: A model class from the traincore.models module.
    """

    return getattr(models, request.param)


def test_metrics_should_instantiate_and_follow_protocol(model_cls: AuemModelProtocol):
    model = model_cls()

    assert isinstance(model, AuemModelProtocol)
