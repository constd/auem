"""Test resnet models."""

# noqa: D400, D103
import pytest
from omegaconf import DictConfig

from traincore import models
from traincore.models import __all__ as all_registered_models


@pytest.fixture(params=all_registered_models)
def model_config(request):
    """This returns.

    Args:
        request (_type_): _description_

    Returns:
        _type_: _description_
    """
    return getattr(models, request.param)


def test_metrics_should_instantiate_and_follow_protocol(model_config: DictConfig):
    model = model_config()

    assert isinstance(model, models.protocol.AuemModelProtocol)
