"""Test if all exported models are instantiated and follow the protocol."""

import pytest

from traincore.data import modules
from traincore.data.modules import __all__ as all_registered_datamodules
from traincore.data.modules.protocol import DataModuleProtocol


@pytest.fixture(params=all_registered_datamodules)
def data_cls(request) -> type[DataModuleProtocol]:
    """Fixture that yields each registered model class for parametrized testing.

    Args:
        request: Pytest request object containing the current parameter value.

    Returns:
        type: A model class from the traincore.models module.
    """

    return getattr(modules, request.param)


def test_metrics_should_instantiate_and_follow_protocol(
    data_cls: type[DataModuleProtocol],
):
    dm = data_cls(datasets={})

    assert isinstance(dm, DataModuleProtocol)
