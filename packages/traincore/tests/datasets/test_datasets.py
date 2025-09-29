"""Test if all exported models are instantiated and follow the protocol."""

import pytest

from traincore.data import sets
from traincore.data.sets import __all__ as all_registered_datasets
from traincore.data.sets.protocol import DatasetProtocol


@pytest.fixture(params=all_registered_datasets)
def data_cls(request) -> type[DatasetProtocol]:
    """Fixture that yields each registered model class for parametrized testing.

    Args:
        request: Pytest request object containing the current parameter value.

    Returns:
        type: A model class from the traincore.models module.
    """

    return getattr(sets, request.param)


def test_datasets_should_instantiate_and_follow_protocol(
    data_cls: type[DatasetProtocol],
):
    ds = data_cls()

    assert isinstance(ds, DatasetProtocol)
