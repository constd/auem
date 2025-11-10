import pytest
import torch

from traincore.models.activations import Activation1d, Activation2d, Snake, SnakeBeta


def test_snake_activation():
    # Test basic functionality
    x = torch.randn(2, 3, 10)
    snake = Snake(in_features=3)
    output = snake(x)
    assert output.shape == x.shape


def test_snake_beta_activation():
    # Test basic functionality
    x = torch.randn(2, 3, 10)
    snake_beta = SnakeBeta(in_features=3)
    output = snake_beta(x)
    assert output.shape == x.shape


@pytest.mark.parametrize(
    "activation,kwargs",
    [
        (torch.nn.LeakyReLU, {}),
        (torch.nn.ReLU, {}),
        (Snake, {"in_features": 3}),
        (SnakeBeta, {"in_features": 3}),
    ],
)
def test_activation1d(activation, kwargs):
    # Test basic functionality
    x = torch.randn(2, 3, 10)
    act1d = Activation1d(activation(**kwargs))
    output = act1d(x)
    assert output.shape == x.shape


@pytest.mark.parametrize(
    "activation,kwargs",
    [
        (torch.nn.LeakyReLU, {}),
        (torch.nn.ReLU, {}),
        (Snake, {"in_features": 3}),
        (SnakeBeta, {"in_features": 3}),
    ],
)
def test_activation2d(activation, kwargs):
    # Test basic functionality
    x = torch.randn(2, 3, 10, 10)
    act2d = Activation2d(torch.nn.ReLU())
    output = act2d(x)
    assert output.shape == x.shape
