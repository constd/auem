"""Test if all exported recipes can be instantiated and follow the protocol."""

import pytest
from torch import nn, optim

from traincore import recipes
from traincore.data.sets.random import RandomAudioWithClassifierDataset
from traincore.recipes import SimpleRecipe
from traincore.recipes import __all__ as all_registered_recipes
from traincore.recipes.protocol import AuemRecipeProtocol


@pytest.fixture(params=all_registered_recipes)
def recipe_cls(request) -> type[AuemRecipeProtocol]:
    """Fixture that yields each registered model class for parametrized testing.

    Args:
        request: Pytest request object containing the current parameter value.

    Returns:
        type: A model class from the traincore.recipes module.
    """

    return getattr(recipes, request.param)


def test_recipes_should_instantiate_and_follow_protocol(
    recipe_cls: type[AuemRecipeProtocol],
):
    # It doesn't matter what these are
    model = nn.Linear(1000, 2)
    loss = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.001)

    recipe = recipe_cls(model, loss, opt)

    assert isinstance(recipe, AuemRecipeProtocol)


def test_train_and_val_steps_should_return_loss():
    ds = RandomAudioWithClassifierDataset(5, 1000, n_channels=1, num_classes=2)
    ds.setup()
    item = ds[0]
    # add a batch dimension
    item["audio"] = item["audio"].unsqueeze(0).repeat(10, 1, 1)  # ty: ignore[possibly-unbound-attribute]

    item["class"] = item["class"].unsqueeze(0).float().repeat(10, 1)  # ty: ignore[possibly-unbound-attribute]

    class SqueezeLayer(nn.Module):
        def forward(self, x):
            return x.squeeze(1)

    model = nn.Sequential(nn.Linear(1000, 2), SqueezeLayer())
    loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    recipe = SimpleRecipe(model, loss, optimizer)

    loss = recipe.training_step({"random": item}, 0)
    assert isinstance(loss, dict)

    val_loss = recipe.validation_step(item, 0)
    assert isinstance(val_loss, dict)
