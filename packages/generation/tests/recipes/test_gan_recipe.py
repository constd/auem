from functools import partial

import lightning as l
from generation.recipe.gan_recipe import GanTrainRecipe
from torch import Tensor, nn, optim
from torch.nn.modules.loss import _WeightedLoss
from traincore.data.sets.random import RandomAudioDataset


class RandomGenerator(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.encoder = nn.Linear(in_features, 10)
        self.decoder = nn.Linear(10, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(self.encoder(x))


class RandomDiscriminator(nn.Module):
    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        self.encoder = nn.Linear(in_features, 10)
        self.decoder = nn.Linear(10, out_features)

    def forward(self, x: Tensor, x_hat: Tensor | None = None) -> dict[str, Tensor]:
        return {"test": self.decoder(self.encoder(x))}


class RandomLoss(_WeightedLoss):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        discriminator_outut: dict[str, Tensor],
        clean_mix: Tensor | None = None,
        generated_mix: Tensor | None = None,
    ) -> Tensor:
        return discriminator_outut["test"]


class TestGANRecipe:
    def test_should_do_a_training_step(self) -> None:
        trainer = l.Trainer(fast_dev_run=True)

        model = nn.ModuleDict({
            "generator": RandomGenerator(100, 100),
            "discriminator": RandomDiscriminator(100, 1),
        })

        loss = nn.ModuleDict({
            "generator": RandomLoss(),
            "discriminator": RandomLoss(),
        })

        optimizer: dict[str, partial] = {
            "generator": partial(optim.Adam, lr=0.05),
            "discriminator": partial(optim.Adam, lr=0.05),
        }

        ds = RandomAudioDataset(
            n_examples=100, n_sources=2, n_channels=1, n_samples=100
        )
        ds.setup()

        dm = l.LightningDataModule.from_datasets(train_dataset=ds)

        gan_recipe = GanTrainRecipe(
            model=model,
            loss=loss,
            optimizer=optimizer,  # ty: ignore
            scheduler={},
        )
        trainer.fit(model=gan_recipe, datamodule=dm)
