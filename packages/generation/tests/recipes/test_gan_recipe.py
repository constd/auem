from functools import partial

import lightning as l
from generation.losses.adversarial_loss import FeatureMatchingLoss
from generation.models.discriminators.combiner import CombinerDiscriminator
from generation.recipe.gan_recipe import GanTrainRecipe
from torch import Tensor, nn, optim
from torch.nn.modules.loss import _WeightedLoss
from traincore.data.modules import GenericDataModule
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

    def forward(
        self, x: Tensor, x_hat: Tensor | None = None
    ) -> dict[str, Tensor | list[Tensor]]:
        out = self.decoder(self.encoder(x))
        return {"estimates": out, "feature_maps": list(out)}


class RandomLoss(_WeightedLoss):
    def forward(
        self,
        discriminator_outut: dict[str, Tensor],
        clean_mix: Tensor | None = None,
        generated_mix: Tensor | None = None,
    ) -> dict[str, Tensor]:
        return {"loss": discriminator_outut["estimates_real"][0].mean()}


class TestGANRecipe:
    def test_should_do_a_training_step(self) -> None:
        device: str = "cpu"
        trainer = l.Trainer(fast_dev_run=True, accelerator=device)

        model = nn.ModuleDict({
            "generator": RandomGenerator(100, 100),
            "discriminator": CombinerDiscriminator(
                discriminators={"test": RandomDiscriminator(100, 1)}
            ),
        })

        loss = {
            "generator": RandomLoss(),
            "discriminator": RandomLoss(),
            "feature": FeatureMatchingLoss(),
            "reconstruction": nn.L1Loss(),
        }

        optimizer: dict[str, partial] = {
            "generator": partial(optim.Adam, lr=0.05),
            "discriminator": partial(optim.Adam, lr=0.05),
        }

        train_ds = RandomAudioDataset(
            n_examples=100, n_sources=2, n_channels=1, n_samples=100
        )
        train_ds.setup()

        dm = GenericDataModule(
            datasets={
                "train": {"main": train_ds},
                "batch_size": {"train": 1},
            },
            num_workers=1,
        )

        gan_recipe = GanTrainRecipe(
            model=model,
            loss=loss,
            optimizer=optimizer,  # ty: ignore
            scheduler={},
        )
        trainer.fit(model=gan_recipe, datamodule=dm)
