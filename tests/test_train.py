from omegaconf import DictConfig
import auem.train


def test_binary_classification_with_random_data():
    config = DictConfig(
        {
            "logger": None,
            "data": None,
            "model": None,
            "recipe": None,
            "trainer": "lightning.pytorch.trainer.trainer.Trainer",
        }
    )

    auem.train.train(config)
