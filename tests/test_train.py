from hydra import compose, initialize

import auem.train


def test_binary_classification_with_random_data():
    initialize(
        config_path="../src/auem/configs",
        job_name="test",
        version_base="1.3",
    )
    config = compose(
        config_name="train_config", overrides=["+experiment=tests/classification"]
    )

    auem.train.train(config)


def test_gan_recipe_with_random_data():
    initialize(
        config_path="../src/auem/configs",
        job_name="test",
        version_base="1.3",
    )
    config = compose(config_name="train_config", overrides=["+experiment=tests/gan"])

    auem.train.train(config)
