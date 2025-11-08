from hydra import compose, initialize

import auem.train


def test_binary_classification_with_random_data():
    with initialize(
        config_path="../src/auem/configs",
        job_name="test",
        version_base="1.3",
    ):
        config = compose(
            config_name="train_config", overrides=["+experiment=tests/classification"]
        )

    auem.train.train(config)


def test_gan_recipe_with_random_data():
    with initialize(
        config_path="../src/auem/configs",
        job_name="test",
        version_base="1.3",
    ):
        config = compose(
            config_name="train_config",
            overrides=[
                "+experiment=tests/gan_mel",
                "+model/discriminator@recipe.model.discriminator.discriminators.mp=multiperiod",
                "+recipe.model.discriminator.discriminators.mp.configs.period=[4]",
            ],
        )

    auem.train.train(config)


def test_gan_recipe_with_cqt_discriminator():
    with initialize(
        config_path="../src/auem/configs",
        job_name="test",
        version_base="1.3",
    ):
        config = compose(
            config_name="train_config",
            overrides=[
                "+experiment=tests/gan_mel",
                "+model/discriminator@recipe.model.discriminator.discriminators.cqt=multicqt",
            ],
        )

    # - /model/discriminator@recipe.model.discriminator.discriminators.mp: multiperiod
    # - /model/discriminator@recipe.model.discriminator.discriminators.ms: multiscale

    auem.train.train(config)
