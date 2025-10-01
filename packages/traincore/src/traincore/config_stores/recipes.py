from hydra_zen import ZenStore

__all__ = ["recipe_store"]

recipe_store: ZenStore = ZenStore()(
    group="recipe",
    populate_full_signature=True,
    hydra_convert="all",
)


from traincore.recipes import *  # noqa
