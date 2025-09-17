from hydra_zen import ZenStore

__all__ = ["recipe"]

recipe: ZenStore = ZenStore()(
    group="recipe",
    populate_full_signature=True,
    hydra_convert="all",
)


from traincore.recipes import *  # noqa

recipe.add_to_hydra_store()
