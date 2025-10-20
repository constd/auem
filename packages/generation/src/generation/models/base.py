from torch import nn

from traincore.config_stores.models import model_store

__all__ = ["GanBase"]


@model_store(name="ganbase")
class GanBase(nn.Module):
    def __init__(self, generator: nn.Module, discriminator: nn.Module):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
