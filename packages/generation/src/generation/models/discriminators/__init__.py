from generation.models.discriminators.period import PeriodDiscriminator
from generation.models.discriminators.multi import MultiDiscriminator
from generation.models.discriminators.combiner import CombinerDiscriminator
from generation.models.discriminators.scale import ScaleDiscriminator

__all__ = [
    "CombinerDiscriminator",
    "MultiDiscriminator",
    "PeriodDiscriminator",
    "ScaleDiscriminator",
]
