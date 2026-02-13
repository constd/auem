from hydra_zen import ZenStore
from hydra_zen.third_party.beartype import (
    validates_with_beartype,
)
from torchmetrics import audio as tma  # noqa

__all__ = ["metric_store"]

metric_store: ZenStore = ZenStore()(
    group="metric",
    populate_full_signature=True,
    zen_wrappers=validates_with_beartype,
    hydra_convert="all",
)

metric_store(
    tma.ComplexScaleInvariantSignalNoiseRatio,
    name="csisnr",
    populate_full_signature=True,
)
if getattr(tma, "DeepNoiseSuppressionMeanOpinionScore"):
    metric_store(
        getattr(tma, "DeepNoiseSuppressionMeanOpinionScore"),
        name="dnsmos",
        populate_full_signature=True,
    )
if getattr(tma, "NonIntrusiveSpeechQualityAssessment"):
    metric_store(
        getattr(tma, "NonIntrusiveSpeechQualityAssessment"),
        name="nisqa",
        populate_full_signature=True,
    )
if getattr(tma, "PerceptualEvaluationSpeechQuality"):
    metric_store(
        getattr(tma, "PerceptualEvaluationSpeechQuality"),
        name="pesq",
        populate_full_signature=True,
    )
metric_store(
    tma.ScaleInvariantSignalDistortionRatio,
    name="sisdr",
    populate_full_signature=True,
)
metric_store(
    tma.ScaleInvariantSignalNoiseRatio,
    name="sisnr",
    populate_full_signature=True,
)
metric_store(
    tma.SignalDistortionRatio,
    name="sdr",
    populate_full_signature=True,
)
