from lightning.pytorch.callbacks import Callback
from traincore.data.sets import FolderDataset

from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch import loggers as llog

from traincore.config_stores.callbacks import callback_store


@callback_store(name="audiologger")
class LogAudio(Callback):
    def __init__(self, dataset: FolderDataset, mix_of_interest: str = "mix") -> None:
        self.dataset = dataset
        self.mix_of_interest: str = mix_of_interest

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        self.dataset.setup()
        logger: llog.Logger | None = trainer.logger
        if logger:
            for datum in self.dataset:
                audio = datum[self.mix_of_interest].to(pl_module.device).unsqueeze(0)
                # grab the first (and only) item and the first (and only source)
                estimate = (
                    pl_module.model.generator(audio)[0, 0].cpu().numpy(force=True)  # ty: ignore[call-non-callable, possibly-missing-attribute]
                )  # ty: ignore[call-non-callable]
                description = f"""{datum["id"]}"""
                match logger:
                    case llog.TensorBoardLogger():
                        pass
                    case llog.WandbLogger():
                        pass
                    case llog.CometLogger():
                        logger.experiment.log_audio(
                            audio_data=estimate.T,
                            sample_rate=int(pl_module.model.generator.sample_rate),  # ty: ignore[possibly-missing-attribute]
                            file_name=description,
                            step=trainer.global_step,
                        )
                    case _:
                        pass
