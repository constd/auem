"""Train a pytorch model using a pytorch Dataset with a known length."""
import logging
import os
from math import ceil
from typing import NamedTuple, Tuple

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.cuda import is_available as is_cuda_available
from torch.utils import tensorboard as tb
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# import auem.evaluation.confusion as confusion

logger = logging.getLogger(__name__)


def datasets(cfg: DictConfig) -> Tuple[Dataset]:
    """Create dataset objects using the config."""
    transforms = hydra.utils.instantiate(cfg.transform)
    ds_train = hydra.utils.get_class(cfg.dataset["class"])(
        audioset_annotations=cfg.dataset["folds"]["train"],
        transforms=transforms,
        **cfg.dataset.params,
    )
    ds_valid = hydra.utils.get_class(cfg.dataset["class"])(
        audioset_annotations=cfg.dataset["folds"]["val"],
        transforms=transforms,
        evaluate=True,
        **cfg.dataset.params,
    )
    return (ds_train, ds_valid)


def dataloaders(
    cfg: DictConfig, ds_train: Dataset, ds_valid: Dataset
) -> Tuple[DataLoader]:
    """Create dataloader objects using the config and datasets."""
    dl_train = hydra.utils.get_class(cfg.dataloader["class"])(
        ds_train, **cfg.dataloader.params
    )
    dl_valid = hydra.utils.get_class(cfg.dataloader["class"])(
        ds_valid, **cfg.dataloader.params
    )
    return (dl_train, dl_valid)


class TrainingSetupTuple(NamedTuple):
    """A TrainingSetup namedtuple.

    Used to enforce type and make the result of `training_setup` accessible
    by property/name.
    """

    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    criterion: nn.modules.loss._Loss


def training_setup(cfg: DictConfig, model: nn.Module) -> TrainingSetupTuple:
    """Create non-data, non-model training components.

    (eg. optimizer, scheduler, criteron, ...)

    Parameters
    ----------
    cfg : DictConfig
        The complete config.

    model : nn.Module
        Instantiated model.

    Returns
    -------
    optimizer, schedule, criterion
        Tuple of the above, instantiated.
    """
    optimizer = hydra.utils.get_class(cfg.optim["class"])(
        model.parameters(), **cfg.optim.params
    )
    scheduler = hydra.utils.get_class(cfg.scheduler["class"])(
        optimizer, **cfg.scheduler.params
    )
    criterion = hydra.utils.instantiate(cfg.criterion)
    return TrainingSetupTuple(optimizer, scheduler, criterion)


def train(cfg: DictConfig) -> None:
    """Train a model given a configuration from the config."""
    device = cfg.cuda.device if cfg.cuda.enable and is_cuda_available() else "cpu"

    ds_train, ds_valid = datasets(cfg)
    dl_train, dl_valid = dataloaders(cfg, ds_train, ds_valid)
    example_batch = iter(dl_train).next()

    model = hydra.utils.instantiate(cfg.model, example_batch["X"].shape).to(device)
    optimizer, scheduler, criterion = training_setup(cfg, model)

    num_batches_train = ceil(len(ds_train) / cfg.dataloader.params.batch_size)
    num_batches_valid = ceil(len(ds_valid) / cfg.dataloader.params.batch_size)

    writer = tb.SummaryWriter()
    writer.add_graph(model, example_batch["X"].to(device))
    for epoch in tqdm(range(cfg.epochs), position=0, desc="Epoch"):
        losses = 0
        model.train()
        for batch_num, batch in tqdm(
            enumerate(dl_train), total=num_batches_train, position=1, desc="Batch"
        ):
            X, y = batch["X"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            losses += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            writer.add_scalar(
                f"loss/training/batch", loss.item(), global_step=batch_num,
            )
        writer.add_scalar(f"loss/epoch/training", losses / batch_num, global_step=epoch)
        # writer.add_scalars(f"accuracy/training", accuracies, global_step=epoch)
        # evaluation loop
        if cfg.eval:
            model.eval()
            embeddings, ys, losses = [], [], 0
            for _, batch in tqdm(
                enumerate(dl_valid), total=num_batches_valid, position=1, desc="Batch"
            ):
                X, y = batch["X"].to(device), batch["label"].to(device)
                output = model.get_embedding(X)
                loss = criterion(output, y.squeeze_())
                if (
                    cfg.checkpoint.embeddings.enabled
                    and epoch % cfg.checkpoint.frequency == 0
                ):
                    embeddings.extend(output.to("cpu").tolist())
                    ys.extend([ds_valid.c2l[x] for x in y.tolist()])
            writer.add_scalar(
                f"loss/epoch/validation", losses / len(batch), global_step=epoch
            )
            # writer.add_scalars(f"accuracy/validation", accuracies, global_step=epoch)
            if (
                cfg.checkpoint.embeddings.enabled
                and epoch % cfg.checkpoint.embeddings.frequency == 0
            ):
                writer.add_embedding(
                    torch.tensor(embeddings), metadata=ys, global_step=epoch
                )

            # confusion.log_confusion_matrix(writer, y, output, class_names)
        if cfg.checkpoint.model.enabled and epoch % cfg.checkpoint.model.frequency == 0:
            torch.save(model, f"{os.getcwd()}/{cfg.model.name}_{cfg.epochs}_final.pt")
    torch.save(model, f"{os.getcwd()}/{cfg.model.name}_{cfg.epochs}_final.pt")


@hydra.main(config_path="config/config.yaml")
def main(cfg: DictConfig) -> None:
    """Execute Training using hydra configs."""
    import git

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    logger.info(f"""Git hash: {str(sha)}""")
    train(cfg)


if __name__ == "__main__":
    main()
