import logging
import os
from math import ceil

import hydra
import torch
from omegaconf import DictConfig
from torch.cuda import is_available as is_cuda_available
from torch.utils import tensorboard as tb
from tqdm import tqdm

import auem.evaluation.confusion as confusion

logger = logging.getLogger(__name__)


def train(cfg: DictConfig) -> None:
    device = cfg.cuda.device if cfg.cuda.enable and is_cuda_available() else "cpu"

    transforms = hydra.utils.instantiate(cfg.transform)
    dataset = hydra.utils.get_class(cfg.dataset["class"])(
        transforms=transforms, **cfg.dataset.params
    )

    # TODO: there's got to be a better way to do this!
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    ds_train, ds_valid = torch.utils.data.random_split(
        dataset, [train_size, valid_size]
    )

    # hydra doesn't work with non primitives like the dataset class
    # TODO: file a bug with hydra to allow non-promitive pass-through of non-primitives
    dl_train = hydra.utils.get_class(cfg.dataloader["class"])(
        ds_train, **cfg.dataloader.params
    )
    dl_valid = hydra.utils.get_class(cfg.dataloader["class"])(
        ds_valid, **cfg.dataloader.params
    )

    model = hydra.utils.instantiate(cfg.model).to(device)

    # hydra doesn't work with non primitives
    # like the model.parameters() generator in the following
    # TODO: file a bug with hydra to allow non-promitive pass-through of non-primitives
    optimizer = hydra.utils.get_class(cfg.optim["class"])(
        model.parameters(), **cfg.optim.params
    )

    criterion = hydra.utils.instantiate(cfg.criterion)

    num_batches_train = ceil(len(ds_train) / cfg.dataloader.params.batch_size)
    num_batches_valid = ceil(len(ds_valid) / cfg.dataloader.params.batch_size)

    writer = tb.SummaryWriter()
    writer.add_graph(model, iter(dl_train).next()["X"].to(device))
    for epoch in tqdm(range(cfg.epochs), position=0, desc="Epoch"):
        losses, items_seen = 0, 0
        model.train()
        for _, batch in tqdm(
            enumerate(dl_train), total=num_batches_train, position=1, desc="Batch"
        ):
            X, y = batch["X"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            losses += loss.item()
            items_seen += X.shape[0]
            loss.backward()
            optimizer.step()
        writer.add_scalar(f"loss/training", losses / batch.shape[0], global_step=epoch)
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
                loss = criterion(output, y)
                if (
                    cfg.checkpoint.embeddings.enabled
                    and epoch % cfg.checkpoint.frequency == 0
                ):
                    embeddings.extend(output.to("cpu").tolist())
                    ys.extend([ds_valid.c2l[x] for x in y.tolist()])
            writer.add_scalar(
                f"loss/validation", losses / len(batch), global_step=epoch
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
            model.save(f"{os.getcwd()}/{cfg.model.name}_{cfg.epochs}_final.pt")
    model.save(f"{os.getcwd()}/{cfg.model.name}_{cfg.epochs}_final.pt")


@hydra.main(config_path="config/config.yaml")
def main(cfg: DictConfig) -> None:
    # print(cfg.pretty())
    train(cfg)


if __name__ == "__main__":
    main()
