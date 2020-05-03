"""Train a 'classification' model using an IterableDataset."""
import logging
import os

import hydra
import torch
from omegaconf import DictConfig
from torch.cuda import is_available as is_cuda_available
from torch.utils import tensorboard as tb
from tqdm import tqdm

from auem.train import dataloaders, datasets, training_setup

# import auem.evaluation.confusion as confusion

logger = logging.getLogger(__name__)


def train(cfg: DictConfig) -> None:
    """Train a model given a configuration from the config."""
    device = cfg.cuda.device if cfg.cuda.enable and is_cuda_available() else "cpu"

    ds_train, ds_valid = datasets(cfg)
    dl_train, dl_valid = dataloaders(cfg, ds_train, ds_valid)

    # Setup the train iterator, and get an example batch to use in defining the model.
    train_iterator = iter(dl_train)
    batch = train_iterator.next()

    model = hydra.utils.instantiate(cfg.model, batch["X"].shape).to(device)
    optimizer, scheduler, criterion = training_setup(cfg, model)

    writer = tb.SummaryWriter()
    writer.add_graph(model, batch["X"].to(device))

    for epoch in tqdm(range(cfg.epochs), position=0, desc="Epoch"):
        losses = 0
        model.train()
        for batch_num in tqdm(range(cfg.steps), position=1, desc="Batch"):
            X, y = batch["X"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            losses += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            writer.add_scalar(
                f"loss/training/step", loss.item(), global_step=batch_num,
            )
            batch = train_iterator.next()

        writer.add_scalar(f"loss/training/epoch", losses / batch_num, global_step=epoch)
        # writer.add_scalars(f"accuracy/training", accuracies, global_step=epoch)
        # evaluation loop
        if cfg.eval:
            eval_iterator = iter(dl_valid)
            model.eval()
            losses = 0
            for _, batch in tqdm(
                enumerate(eval_iterator),
                total=len(ds_valid) // cfg.dataloader.params.batch_size,
                position=1,
                desc="Batch",
            ):
                X, y = batch["X"].to(device), batch["label"].to(device)
                output = model(X)
                loss = criterion(output, y.squeeze_())
                # Get the embeddings for each batch, so we can save them for tensorboard
                # if (
                #     cfg.checkpoint.embeddings.enabled
                #     and epoch % cfg.checkpoint.embeddings.frequency == 0
                # ):
                #     embeddings.extend(output.to("cpu").tolist())
                #     ys.extend([ds_valid.c2l[x] for x in y.tolist()])
            writer.add_scalar(
                f"loss/validation/epoch", losses / len(batch), global_step=epoch
            )
            # writer.add_scalars(f"accuracy/validation", accuracies, global_step=epoch)
            # if (
            #     cfg.checkpoint.embeddings.enabled
            #     and epoch % cfg.checkpoint.embeddings.frequency == 0
            # ):
            #     writer.add_embedding(
            #         torch.tensor(embeddings), metadata=ys, global_step=epoch
            #     )
            # confusion.log_confusion_matrix(writer, y, output, class_names)
        if cfg.checkpoint.model.enabled and epoch % cfg.checkpoint.model.frequency == 0:
            torch.save(model, f"{os.getcwd()}/{cfg.model['class']}_{cfg.epochs}.pt")
    torch.save(model, f"{os.getcwd()}/{cfg.model['class']}_{cfg.epochs}_final.pt")


@hydra.main(config_path="config/config-iterable.yaml")
def main(cfg: DictConfig) -> None:
    """Execute Training using hydra configs."""
    import git

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    logger.info(f"""Git hash: {str(sha)}""")
    train(cfg)


if __name__ == "__main__":
    main()
