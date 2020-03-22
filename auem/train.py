import hydra
import logging
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def train(cfg: DictConfig) -> None:
    logger.debug(f"in train loop {cfg.intrain}")


@hydra.main(config_path="config/config.yaml")
def main(cfg: DictConfig) -> None:
    logger.info(f"{cfg.inmain}")
    train(cfg)


if __name__ == "__main__":
    main()
