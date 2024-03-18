import logging
import os
import shutil
import subprocess
import tarfile

import attrs
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from lib.utils import get_data_path

logger = logging.getLogger(__name__)


@attrs.define
class Config:
    defaults: list = [{"dataset": "openwikitable"}, "_self_"]


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    dataset_dir = get_data_path() / cfg.dataset.name / "download"
    os.makedirs(dataset_dir, exist_ok=True)

    repo_dir = dataset_dir / "Open-WikiTable"
    if not repo_dir.is_dir():
        logger.info(f"Clone the git repository.")
        try:
            subprocess.check_call(["git", "clone", cfg.dataset.url, str(repo_dir)])
        except:
            logger.error("Automatically cloning the git repository failed.")
            logger.error(f"Please manually execute `git clone {cfg.url} {repo_dir}` and restart download.py!")
            raise

    logger.info("Extract the data.")
    with tarfile.open(repo_dir / "data" / "data.tar.gz", 'r:gz') as tar:
        tar.extractall(dataset_dir)

    logger.info("Remove repository.")
    shutil.rmtree(repo_dir)

    logger.info("Done!")


if __name__ == "__main__":
    main()
