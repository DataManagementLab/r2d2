import logging
from typing import Any

import attrs
import hydra
import pandas as pd
import tqdm
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, MISSING

from lib.utils import get_data_path

logger = logging.getLogger(__name__)


@attrs.define
class Config:
    defaults: list[Any] = [{"dataset": MISSING}, "_self_"]

    partition: str = MISSING


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    eval_path = get_data_path() / cfg.dataset.name / "slicing" / cfg.partition
    run_paths = list(sorted(eval_path.joinpath("runs").glob("*.csv")))

    results = []
    for run_path in tqdm.tqdm(run_paths, desc="process runs"):
        results.append(pd.read_csv(run_path))

    results = pd.concat(results)
    results.to_csv(eval_path / "results.csv")
    del results["lake"]
    del results["ground_truth"]

    # for each slicer and TpT, aggregation contains the mean (across instances) of metrics
    aggregation = results.groupby(["slicer", "TpT"]).mean()

    precisions = aggregation["precision"].unstack()
    precisions.to_csv(eval_path / f"precisions_{cfg.dataset.name}.csv")


if __name__ == "__main__":
    main()
