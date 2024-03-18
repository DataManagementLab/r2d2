import json
import logging
import math
import pathlib
from typing import Any

import attrs
import hydra
from hydra.core.config_store import ConfigStore
from matplotlib import pyplot as plt
from omegaconf import MISSING, DictConfig

from lib.colors import COLOR_9A
from lib.utils import get_data_path

logger = logging.getLogger(__name__)


@attrs.define
class Config:
    defaults: list[Any] = [{"dataset": MISSING}, "_self_"]

    partition: str = MISSING


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    explore_path = get_data_path() / cfg.dataset.name / "explore" / cfg.partition

    for identifier, log in [
        ("tables_per_lake", False),
        ("cells_per_lake", True),
        ("rows_per_table", True),
        ("cols_per_table", False),
        ("cells_per_table", True),
        ("tables_per_result", False),
        ("cells_per_result", True)
    ]:
        with open(explore_path / f"{identifier}.json", "r", encoding="utf-8") as file:
            values = json.load(file)

        histogram(values, explore_path, identifier, log)


def histogram(values: list[int], explore_path: pathlib.Path, identifier: str, log: bool):
    if len(set(values)) == 1:
        bins = [values[0] - 0.1, values[0] + 0.1]
    elif log:
        bins = [0] + [10 ** x for x in range(0, int(math.log10(max(values))) + 3)]
    else:
        bins = list(range(min(values), max(values) + 1))

    plt.style.use("ggplot")
    plt.rcParams["figure.figsize"] = (5.25, 2.75)
    plt.rcParams["axes.labelsize"] = 10
    plt.figure(constrained_layout=True)

    plt.hist(values, bins=bins, color=COLOR_9A)
    plt.title(" ".join(identifier.split("_")))
    plt.xlabel(f"number of {identifier.split('_')[0]}")
    plt.ylabel("count")
    if log:
        plt.xscale("log")

    plt.savefig(explore_path / f"{identifier}.pdf", bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":
    main()
