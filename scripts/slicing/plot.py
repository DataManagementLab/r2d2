import logging
from typing import Any

import attrs
import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
from matplotlib import pyplot as plt
from omegaconf import MISSING, DictConfig

from lib.colors import COLOR_1A, COLOR_3A, COLOR_9A, COLOR_BLACK
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
    precisions = pd.read_csv(eval_path / f"precisions_{cfg.dataset.name}.csv", index_col="slicer")
    precisions.columns = list(map(int, precisions.columns))

    plt.style.use("ggplot")
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (2.5, 1.75)
    plt.rcParams["axes.labelsize"] = 10
    plt.figure(constrained_layout=True)

    labels = {
        "lib.slice.NoSlicer": "no slicing",
        "lib.slice.RowWiseSlicer": "row-wise",
        "lib.slice.ColWiseSlicer": "column-wise",
        "lib.slice.DivisiveSlicer": "divisive"
    }

    colors = {
        "lib.slice.NoSlicer": COLOR_BLACK,
        "lib.slice.RowWiseSlicer": COLOR_1A,
        "lib.slice.ColWiseSlicer": COLOR_3A,
        "lib.slice.DivisiveSlicer": COLOR_9A
    }

    for slicer, label in labels.items():
        if slicer not in precisions.index:
            logger.warning(f"Missing slicer: '{slicer}'!")
            continue

        values = precisions.loc[slicer]

        # plot the last values that are not changing with a dotted line
        # constant_values = values.loc[values == values.iloc[-1]]
        # changing_values = values.loc[values.index.difference(constant_values.index[1:])]
        #
        # plt.plot(
        #     changing_values.index,
        #     changing_values,
        #     "-",
        #     color=colors[slicer],
        #     label=labels[slicer]
        # )
        # plt.plot(
        #     constant_values.index,
        #     constant_values,
        #     "--",
        #     color=colors[slicer],
        # )

        # plot all values normally
        plt.plot(
            values.index,
            values,
            "-",
            color=colors[slicer],
            label=labels[slicer]
        )

    plt.ylabel("precision")
    plt.xlabel("tiles per table")
    plt.ylim((0, 1))
    plt.yticks((0, 0.25, 0.50, 0.75, 1.00), labels=("0", "", "0.5", "", "1.0"))
    # plt.legend()
    if cfg.dataset.name == "openwikitable":
        plt.xticks((1, 500, 1000, 1500, 2000), labels=("1", "", "1000", "", "2000"))
    elif cfg.dataset.name == "spider":
        plt.xticks((1, 25, 50, 75, 100), labels=("1", "", "50", "", "100"))
    plt.savefig(eval_path / f"precisions_{cfg.dataset.name}.pdf", bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":
    main()
