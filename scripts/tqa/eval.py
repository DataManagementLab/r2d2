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
    eval_path = get_data_path() / cfg.dataset.name / "tqa" / cfg.partition
    run_paths = list(sorted(eval_path.joinpath("runs").glob("*.csv")))

    results = []
    for run_path in tqdm.tqdm(run_paths, desc="process runs"):
        results.append(pd.read_csv(run_path))

    results = pd.concat(results)
    results.to_csv(eval_path / "results.csv")

    metrics = results.copy()
    del metrics["lake"]
    del metrics["ground_truth"]
    del metrics["information_need"]
    del metrics["sql_query"]
    del metrics["request"]
    del metrics["answer"]
    del metrics["prediction"]

    # for each configuration, aggregation contains the mean (across instances) of metrics
    aggregation = metrics.groupby(["embedder", "slicer", "index", "merger", "TpT", "tok"]).mean()
    print(aggregation[["accuracy", "recall", "num_result_tiles", "num_retrieved_tiles"]])

    accuracy_recall = aggregation.reset_index()[["slicer", "merger", "accuracy", "recall"]]
    accuracy_recall.to_csv(eval_path / f"accuracy_recall_{cfg.dataset.name}.csv", index=False)

    predictions = results.copy()
    predictions = predictions.groupby(["ground_truth", "slicer"]).sum()[["answer", "prediction", "accuracy", "recall", "num_result_tiles"]]
    pd.set_option("display.max_rows", None)
    print(predictions.head())


if __name__ == "__main__":
    main()
