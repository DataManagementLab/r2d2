import json
import logging
import multiprocessing
import os
import pathlib
from typing import Any

import attrs
import cattrs
import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig, MISSING
from pqdm.processes import pqdm

from lib.embed import BaseEmbedder
from lib.lake import DataLake
from lib.slice import BaseSlicer
from lib.utils import TileSet, GroundTruth, get_data_path, tqdm_silent, ConfusionMatrix

logger = logging.getLogger(__name__)


@attrs.define
class Config:
    defaults: list[Any] = [
        {"dataset": MISSING},
        {"table_embedder": MISSING},
        {"slicer": MISSING},
        "_self_"
    ]

    config_name: str = MISSING
    partition: str = MISSING

    use_cached_embedded_lakes: bool = True

    skip_large_lakes: int | None = None

    num_processes: int = 8
    silent: bool = True


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.debug("Parse config.")
    config_parts = cfg.config_name.split("/")[-1].split("_")
    config_parts = {part.split("=")[0]: part.split("=")[1] for part in config_parts}
    logger.debug(f"Parameters: {config_parts}")

    logger.debug("Prepare directories.")
    eval_path = get_data_path() / cfg.dataset.name / "slicing" / cfg.partition
    os.makedirs(eval_path, exist_ok=True)

    lakes_path = get_data_path() / cfg.dataset.name / "lakes" / cfg.partition
    lake_paths = list(sorted(lakes_path.glob("*.zip")))

    logger.debug("Load components.")
    table_embedder = instantiate(cfg.table_embedder)
    slicer = instantiate(cfg["slicer"])

    logger.debug("Evaluate lakes.")
    arguments = [{"lake_path": lake_path, "eval_path": eval_path, "table_embedder": table_embedder, "slicer": slicer, "cfg": cfg} for lake_path in lake_paths]

    multiprocessing.set_start_method("spawn", force=True)  # allow model sharing and hydra sweeps
    all_results = pqdm(
        arguments,
        evaluate_on_lake,
        n_jobs=cfg.num_processes,
        exception_behaviour="immediate",
        desc="evaluate lakes",
        argument_type="kwargs"
    )

    logger.debug("Save results.")
    results = [b for a in all_results for b in a]  # flatten all results list
    results = pd.DataFrame(results)
    for k, v in config_parts.items():
        results[k] = v

    results_path = eval_path / "runs" / f"{cfg.config_name}.csv"
    os.makedirs(results_path.parent, exist_ok=True)
    results.to_csv(results_path, index=False)

    logger.info("Done!")


def evaluate_on_lake(lake_path: pathlib.Path, eval_path: pathlib.Path, table_embedder: BaseEmbedder, slicer: BaseSlicer, cfg: DictConfig) -> list[dict[str, Any]]:
    embedded_lake_path = eval_path / "embedded_lakes" / lake_path.name
    if cfg.use_cached_embedded_lakes and embedded_lake_path.is_file():
        lake = DataLake.load(embedded_lake_path, silent=cfg.silent)
    else:
        lake = DataLake.load(lake_path, silent=cfg.silent)

        if cfg.skip_large_lakes is not None and lake.num_cells > cfg.skip_large_lakes:
            logger.warning(f"Skipped large lake '{lake.name}'.")
            return []

        table_embedder(lake, what_to_embed=["table-cells"], silent=cfg.silent)
        os.makedirs(embedded_lake_path.parent, exist_ok=True)
        lake.save(embedded_lake_path, silent=cfg.silent)

    gt_path = lake_path.parent / f"{lake.name}-ground-truth.json"
    with open(gt_path, "r", encoding="utf-8") as file:
        ground_truths = [cattrs.structure(gt_data, GroundTruth) for gt_data in (json.load(file))]

    slicer(lake, silent=cfg.silent)

    all_results = []
    for gt_ix, ground_truth in enumerate(tqdm_silent(ground_truths, cfg.silent, desc="evaluate ground truths", leave=False)):
        gt_tile_set = TileSet.from_lake(lake, tile_kind=f"ground-truth-{ground_truth.name}")

        confusion_matrix = ConfusionMatrix.empty()
        for tile in lake.get_all_tiles(tile_kind="slice-tile"):
            t_tile_set = TileSet({tile})
            num_in_other, num_not_in_other = t_tile_set.compute_overlap(gt_tile_set)
            if num_in_other > 0:  # we "return" every tile that contains any relevant cells to achieve FN == 0 and recall == 1
                confusion_matrix.TP += num_in_other
                confusion_matrix.FP += num_not_in_other
            else:
                confusion_matrix.TN += tile.num_cells

        result = {
            "lake": lake.name,
            "ground_truth": ground_truth.name
        }
        result.update(confusion_matrix.as_dict())
        all_results.append(result)

    return all_results


if __name__ == "__main__":
    main()
