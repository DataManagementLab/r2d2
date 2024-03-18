import collections
import json
import logging
import os

import attrs
import cattrs
import hydra
import tqdm
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from lib.lake import DataLake
from lib.utils import TileSet, GroundTruth, get_data_path

logger = logging.getLogger(__name__)
stats = collections.Counter()


@attrs.define
class ExploreLakesConfig:
    defaults: list = [{"dataset": "???"}, "_self_"]

    partition: str = "val"
    silent: bool = False

    visualize_slicing: bool = False
    visualize_ground_truths: bool = False
    sort_slicing: bool = True
    max_value_len: int = 30


ConfigStore.instance().store(name="explore_lakes", node=ExploreLakesConfig)


@hydra.main(version_base=None, config_path="../../config", config_name="explore_lakes")
def explore_lakes(cfg: DictConfig) -> None:
    logger.debug("Prepare directories.")
    explore_path = get_data_path() / cfg.dataset.name / "explore" / cfg.partition
    os.makedirs(explore_path, exist_ok=True)

    logger.debug("Glob lakes.")
    lakes_path = get_data_path() / cfg.dataset.name / "lakes" / cfg.partition
    lake_paths = list(sorted(lakes_path.glob("*.zip")))
    stats["num_lakes"] = len(lake_paths)

    logger.debug("Load ground truths")
    ground_truths = []
    for lake_path in lake_paths:
        ground_truth_path = lake_path.parent / f"{lake_path.name[:-4]}-ground-truth.json"
        with open(ground_truth_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        ground_truths += [cattrs.structure(gt_data, GroundTruth) for gt_data in data]
    gt_by_name = {ground_truth.name: ground_truth for ground_truth in ground_truths}
    stats["num_ground_truths"] = len(ground_truths)

    logger.debug("Explore lakes.")
    tables_per_lake = []
    cells_per_lake = []
    rows_per_table = []
    cols_per_table = []
    cells_per_table = []
    tables_per_result = []
    cells_per_result = []
    for lake_ix, lake_path in enumerate(tqdm.tqdm(lake_paths, desc="Explore lakes")):
        lake = DataLake.load(lake_path, silent=cfg.silent)

        tables_per_lake.append(len(lake.tables))
        cells_per_lake.append(lake.num_cells)

        logger.debug("Scan tables.")
        for table in lake.get_all_tables():
            rows_per_table.append(len(table.row_index))
            cols_per_table.append(len(table.col_index))
            cells_per_table.append(table.num_cells)

        logger.debug("Scan tiles.")
        for tile_kind_ix, tile_kind in enumerate(lake.tiles.keys()):
            if tile_kind.startswith("ground-truth"):
                gt_tile_set = TileSet.from_lake(lake, tile_kind=tile_kind)
                tables_per_result.append(len(gt_tile_set.get_all_tables()))
                cells_per_result.append(gt_tile_set.compute_num_cells())

                ground_truth = gt_by_name[tile_kind[13:]]
                if "where" in ground_truth.sql_query.lower():
                    stats["ground_truth_contains_where"] += 1

        lake_tile_set = TileSet.from_lake(lake)

        if cfg.visualize_slicing:
            logger.debug("Visualize slicing.")
            for table_ix, table in enumerate(lake.get_all_tables()):
                table_tile_set = TileSet.from_table(table, tile_kind="slice-tile")
                title = f"""
                    <a href=\"lake-{lake_ix - 1}-table-0.html\">Previous lake</a>
                    <a href=\"lake-{lake_ix}-table-{table_ix - 1}.html\">Previous table</a>
                    <a href=\"lake-{lake_ix}-table-{table_ix + 1}.html\">Next table</a>
                    <a href=\"lake-{lake_ix + 1}-table-0.html\">Next lake</a>
                    <br/><br/>Lake: {lake.name}
                    <br/>Table: {table.name}
                    <br/><br/>Number of tiles: {len(table_tile_set.get_all_tiles())}
                """
                slicing_path = explore_path / "slicing"
                os.makedirs(slicing_path, exist_ok=True)
                table_tile_set.visualize_as_html(
                    slicing_path / f"lake-{lake_ix}-table-{table_ix}.html",
                    tile_kind="slice-tile",
                    show_all_tables=False,
                    sort=cfg.sort_slicing,
                    title=title,
                    max_value_len=cfg.max_value_len,
                    color_tile_kinds=["slice-tile"]
                )

        if cfg.visualize_ground_truths:
            logger.debug("Visualize ground truths.")
            for tile_kind_ix, tile_kind in enumerate(lake.tiles.keys()):
                if tile_kind.startswith("ground-truth"):
                    ground_truth = gt_by_name[tile_kind[13:]]
                    title = f"""
                        <a href=\"lake-{lake_ix - 1}-ground-truth-0.html\">Previous lake</a>
                        <a href=\"lake-{lake_ix}-ground-truth-{tile_kind_ix - 1}.html\">Previous query</a>
                        <a href=\"lake-{lake_ix}-ground-truth-{tile_kind_ix + 1}.html\">Next query</a>
                        <a href=\"lake-{lake_ix + 1}-ground-truth-0.html\">Next lake</a>
                        <br/><br/>{lake.name}<br/>Query: {ground_truth.sql_query}
                        <br/>Information need: {ground_truth.information_need}
                    """
                    ground_truth_path = explore_path / "ground_truth"
                    os.makedirs(ground_truth_path, exist_ok=True)
                    lake_tile_set.visualize_as_html(
                        ground_truth_path / f"lake-{lake_ix}-ground-truth-{tile_kind_ix}.html",
                        tile_kind=tile_kind,
                        show_all_tables=False,
                        title=title,
                        max_value_len=(cfg.max_value_len),
                        color_tile_kinds=[tile_kind],
                        tile_kind_to_font_style={tile_kind: "font-weight:bold;"}
                    )

    with open(explore_path / "tables_per_lake.json", "w", encoding="utf-8") as file:
        json.dump(tables_per_lake, file)

    with open(explore_path / "cells_per_lake.json", "w", encoding="utf-8") as file:
        json.dump(cells_per_lake, file)

    with open(explore_path / "rows_per_table.json", "w", encoding="utf-8") as file:
        json.dump(rows_per_table, file)

    with open(explore_path / "cols_per_table.json", "w", encoding="utf-8") as file:
        json.dump(cols_per_table, file)

    with open(explore_path / "cells_per_table.json", "w", encoding="utf-8") as file:
        json.dump(cells_per_table, file)

    with open(explore_path / "tables_per_result.json", "w", encoding="utf-8") as file:
        json.dump(tables_per_result, file)

    with open(explore_path / "cells_per_result.json", "w", encoding="utf-8") as file:
        json.dump(cells_per_result, file)

    with open(explore_path / "stats.json", "w", encoding="utf-8") as file:
        json.dump(dict(stats), file)

    logger.info("Done!")


if __name__ == "__main__":
    explore_lakes()
