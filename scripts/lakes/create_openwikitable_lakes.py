import collections
import json
import logging
import os
import random
import sqlite3

import attrs
import cattrs
import hydra
import numpy as np
import pandas as pd
import sql_metadata
import sqlglot
import sqlglot.executor
import tqdm
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from lib.lake import DataLake, Table
from lib.utils import GroundTruth, get_data_path

logger = logging.getLogger(__name__)
stats = collections.Counter()


@attrs.define
class Config:
    defaults: list = [{"dataset": "openwikitable"}, "_self_"]


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Create Open-WikiTable Data Lakes!")

    random.seed(550677792)
    np_random_state = np.random.default_rng(seed=14307408)

    # create directories
    dataset_path = get_data_path() / cfg.dataset.name
    lakes_path = dataset_path / "lakes"
    os.makedirs(lakes_path)

    # load tables
    tables = pd.read_json(dataset_path / "download" / "tables.json")
    stats[f"original.num_tables"] = len(tables.index)

    if cfg.dataset.large_tables is not None:
        logger.warning("Restricting to large tables.")
        tables["num_cells"] = tables.apply(lambda row: len(row["rows"]) * len(row["header"]), axis=1)
        tables = tables.sort_values(by="num_cells", ascending=False)
        tables = tables.head(cfg.dataset.large_tables)
        tables = tables.sample(frac=1, random_state=np_random_state).reset_index(drop=True)

    for partition in ("train", "val", "test"):
        logger.info(f"Partition: {partition}")

        partition_path = lakes_path / partition
        os.makedirs(partition_path, exist_ok=True)

        # load ground truths
        openwikitable_partition = "valid" if partition == "val" else partition
        instances = pd.read_json(dataset_path / "download" / f"{openwikitable_partition}.json")
        instances = instances.sample(frac=1, random_state=np_random_state).reset_index(drop=True)
        stats[f"original.num_ground_truths.{partition}"] = len(instances.index)

        # create data lake
        logger.info("Create lake.")
        lake = DataLake(partition)
        id2table = {}
        for ix, table in tables.iterrows():
            name = f"page title: {table['page_title']}, section title: {table['section_title']}, caption: {table['caption']}"
            if name in lake.tables.keys():
                stats["table.name_already_exists"] += 1
                name += f" {table['name']}"
            data = pd.DataFrame(data=table["rows"], columns=table["header"])
            lake_table = Table(lake, name, data)
            id2table[table["original_table_id"]] = lake_table

        # create ground truths
        logger.info("Create ground truths.")
        ground_truths = []
        with sqlite3.connect(dataset_path / "download" / f"table.db") as conn:
            for _, instance in tqdm.tqdm(instances.iterrows(), desc=f"create {partition} instances", total=len(instances.index)):
                if instance["original_table_id"] not in id2table.keys():
                    stats[f"ground_truth.skipped_because_table_filtered_out"] += 1
                    continue
                ground_truth = GroundTruth(
                    name=f"ground-truth-{partition}-{len(ground_truths)}",
                    lake_name=partition,
                    partition=partition,
                    information_need=instance["question"],
                    sql_query=instance["sql"],
                    data=instance.to_dict(),
                    answer_str=None,
                    answer_list=instance["answer"],
                    answer_table=None
                )

                table = id2table[instance["original_table_id"]]
                if apply_ground_truth(table, ground_truth, conn):
                    stats[f"ground_truth.num_suitable_ground_truths.{partition}"] += 1
                    ground_truths.append(ground_truth)
                else:
                    stats[f"ground_truth.num_unsuitable_ground_truths.{partition}"] += 1

        ground_truths = [cattrs.unstructure(ground_truth) for ground_truth in ground_truths]
        with open(partition_path / f"{lake.name}-ground-truth.json", "w", encoding="utf-8") as file:
            json.dump(ground_truths, file)

        if cfg.dataset.shuffle_rows_and_cols:
            for table in lake.get_all_tables():
                row_index = table.row_index.to_list()
                col_index = table.col_index.to_list()
                random.shuffle(col_index)
                random.shuffle(row_index)
                for tile in table.get_all_tiles():
                    new_row_index = pd.Index([row_index.index(idx) for idx in tile.row_index.to_list()])
                    tile._row_index = new_row_index
                data = table.data.loc[row_index, col_index]
                data.reset_index(drop=True, inplace=True)
                table._data = data

        lake.save(partition_path / f"{lake.name}.zip")

    with open(lakes_path / "stats.json", "w", encoding="utf-8") as file:
        json.dump(dict(stats), file)

    logger.info("Done!")
    logger.info(f"Statistics:\n{json.dumps(stats, indent=4, sort_keys=True)}")


def apply_ground_truth(
        table: Table,
        ground_truth: GroundTruth,
        connection: sqlite3.Connection
) -> bool:
    try:
        parse = sqlglot.parse_one(ground_truth.sql_query)
    except:
        logger.debug(f"Could not parse query with sqlglot: '{ground_truth.sql_query}'")
        stats["ground_truth.skipped_because_sqlglot_parsing_failed"] += 1
        return False
    logger.debug(f"Original query: {parse.sql()}")

    selects = parse.find_all(sqlglot.exp.Select)
    row_indices = []
    for select in selects:
        reduced_query_parts = ["SELECT rowid"]
        reduced_query_parts += [e.sql() for e in select.find_all(sqlglot.exp.From) if e.parent_select == select]
        reduced_query_parts += [e.sql() for e in select.find_all(sqlglot.exp.Where) if e.parent_select == select]
        reduced_query = " ".join(reduced_query_parts)
        logger.debug(f"Reduced query: {reduced_query}")

        try:
            res_table = connection.execute(reduced_query).fetchall()
        except:
            logger.debug(f"Could not execute reduced query: '{reduced_query}'")
            stats["ground_truth.skipped_because_could_not_execute_reduced_query_to_determine_rows"] += 1
            return False

        vals = [row[0] - 1 for row in res_table]  # sqlite row ids seem to start at 1, so we subtract 1
        row_index = pd.Index(vals)
        row_indices.append(row_index)

    # join row indices per table
    joined_index = row_indices[0]
    for index in row_indices[1:]:
        joined_index = joined_index.union(index)
    joined_index = table.row_index.intersection(joined_index)  # preserve row order

    if len(joined_index) == 0:
        stats["ground_truth.skipped_because_result_is_empty"] += 1
        return False

    try:
        parsed_query = sql_metadata.Parser(ground_truth.sql_query)
        logger.debug(f"Query: {ground_truth.sql_query}")
        logger.debug(f"Tables: {parsed_query.tables}")
        logger.debug(f"Columns: {parsed_query.columns}")
    except:
        stats["ground_truth.skipped_because_sql_metadata_parsing_failed"] += 1
        return False

    if parsed_query.columns is None:
        stats["ground_truth.skipped_because_columns_is_none"] += 1
        return False

    if parsed_query.columns_dict is None:
        stats["ground_truth.skipped_because_columns_dict_is_none"] += 1
        return False

    if "*" in parsed_query.columns:
        column_names = table.col_index.to_list()
    else:
        column_names = parsed_query.columns

    col_index = pd.Index(column_names)
    length_before = len(col_index)
    col_index = table.col_index.intersection(col_index)  # preserve column order
    assert len(col_index) == length_before, f"The ground truth tile column index contains fewer columns after intersection with table column index."
    table.create_tile(
        row_index=joined_index,
        col_index=col_index,
        tile_kind=f"ground-truth-{ground_truth.name}"
    )

    return True


if __name__ == "__main__":
    main()
