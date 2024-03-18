import collections
import json
import logging
import os
import pathlib
import random
import sqlite3
from typing import Any

import attrs
import cattrs
import hydra
import pandas as pd
import sql_metadata
import sqlglot
import sqlglot.executor
import tqdm
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from lib.lake import DataLake
from lib.utils import GroundTruth, get_data_path

logger = logging.getLogger(__name__)
stats = collections.Counter()


@attrs.define
class CreateSpiderLakesConfig:
    defaults: list = [{"dataset": "spider"}, "_self_"]


ConfigStore.instance().store(name="create_spider_lakes", node=CreateSpiderLakesConfig)


@hydra.main(version_base=None, config_path="../../config", config_name="create_spider_lakes")
def create_spider_lakes(cfg: DictConfig) -> None:
    logger.info("Create Spider Data Lakes!")

    random.seed(58394737)

    dataset_path = get_data_path() / cfg.dataset.name
    lakes_path = dataset_path / "lakes"
    os.makedirs(lakes_path)

    logger.info("Load ground truths.")
    ground_truths = load_ground_truths(cfg)
    stats["original.num_ground_truths"] = sum(len(gts) for gts in ground_truths.values())

    logger.info("Glob databases.")
    database_paths = list(sorted(dataset_path.joinpath("download").joinpath("database").glob("*/*.sqlite")))
    stats["original.num_databases"] = len(database_paths)
    random.shuffle(database_paths)  # so that the validation set is selected at random

    num_val_lakes = 0
    for database_path in tqdm.tqdm(database_paths, desc="create lakes"):
        lake = create_lake(database_path, cfg)
        if lake is None:
            stats["lake.num_unsuitable_lakes"] += 1
            continue

        suitable_ground_truths = []
        with sqlite3.connect(database_path) as conn:
            conn.row_factory = sqlite3.Row
            for ground_truth in ground_truths[lake.name]:
                if ground_truth.partition == "train-or-val":
                    if num_val_lakes < cfg.dataset.num_val_lakes:
                        ground_truth.partition = "val"
                    else:
                        ground_truth.partition = "train"
                if apply_ground_truth(lake, ground_truth, conn, cfg):
                    stats["ground_truth.num_suitable_ground_truths"] += 1
                    suitable_ground_truths.append(ground_truth)
                else:
                    stats["ground_truth.num_unsuitable_ground_truths"] += 1

            if len(suitable_ground_truths) == 0:
                stats["lake.skipped_because_no_ground_truth"] += 1
                continue

            stats["lake.num_suitable_lakes"] += 1

            assert len(set(ground_truth.partition for ground_truth in suitable_ground_truths)) == 1
            partition = suitable_ground_truths[0].partition
            stats[f"lake.num_suitable_lakes.{partition}"] += 1
            if partition == "val":
                num_val_lakes += 1

            partition_path = lakes_path / partition
            os.makedirs(partition_path, exist_ok=True)
            lake.save(partition_path / f"{lake.name}.zip")

            suitable_ground_truths = [cattrs.unstructure(ground_truth) for ground_truth in suitable_ground_truths]
            with open(partition_path / f"{lake.name}-ground-truth.json", "w", encoding="utf-8") as file:
                json.dump(suitable_ground_truths, file)

    with open(lakes_path / "stats.json", "w", encoding="utf-8") as file:
        json.dump(dict(stats), file)

    logger.info(f"Statistics:\n{json.dumps(stats, indent=4, sort_keys=True)}")
    logger.info("Done!")


def load_ground_truths(cfg: DictConfig) -> dict[str, list[GroundTruth]]:
    spider_path = get_data_path() / cfg.dataset.name / "download"

    with open(spider_path / "train_spider.json", "r", encoding="utf-8") as file:
        train_data = json.load(file)

    with open(spider_path / "train_others.json", "r", encoding="utf-8") as file:
        train_data += json.load(file)

    ground_truths = collections.defaultdict(list)
    for ix, spider_data in enumerate(train_data):
        ground_truth = GroundTruth(
            name=f"ground-truth-{ix}",
            lake_name=spider_data["db_id"],
            partition="train-or-val",
            information_need=spider_data["question"],
            sql_query=spider_data["query"],
            data=spider_data,
            answer_str=None,
            answer_list=None,
            answer_table=None
        )
        ground_truths[ground_truth.lake_name].append(ground_truth)

    with open(spider_path / "dev.json", "r", encoding="utf-8") as file:
        dev_data = json.load(file)

    for ix, spider_data in enumerate(dev_data):
        ground_truth = GroundTruth(
            name=f"ground-truth-{ix}",
            lake_name=spider_data["db_id"],
            partition="test",
            information_need=spider_data["question"],
            sql_query=spider_data["query"],
            data=spider_data,
            answer_str=None,
            answer_list=None,
            answer_table=None
        )
        ground_truths[ground_truth.lake_name].append(ground_truth)

    return ground_truths


def create_lake(database_path: pathlib.Path, cfg: DictConfig) -> DataLake | None:
    lake_name = database_path.name[:database_path.name.rindex(".")]
    lake = DataLake.create_from_sqlite(lake_name, database_path)

    for table in lake.get_all_tables():
        if table.num_cells == 0:
            stats["lake.skipped_because_empty_table"] += 1
            return None

    if cfg.dataset.shuffle_rows_and_cols:
        for table in lake.get_all_tables():
            row_index = table.row_index.to_list()
            col_index = table.col_index.to_list()
            random.shuffle(row_index)
            random.shuffle(col_index)
            data = table.data.loc[row_index, col_index]
            data.reset_index(drop=True, inplace=True)
            table._data = data

    return lake


def apply_ground_truth(lake: DataLake, ground_truth: GroundTruth, conn: sqlite3.Connection, cfg: DictConfig) -> bool:
    try:
        ground_truth.answer_table = collections.defaultdict(list)
        for row in conn.execute(ground_truth.sql_query).fetchall():
            row: sqlite3.Row = row
            for key in row.keys():
                ground_truth.answer_table[key].append(row[key])
        ground_truth.answer_table = dict(ground_truth.answer_table)

    except:
        logger.debug(f"Could not execute original SQL query on sqlite: '{ground_truth.sql_query}'")
        stats["ground_truth.skipped_because_could_not_execute_original_query_to_determine_answer"] += 1
        return False

    if "*" in ground_truth.sql_query:
        stats["ground_truth.skipped_because_query_contains_*"] += 1
        return False

    try:
        parsed_query = sql_metadata.Parser(ground_truth.sql_query)
        logger.debug(f"Query: {ground_truth.data['query']}")
        logger.debug(f"Tables: {parsed_query.tables}")
        logger.debug(f"Columns: {parsed_query.columns}")
    except:
        stats["ground_truth.skipped_ground_truth_because_parsing_failed"] += 1
        return False

    if parsed_query.columns is None:
        stats["ground_truth.skipped_because_columns_is_none"] += 1
        return False

    if "*" in parsed_query.columns:
        stats["ground_truth.skipped_because_columns_contain_*"] += 1
        return False

    if parsed_query.columns_dict is None:
        stats["ground_truth.skipped_because_columns_dict_is_none"] += 1
        return False

    table_names = resolve_table_names(parsed_query, lake)
    if table_names is None:
        stats["ground_truth.skipped_because_resolving_table_names_failed"] += 1
        return False

    res = resolve_column_names(parsed_query, table_names, lake, cfg)
    if res is None:
        stats["ground_truth.skipped_because_resolving_column_names_failed"] += 1
        return False
    column_names_clean, column_names_dirty = res

    lake_dict = lake_to_dict(lake)

    parse = sqlglot.parse_one(ground_truth.sql_query)
    logger.debug(f"Original query: {parse.sql()}")

    selects = parse.find_all(sqlglot.exp.Select)
    row_indices = collections.defaultdict(list)
    for select in selects:
        reduced_query_parts = ["SELECT *"]
        reduced_query_parts += [e.sql() for e in select.find_all(sqlglot.exp.From) if e.parent_select == select]
        reduced_query_parts += [e.sql() for e in select.find_all(sqlglot.exp.Where) if e.parent_select == select]
        reduced_query = " ".join(reduced_query_parts)
        logger.debug(f"Reduced query: {reduced_query}")

        try:
            res_table = sqlglot.executor.execute(reduced_query, tables=lake_dict)
        except:
            logger.debug(f"Could not execute reduced query: '{reduced_query}'")
            stats["ground_truth.determine_rows.could_not_execute_reduced_query"] += 1
            stats["ground_truth.skipped_because_could_not_determine_rows"] += 1
            return False

        for col_idx, col_name in enumerate(res_table.columns):
            if col_name.startswith("index_a8d938f938e2_"):
                table_name = col_name[19:]
                for lake_table_name, lake_table in lake.tables.items():
                    if identifier_eq(lake_table_name, table_name):
                        break
                else:
                    logger.debug(f"Could not find lake table for query table '{table_name}'")
                    stats["ground_truth.determine_rows.no_matching_lake_table"] += 1
                    stats["ground_truth.skipped_because_could_not_determine_rows"] += 1
                    return False

                vals = [t[col_idx] for t in res_table.rows]
                row_index = pd.Index(vals)
                row_index = lake_table.row_index.intersection(row_index)  # preserve row order
                row_indices[lake_table_name].append(row_index)

    # join row indices
    joined_row_indices = {}
    for table_name, indices in row_indices.items():
        joined_index = indices[0]
        for index in indices[1:]:
            joined_index = joined_index.union(index)
        joined_row_indices[table_name] = joined_index
    row_indices = joined_row_indices

    # group by table
    table_column_names = collections.defaultdict(list)
    for table_name, column_name in column_names_clean.values():
        table_column_names[table_name].append(column_name)

    for table_name in table_column_names.keys():
        if table_name not in row_indices.keys():
            logger.debug(f"No row index for table '{table_name}'")
            stats["ground_truth.skipped_because_no_row_index"] += 1
            return False

    for table_name, column_names in table_column_names.items():
        table = lake.tables[table_name]
        table.create_tile(
            row_index=row_indices[table_name],
            col_index=pd.Index(column_names),
            tile_kind=f"ground-truth-{ground_truth.name}"
        )

    return True


def lake_to_dict(lake: DataLake) -> dict[str, list[dict[str, Any]]]:
    res = {}
    for table_name, table in lake.tables.items():
        res_table = []
        for row_idx, row in table.data.iterrows():
            res_tuple = {f"index_a8d938f938e2_{table_name}": row_idx} | row.to_dict()
            res_table.append(res_tuple)
        res[table_name] = res_table
    return res


def resolve_table_names(
        parsed_query: sql_metadata.Parser,
        lake: DataLake
) -> dict[str, str] | None:
    # query table name ==> lake table name
    res = {}

    for table_name in set(parsed_query.tables):
        for lake_table_name in lake.tables.keys():
            if identifier_eq(lake_table_name, table_name):
                if table_name in res.keys():
                    logger.debug(f"Found multiple matching lake tables for query table '{table_name}'")
                    stats["ground_truth.resolve_table_names.multiple_matching_lake_tables"] += 1
                    return None
                res[table_name] = lake_table_name

        if table_name not in res.keys():
            logger.debug(f"Found no matching lake table for query table '{table_name}'")
            stats["ground_truth.resolve_table_names.no_matching_lake_table"] += 1
            return None

    return res


def resolve_column_names(
        parsed_query: sql_metadata.Parser,
        table_names: dict[str, str],
        lake: DataLake,
        cfg: DictConfig
) -> tuple[dict[tuple[str | None, str], tuple[str, str]], dict[tuple[str | None, str], tuple[str, str]]] | None:
    # (lake table name, query column name) ==> (lake table name, lake column name)
    # (query table name, query column name) ==> (lake table name, lake column name)
    res_clean_table_names = {}
    res_dirty_table_names = {}

    for column_name in set(parsed_query.columns):
        if "." in column_name:
            # assume the column name already includes the table name, search only this table
            parts = column_name.split(".")
            if len(parts) != 2:
                logger.debug(f"Could not split column name: '{column_name}'")
                stats["ground_truth.resolve_column_names.could_not_split_column_name"] += 1
                return None
            table_name, column_name = parts

            if table_name not in table_names.keys():
                logger.debug(f"Could not find split column's table name: '{table_name}'")
                stats["ground_truth.resolve_column_names.no_matching_lake_table_for_split_column_table_name"] += 1
                return None

            table = lake.tables[table_names[table_name]]
            for lake_column_name in table.col_index:
                if identifier_eq(column_name, lake_column_name):
                    # find duplicates based on query table name and query column name
                    if (table_name, column_name) in res_dirty_table_names.keys():
                        logger.debug(
                            f"Found multiple matching columns for query column '{column_name}' in table '{table_name}'"
                        )
                        stats["ground_truth.resolve_column_names.multiple_matching_lake_columns_for_split_column_table_name"] += 1
                        return None
                    res_dirty_table_names[(table_name, column_name)] = (table_names[table_name], lake_column_name)
                    res_clean_table_names[(table_names[table_name], column_name)] = (
                        table_names[table_name], lake_column_name)
            if (table_name, column_name) not in res_dirty_table_names.keys():
                logger.debug(f"Found no matching column for query column '{column_name}' in table '{table_name}'")
                stats["ground_truth.resolve_column_names.no_matching_lake_column_for_split_table_column_name"] += 1
                return None
        else:
            # search for the column in all tables of the query
            for lake_table_name in table_names.values():
                table = lake.tables[lake_table_name]
                for lake_column_name in table.col_index:
                    if identifier_eq(column_name, lake_column_name):
                        # find duplicates based on query column name
                        if (None, column_name) in res_dirty_table_names.keys():
                            logger.debug(f"Found multiple matching columns for query column '{column_name}'")
                            stats["ground_truth.resolve_column_names.multiple_matching_lake_columns"] += 1
                            return None
                        res_dirty_table_names[(None, column_name)] = (lake_table_name, lake_column_name)
                        res_clean_table_names[(lake_table_name, column_name)] = (lake_table_name, lake_column_name)
            if (None, column_name) not in res_dirty_table_names.keys():
                logger.debug(f"Found no matching column for query column '{column_name}'")
                stats["ground_truth.resolve_column_names.no_matching_lake_column"] += 1
                return None

    return res_clean_table_names, res_dirty_table_names


def identifier_eq(a: str, b: str) -> bool:
    return b.lower() == a.lower()


if __name__ == "__main__":
    create_spider_lakes()
