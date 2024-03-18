import collections
import io
import json
import logging
import os
import random

import attrs
import cattrs
import hydra
import pandas as pd
import tiktoken
import tqdm
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig

from lib.index import PerfectIndex
from lib.lake import DataLake, Tile
from lib.openai import openai_model, openai_execute
from lib.utils import get_data_path, GroundTruth, Accuracy, TileSet, compute_slicing_and_retrieval_confusion_matrix

logger = logging.getLogger(__name__)
stats = collections.Counter()


@attrs.define
class Config:
    defaults: list = [
        {"dataset": MISSING},
        {"table_embedder": MISSING},
        {"slicer": MISSING},
        {"tile_embedder": MISSING},
        {"query_embedder": MISSING},
        {"index": MISSING},
        {"merger": MISSING},
        "_self_"
    ]

    config_name: str = MISSING
    partition: str = MISSING

    use_cached_embedded_lakes: bool = True
    use_cached_sliced_lakes: bool = True
    use_cached_tile_embedded_lakes: bool = True

    skip_large_lakes: int | None = None
    limit_ground_truths: int | None = None

    silent: bool = False

    limit_tabular_context_tokens: int = MISSING
    limit_retrieved_tiles: int = 2000

    model: str = "gpt-4-1106-preview"
    max_tokens_over_ground_truth: int = 20
    temperature: int = 0


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    random.seed(309793332)

    logger.debug("Parse config.")
    config_parts = cfg.config_name.split("/")[-1].split("_")
    config_parts = {part.split("=")[0]: part.split("=")[1] for part in config_parts}
    logger.debug(f"Parameters: {config_parts}")

    logger.debug("Prepare directories.")
    eval_path = get_data_path() / cfg.dataset.name / "tqa" / cfg.partition
    os.makedirs(eval_path, exist_ok=True)

    lakes_path = get_data_path() / cfg.dataset.name / "lakes" / cfg.partition
    lake_paths = list(sorted(lakes_path.glob("*.zip")))

    logger.debug("Load components.")
    table_embedder = instantiate(cfg.table_embedder)
    slicer = instantiate(cfg.slicer)
    tile_embedder = instantiate(cfg.tile_embedder)
    query_embedder = instantiate(cfg.query_embedder)
    index = instantiate(cfg.index)
    merger = instantiate(cfg.merger)

    logger.debug("Prepare lakes.")
    lakes = []
    for lake_path in tqdm.tqdm(lake_paths, desc="prepare lakes"):
        embedded_lake_path = eval_path / "embedded_lakes" / cfg.table_embedder._target_ / lake_path.name
        sliced_lake_path = eval_path / "sliced_lakes" / f"{cfg.table_embedder._target_}_{cfg.slicer._target_}" / lake_path.name
        tile_embedded_lake_path = eval_path / "tile_embedded_lakes" / f"{cfg.table_embedder._target_}_{cfg.slicer._target_}_{cfg.tile_embedder._target_}" / lake_path.name

        if cfg.use_cached_tile_embedded_lakes and tile_embedded_lake_path.is_file():
            lake = DataLake.load(tile_embedded_lake_path, silent=cfg.silent)
        else:  # we need to embed the tiles
            if cfg.use_cached_sliced_lakes and sliced_lake_path.is_file():
                lake = DataLake.load(sliced_lake_path, silent=cfg.silent)
            else:  # we have to slice the tables
                if cfg.use_cached_embedded_lakes and embedded_lake_path.is_file():
                    lake = DataLake.load(embedded_lake_path, silent=cfg.silent)
                else:  # we have to embed the tables
                    lake = DataLake.load(lake_path, silent=cfg.silent)
                    if cfg.skip_large_lakes is not None and lake.num_cells > cfg.skip_large_lakes:
                        logger.warning(f"Skipped large lake '{lake.name}'.")
                        continue

                    table_embedder(lake, what_to_embed=["table-cells"], silent=cfg.silent)
                    for table in lake.get_all_tables():
                        table.table_name_embedding = None
                        table.table_name_embedding_by = None
                        table.col_names_embeddings = None
                        table.col_names_embeddings_by = None
                        table.cell_values_embeddings = None
                        table.cell_values_embedding_by = None
                    os.makedirs(embedded_lake_path.parent, exist_ok=True)
                    lake.save(embedded_lake_path, silent=cfg.silent)

                slicer(lake, silent=cfg.silent)
                for table in lake.get_all_tables():
                    table.cell_embeddings = None
                    table.cell_embeddings_by = None
                os.makedirs(sliced_lake_path.parent, exist_ok=True)
                lake.save(sliced_lake_path, silent=cfg.silent)

            tile_embedder(lake, what_to_embed=["tile"], tile_kinds=["slice-tile"], silent=cfg.silent)
            for table in lake.get_all_tables():
                table.table_name_embedding = None
                table.table_name_embedding_by = None
                table.col_names_embeddings = None
                table.col_names_embeddings_by = None
                table.cell_values_embeddings = None
                table.cell_values_embedding_by = None
            for tile in lake.get_all_tiles():
                tile.table_name_embedding = None
                tile.table_name_embedding_by = None
                tile.col_names_embeddings = None
                tile.col_names_embeddings_by = None
                tile.cell_values_embeddings = None
                tile.cell_values_embedding_by = None
            os.makedirs(tile_embedded_lake_path.parent, exist_ok=True)
            lake.save(tile_embedded_lake_path, silent=cfg.silent)

        lakes.append(lake)

    encoding = tiktoken.encoding_for_model(cfg.model)

    logger.debug("Prepare requests.")
    instances = []
    for lake_path, lake in zip(tqdm.tqdm(lake_paths, desc="prepare requests"), lakes):

        logger.debug("Shuffle and sample ground truths.")
        gt_path = lake_path.parent / f"{lake.name}-ground-truth.json"
        with open(gt_path, "r", encoding="utf-8") as file:
            ground_truths = [cattrs.structure(gt_data, GroundTruth) for gt_data in (json.load(file))]

        random.shuffle(ground_truths)
        if cfg.limit_ground_truths is not None:
            logger.warning(f"Limit to {cfg.limit_ground_truths} ground truths per lake.")
            ground_truths = ground_truths[:cfg.limit_ground_truths]

        for ground_truth in tqdm.tqdm(ground_truths, desc="prepare requests for lake", leave=False):
            query_embedding = query_embedder(ground_truth.information_need)
            if isinstance(index, PerfectIndex):
                retrieved_tiles = index(
                    lake,
                    query_embedding,
                    f"ground-truth-{ground_truth.name}",
                    tile_kinds=["slice-tile"],
                    threshold=None,
                    sort=True,
                    return_distances=False
                )
            else:
                retrieved_tiles = index(
                    lake,
                    query_embedding,
                    tile_kinds=["slice-tile"],
                    threshold=None,
                    sort=True,
                    return_distances=False
                )

            actual_retrieved_tiles = []
            actual_result_tiles = []
            actual_tabular_context = ""
            since_last_fitting_tile = 0
            for ix, retrieved_tile in enumerate(retrieved_tiles[:cfg.limit_retrieved_tiles]):
                actual_retrieved_tiles.append(retrieved_tile)
                result_tiles = merger(actual_retrieved_tiles)
                linearized_result_tiles = [linearize_tile(tile) for tile in result_tiles]
                tabular_context = "\n\n".join(linearized_result_tiles)
                l = len(encoding.encode(tabular_context))
                if l > cfg.limit_tabular_context_tokens:
                    if ix == 1:
                        actual_result_tiles = result_tiles
                        actual_tabular_context = encoding.decode(encoding.encode(tabular_context)[:cfg.limit_tabular_context_tokens])
                        logger.warning("Single actual retrieved tile has been split.")
                        stats["single_actual_retrieved_tile_has_been_split"] += 1
                        break
                    actual_retrieved_tiles = actual_retrieved_tiles[:-1]
                    since_last_fitting_tile += 1
                else:
                    since_last_fitting_tile = 0
                    actual_result_tiles = result_tiles
                    actual_tabular_context = tabular_context
            else:
                if since_last_fitting_tile <= 10:
                    logger.warning("One of the last ten retrieved tiles actually fit.")
                    stats["one_of_last_ten_retrieved_tiles_actually_fit"] += 1

            if len(actual_result_tiles) == 0:
                stats["no_actual_result_tiles"] += 1

            assert openai_model(cfg.model)["chat_or_completion"] == "chat", "Currently only supports chat models."

            if cfg.dataset.name == "openwikitable":
                linearized_answer = linearize_list(ground_truth.answer_list)
                system_message = "You are a chatbot that answers questions based on the provided context tables. Provide just the answer value and no explanation! If the answer comprises multiple values, state them as a CSV string."
                request_message = f"{ground_truth.information_need}\n\n{actual_tabular_context}"
            elif cfg.dataset.name == "spider":
                answer_table = pd.DataFrame(data=ground_truth.answer_table)
                linearized_answer = linearize_table(answer_table)
                answer_columns = ", ".join(answer_table.columns)
                system_message = "You are a chatbot that answers questions based on the provided context tables. Provide no explanations! State the answer as a CSV table! The column headers of the answer table are also provided below."
                request_message = f"{ground_truth.information_need}\n\nThe result table must have the following columns: {answer_columns}\n\nContext tables:\n\n{actual_tabular_context}"
            else:
                raise AssertionError(f"Invalid dataset: '{cfg.dataset.name}'!")
            max_tokens = len(encoding.encode(linearized_answer)) + cfg.max_tokens_over_ground_truth
            if openai_model(cfg.model)["max_output_tokens"] is not None:
                if max_tokens > openai_model(cfg.model)["max_output_tokens"]:
                    logger.warning("The models max_output_tokens is not enough to generate the ground truth output!")
                    max_tokens = openai_model(cfg.model)["max_output_tokens"]
            request = {
                "model": cfg.model,
                "max_tokens": max_tokens,
                "temperature": cfg.temperature,
                "messages": [
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": request_message
                    },
                ]
            }

            instances.append(
                Instance(
                    lake=lake,
                    ground_truth=ground_truth,
                    linearized_answer=linearized_answer,
                    retrieved_tiles=actual_retrieved_tiles,
                    result_tiles=actual_result_tiles,
                    linearized_result_tiles=actual_tabular_context,
                    request=request,
                    response=None,
                    prediction=None
                )
            )

    logger.debug("Execute requests.")
    for response, instance in zip(openai_execute([instance.request for instance in instances], force=0.1), instances):
        instance.response = response
        instance.prediction = response["choices"][0]["message"]["content"]

        if cfg.dataset.name == "spider":
            if instance.prediction.startswith("```csv"):
                instance.prediction = instance.prediction[6:-3]
            try:
                instance.prediction = linearize_table(pd.read_csv(io.StringIO(instance.prediction)))
            except:
                logger.warning("Failed to parse prediction with pandas!")
                stats["failed_to_parse_prediction_with_pandas"] += 1

    logger.debug("Evaluate responses.")
    results = []
    for instance in instances:
        gt_tile_set = TileSet.from_lake(instance.lake, tile_kind=f"ground-truth-{instance.ground_truth.name}")
        result_tile_set = TileSet(instance.result_tiles)

        acc = Accuracy.empty()
        acc.push(instance.linearized_answer == instance.prediction)
        confusion = compute_slicing_and_retrieval_confusion_matrix(instance.lake, result_tile_set, gt_tile_set)
        result = {
            "lake": instance.lake.name,
            "ground_truth": instance.ground_truth.name,
            "information_need": instance.ground_truth.information_need,
            "sql_query": instance.ground_truth.sql_query,
            "request": instance.request,
            "answer": instance.linearized_answer,
            "prediction": instance.prediction,
            "num_retrieved_tiles": len(instance.retrieved_tiles),
            "num_result_tiles": len(instance.result_tiles),
            "num_result_cells": result_tile_set.compute_num_cells(),
            "num_ground_truth_cells": gt_tile_set.compute_num_cells()
        }
        result.update(confusion.as_dict())
        result.update(acc.as_dict())
        results.append(result)

    results = pd.DataFrame(results)
    for k, v in config_parts.items():
        results[k] = v

    results_path = eval_path / "runs" / f"{cfg.config_name}.csv"
    os.makedirs(results_path.parent, exist_ok=True)
    results.to_csv(results_path)
    stats_path = results_path.parent / f"{cfg.config_name}_stats.json"
    with open(stats_path, "w", encoding="utf-8") as file:
        json.dump(dict(stats), file)

    logger.info("Done!")


@attrs.define
class Instance:
    lake: DataLake
    ground_truth: GroundTruth
    linearized_answer: str
    retrieved_tiles: list[Tile]
    result_tiles: list[Tile]
    linearized_result_tiles: str
    request: dict
    response: dict | None
    prediction: str | None


@attrs.define
class Result:
    lake: str
    ground_truth: str
    information_need: str
    sql_query: str
    request: dict
    answer: str
    prediction: str
    confusion: dict[str, float]
    accuracy: dict[str, float]
    num_retrieved_tiles: int
    num_result_tiles: int


def linearize_tile(tile: Tile) -> str:
    linearized_data = linearize_table(tile.data)
    return f"table: \"{tile.table.name}\"\n\n{linearized_data}\n"


def linearize_table(table: pd.DataFrame) -> str:
    return table.to_csv(index=False, header=True)


def linearize_list(l: list[str]) -> str:
    return ",".join(l)


if __name__ == "__main__":
    main()
