import abc
import logging

import attrs
import numpy as np
import omegaconf
import sentence_transformers
from hydra.core.config_store import ConfigStore

from lib.lake import DataLake, Table, Tile, LAKE_COMPONENTS
from lib.utils import get_models_path, tqdm_silent

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()


@attrs.define
class BaseEmbedderConfig(abc.ABC):
    _target_: str = omegaconf.MISSING
    preprocess_names: bool = True  # whether to preprocess table/column names before embedding them


@attrs.define
class BaseEmbedder(abc.ABC):
    """Compute embeddings for queries and data lake components."""
    preprocess_names: bool

    def __call__(
            self,
            value: DataLake | str,
            *,
            what_to_embed: list[str] | None = None,
            tile_kinds: list[str] | None = None,
            silent: bool = False
    ) -> np.ndarray | None:
        """Embed the query or components of the given data lake.

        Args:
            value: Query string or data lake to embed.
            what_to_embed: Which data lake components to embed (None means all).
            tile_kinds: Which tile kinds to embed (None means all).
            silent: Whether to disable progress bars.

        Returns:
            The query embedding or None.

        Raises:
            AssertionError if the given value is neither a query string nor a data lake.
            AssertionError if the specification of what to embed is invalid.
            AssertionError if trying to embed an empty table or tile.
        """
        if isinstance(value, DataLake):
            if what_to_embed is None:
                what_to_embed = LAKE_COMPONENTS

            what_to_embed = set(what_to_embed)
            if tile_kinds is not None:
                tile_kinds = list(set(tile_kinds))

            if any(x not in LAKE_COMPONENTS for x in what_to_embed):
                raise AssertionError(f"Invalid specification of which lake components to embed {what_to_embed}!")

            if any(x.startswith("table") for x in what_to_embed):
                for table in tqdm_silent(value.get_all_tables(), silent, desc="embed tables", leave=False):
                    if table.num_cells == 0:
                        raise AssertionError("Cannot compute embedding of empty table!")
                    self._handle_table(table, what_to_embed)

            if any(x.startswith("tile") for x in what_to_embed):
                for tile in tqdm_silent(value.get_all_tiles(tile_kind=tile_kinds), silent, desc="embed tiles", leave=False):
                    if tile.num_cells == 0:
                        raise AssertionError("Cannot compute embedding of empty tile!")
                    self._handle_tile(tile, what_to_embed)
        elif isinstance(value, str):
            return self._handle_query(value)
        else:
            raise AssertionError(f"Unable to embed {value}!")

    def _handle_table(self, table: Table, what_to_embed: set[str]) -> None:
        if "table-cells" in what_to_embed and table.cell_embeddings_by is not self:
            raise NotImplementedError("Embedding of table cells not implemented!")
        if "table-rows" in what_to_embed and table.row_embeddings_by is not self:
            raise NotImplementedError("Embedding of table rows not implemented!")
        if "table-cols" in what_to_embed and table.col_embeddings_by is not self:
            raise NotImplementedError("Embedding of table columns not implemented!")
        if "table" in what_to_embed and table.table_embedding_by is not self:
            raise NotImplementedError("Embedding of tables not implemented!")

    def _handle_tile(self, tile: Tile, what_to_embed: set[str]) -> None:
        if "tile-cells" in what_to_embed and tile.cell_embeddings_by is not self:
            raise NotImplementedError("Embedding of tile cells not implemented!")
        if "tile-rows" in what_to_embed and tile.row_embeddings_by is not self:
            raise NotImplementedError("Embedding of tile rows not implemented!")
        if "tile-cols" in what_to_embed and tile.col_embeddings_by is not self:
            raise NotImplementedError("Embedding of tile columns not implemented!")
        if "tile" in what_to_embed and tile.tile_embedding_by is not self:
            raise NotImplementedError("Embedding of tiles not implemented!")

    def _handle_query(self, query: str) -> np.ndarray:
        raise NotImplementedError("Embedding of queries not implemented!")

    @staticmethod
    def get_cell_values(table_or_tile: Table | Tile) -> list[str]:
        return table_or_tile.data.astype(str).values.flatten().tolist()

    def get_column_names(self, table_or_tile: Table | Tile) -> list[str]:
        return [self._preprocess_name(name) for name in table_or_tile.col_index.to_list()]

    def get_table_name(self, table_or_tile: Table | Tile) -> list[str]:
        if isinstance(table_or_tile, Table):
            name = table_or_tile.name
        else:
            name = table_or_tile.table.name
        return [self._preprocess_name(name)]

    def _preprocess_name(self, name: str) -> str:
        if self.preprocess_names:
            name = name.replace("-", " ")
            name = name.replace("_", " ")
        return name


@attrs.define
class NoEmbedderConfig(BaseEmbedderConfig):
    _target_: str = "lib.embed.NoEmbedder"


cs.store(name="no_embedder", node=NoEmbedderConfig, group="table_embedder")
cs.store(name="no_embedder", node=NoEmbedderConfig, group="tile_embedder")
cs.store(name="no_embedder", node=NoEmbedderConfig, group="query_embedder")


@attrs.define
class NoEmbedder(BaseEmbedder):
    """Set all embeddings to `np.zeros(...)`."""

    def _handle_table(self, table: Table, what_to_embed: set[str]) -> None:
        if "table-cells" in what_to_embed and table.cell_embeddings_by is not self:
            table.cell_embeddings = np.zeros((len(table.row_index), len(table.col_index), 10,))
            table.cell_embeddings_by = self
        if "table-rows" in what_to_embed and table.row_embeddings_by is not self:
            table.row_embeddings = np.zeros((len(table.row_index), 10,))
            table.row_embeddings_by = self
        if "table-cols" in what_to_embed and table.col_embeddings_by is not self:
            table.col_embeddings = np.zeros((len(table.col_index), 10,))
            table.col_embeddings_by = self
        if "table" in what_to_embed and table.table_embedding_by is not self:
            table.table_embedding = np.zeros((10,))
            table.table_embedding_by = self

    def _handle_tile(self, tile: Tile, what_to_embed: set[str]) -> None:
        if "tile-cells" in what_to_embed and tile.cell_embeddings_by is not self:
            tile.cell_embeddings = np.zeros((len(tile.row_index), len(tile.col_index), 10,))
            tile.cell_embeddings_by = self
        if "tile-rows" in what_to_embed and tile.row_embeddings_by is not self:
            tile.row_embeddings = np.zeros((len(tile.row_index), 10,))
            tile.row_embeddings_by = self
        if "tile-cols" in what_to_embed and tile.col_embeddings_by is not self:
            tile.col_embeddings = np.zeros((len(tile.col_index), 10,))
            tile.col_embeddings_by = self
        if "tile" in what_to_embed and tile.tile_embedding_by is not self:
            tile.tile_embedding = np.zeros((10,))
            tile.tile_embedding_by = self

    def _handle_query(self, query: str) -> np.ndarray:
        return np.zeros((10,))


def _load_sentence_transformer(model_identifier: str) -> sentence_transformers.SentenceTransformer:
    return sentence_transformers.SentenceTransformer(model_identifier, cache_folder=str(get_models_path()))


@attrs.define
class SBERTPartsEmbedderConfig(BaseEmbedderConfig):
    _target_: str = "lib.embed.SBERTPartsEmbedder"
    sentence_transformer: str = "multi-qa-mpnet-base-cos-v1"  # identifier of the sentence transformer model
    aggregation_weights: dict[str, dict[str, float]] = {  # how to compute embeddings for data lake components
        # "table": {"table-name": 2, "col-names": 1, "cell-values": 1},
        "table-rows": {"cell-values": 1},
        # "table-cols": {"col-names": 1, "cell-values": 1},
        "table-cells": {"col-names": 1, "cell-values": 1},
        "tile": {"table-name": 4, "col-names": 1, "cell-values": 1, "table-rows": 2},
        # "tile-rows": {"cell-values": 1},
        # "tile-cols": {"col-names": 1, "cell-values": 1},
        # "tile-cells": {"table-name": 1, "col-names": 1, "cell-values": 2}
    }


cs.store(name="sbertparts_embedder", node=SBERTPartsEmbedderConfig, group="table_embedder")
cs.store(name="sbertparts_embedder", node=SBERTPartsEmbedderConfig, group="tile_embedder")
cs.store(name="sbertparts_embedder", node=SBERTPartsEmbedderConfig, group="query_embedder")


@attrs.define
class SBERTPartsEmbedder(BaseEmbedder):
    """Compute embeddings for data lake components by aggregating embeddings of cell values, column names, and table name."""
    sentence_transformer: sentence_transformers.SentenceTransformer = attrs.field(converter=_load_sentence_transformer)
    aggregation_weights: dict[str, dict[str, float]]

    def _handle_table(self, table: Table, what_to_embed: set[str]) -> None:
        self._embed_parts(table, what_to_embed)

        if "table-cells" in what_to_embed and table.cell_embeddings_by is not self:
            weights = self.aggregation_weights["table-cells"]
            all_embeddings, all_weights = [], []
            if "cell-values" in weights.keys():
                all_embeddings.append(table.cell_values_embeddings)
                all_weights.append(weights["cell-values"])
            if "col-names" in weights.keys():
                all_embeddings.append(self._transform_cols_to_cells(table.col_names_embeddings, table))
                all_weights.append(weights["col-names"])
            if "table-name" in weights.keys():
                all_embeddings.append(self._transform_tab_to_cells(table.table_name_embedding, table))
                all_weights.append(weights["table-name"])
            table.cell_embeddings = self._aggregate_embeddings(all_embeddings, all_weights)
            table.cell_embeddings_by = self
        if "table-rows" in what_to_embed and table.row_embeddings_by is not self:
            weights = self.aggregation_weights["table-rows"]
            all_embeddings, all_weights = [], []
            if "cell-values" in weights.keys():
                all_embeddings.append(self._transform_cells_to_rows(table.cell_values_embeddings))
                all_weights.append(weights["cell-values"])
            if "col-names" in weights.keys():
                all_embeddings.append(self._transform_cols_to_rows(table.col_names_embeddings, table))
                all_weights.append(weights["col-names"])
            if "table-name" in weights.keys():
                all_embeddings.append(self._transform_tab_to_rows(table.table_name_embedding, table))
                all_weights.append(weights["table-name"])
            table.row_embeddings = self._aggregate_embeddings(all_embeddings, all_weights)
            table.row_embeddings_by = self
        if "table-cols" in what_to_embed and table.col_embeddings_by is not self:
            weights = self.aggregation_weights["table-cols"]
            all_embeddings, all_weights = [], []
            if "cell-values" in weights.keys():
                all_embeddings.append(self._transform_cells_to_cols(table.cell_values_embeddings))
                all_weights.append(weights["cell-values"])
            if "col-names" in weights.keys():
                all_embeddings.append(table.col_names_embeddings)
                all_weights.append(weights["col-names"])
            if "table-name" in weights.keys():
                all_embeddings.append(self._transform_tab_to_cols(table.table_name_embedding, table))
                all_weights.append(weights["table-name"])
            table.col_embeddings = self._aggregate_embeddings(all_embeddings, all_weights)
            table.col_embeddings_by = self
        if "table" in what_to_embed and table.table_embedding_by is not self:
            weights = self.aggregation_weights["table"]
            all_embeddings, all_weights = [], []
            if "cell-values" in weights.keys():
                all_embeddings.append(self._transform_cells_to_tab(table.cell_values_embeddings))
                all_weights.append(weights["cell-values"])
            if "col-names" in weights.keys():
                all_embeddings.append(self._transform_cols_to_tab(table.col_names_embeddings))
                all_weights.append(weights["col-names"])
            if "table-name" in weights.keys():
                all_embeddings.append(table.table_name_embedding)
                all_weights.append(weights["table-name"])
            table.table_embedding = self._aggregate_embeddings(all_embeddings, all_weights)
            table.table_embedding_by = self

    def _handle_tile(self, tile: Tile, what_to_embed: set[str]) -> None:
        table_parts_to_embed = set()
        for key in what_to_embed:
            for weight_key in self.aggregation_weights[key]:
                if weight_key in ("table", "table-rows", "table-cols", "table-cells"):
                    table_parts_to_embed.add(weight_key)
        if len(table_parts_to_embed) > 0:
            self._handle_table(tile.table, table_parts_to_embed)

        self._embed_parts(tile, what_to_embed)

        if "tile-cells" in what_to_embed and tile.cell_embeddings_by is not self:
            weights = self.aggregation_weights["tile-cells"]
            all_embeddings, all_weights = [], []
            if "cell-values" in weights.keys():
                all_embeddings.append(tile.cell_values_embeddings)
                all_weights.append(weights["cell-values"])
            if "col-names" in weights.keys():
                all_embeddings.append(self._transform_cols_to_cells(tile.col_names_embeddings, tile))
                all_weights.append(weights["col-names"])
            if "table-name" in weights.keys():
                all_embeddings.append(self._transform_tab_to_cells(tile.table_name_embedding, tile))
                all_weights.append(weights["table-name"])
            if "table" in weights.keys():
                all_embeddings.append(self._transform_tab_to_cells(tile.table.table_embedding, tile))
                all_weights.append(weights["table"])
            if "table-cells" in weights.keys():
                all_embeddings.append(tile.table.cell_embeddings[tile.row_coverage, tile.col_coverage, :])
                all_weights.append(weights["table-cells"])
            if "table-rows" in weights.keys():
                all_embeddings.append(self._transform_rows_to_cells(tile.table.row_embeddings[tile.row_coverage, :], tile))
                all_weights.append(weights["table-rows"])
            if "table-cols" in weights.keys():
                all_embeddings.append(self._transform_cols_to_cells(tile.table.col_embeddings[tile.col_coverage, :], tile))
                all_weights.append(weights["table-cols"])
            tile.cell_embeddings = self._aggregate_embeddings(all_embeddings, all_weights)
            tile.cell_embeddings_by = self
        if "tile-rows" in what_to_embed and tile.row_embeddings_by is not self:
            weights = self.aggregation_weights["tile-rows"]
            all_embeddings, all_weights = [], []
            if "cell-values" in weights.keys():
                all_embeddings.append(self._transform_cells_to_rows(tile.cell_values_embeddings))
                all_weights.append(weights["cell-values"])
            if "col-names" in weights.keys():
                all_embeddings.append(self._transform_cols_to_rows(tile.col_names_embeddings, tile))
                all_weights.append(weights["col-names"])
            if "table-name" in weights.keys():
                all_embeddings.append(self._transform_tab_to_rows(tile.table_name_embedding, tile))
                all_weights.append(weights["table-name"])
            if "table" in weights.keys():
                all_embeddings.append(self._transform_tab_to_rows(tile.table.table_embedding, tile))
                all_weights.append(weights["table"])
            if "table-cells" in weights.keys():
                all_embeddings.append(self._transform_cells_to_rows(tile.table.cell_embeddings[tile.row_coverage, tile.col_coverage, :]))
                all_weights.append(weights["table-cells"])
            if "table-rows" in weights.keys():
                all_embeddings.append(tile.table.row_embeddings[tile.row_coverage, :])
                all_weights.append(weights["table-rows"])
            if "table-cols" in weights.keys():
                all_embeddings.append(self._transform_cols_to_rows(tile.table.col_embeddings[tile.col_coverage, :], tile))
                all_weights.append(weights["table-cols"])
            tile.row_embeddings = self._aggregate_embeddings(all_embeddings, all_weights)
            tile.row_embeddings_by = self
        if "tile-cols" in what_to_embed and tile.col_embeddings_by is not self:
            weights = self.aggregation_weights["tile-cols"]
            all_embeddings, all_weights = [], []
            if "cell-values" in weights.keys():
                all_embeddings.append(self._transform_cells_to_cols(tile.cell_values_embeddings))
                all_weights.append(weights["cell-values"])
            if "col-names" in weights.keys():
                all_embeddings.append(tile.col_names_embeddings)
                all_weights.append(weights["col-names"])
            if "table-name" in weights.keys():
                all_embeddings.append(self._transform_tab_to_cols(tile.table_name_embedding, tile))
                all_weights.append(weights["table-name"])
            if "table" in weights.keys():
                all_embeddings.append(self._transform_tab_to_cols(tile.table.table_embedding, tile))
                all_weights.append(weights["table"])
            if "table-cells" in weights.keys():
                all_embeddings.append(self._transform_cells_to_cols(tile.table.cell_embeddings[tile.row_coverage, tile.col_coverage, :]))
                all_weights.append(weights["table-cells"])
            if "table-rows" in weights.keys():
                all_embeddings.append(self._transform_rows_to_cols(tile.table.row_embeddings[tile.row_coverage, :], tile))
                all_weights.append(weights["table-rows"])
            if "table-cols" in weights.keys():
                all_embeddings.append(tile.table.col_embeddings[tile.col_coverage, :])
                all_weights.append(weights["table-cols"])
            tile.col_embeddings = self._aggregate_embeddings(all_embeddings, all_weights)
            tile.col_embeddings_by = self
        if "tile" in what_to_embed and tile.tile_embedding_by is not self:
            weights = self.aggregation_weights["tile"]
            all_embeddings, all_weights = [], []
            if "cell-values" in weights.keys():
                all_embeddings.append(self._transform_cells_to_tab(tile.cell_values_embeddings))
                all_weights.append(weights["cell-values"])
            if "col-names" in weights.keys():
                all_embeddings.append(self._transform_cols_to_tab(tile.col_names_embeddings))
                all_weights.append(weights["col-names"])
            if "table-name" in weights.keys():
                all_embeddings.append(tile.table_name_embedding)
                all_weights.append(weights["table-name"])
            if "table" in weights.keys():
                all_embeddings.append(tile.table.table_embedding)
                all_weights.append(weights["table"])
            if "table-cells" in weights.keys():
                all_embeddings.append(self._transform_cells_to_tab(tile.table.cell_embeddings[tile.row_coverage, tile.col_coverage, :]))
                all_weights.append(weights["table-cells"])
            if "table-rows" in weights.keys():
                all_embeddings.append(self._transform_rows_to_tab(tile.table.row_embeddings[tile.row_coverage, :]))
                all_weights.append(weights["table-rows"])
            if "table-cols" in weights.keys():
                all_embeddings.append(self._transform_cols_to_tab(tile.table.col_embeddings[tile.col_coverage, :]))
                all_weights.append(weights["table-cols"])
            tile.tile_embedding = self._aggregate_embeddings(all_embeddings, all_weights)
            tile.tile_embedding_by = self

    def _handle_query(self, query: str) -> np.ndarray:
        return self.sentence_transformer.encode(query, show_progress_bar=False)

    def _embed_parts(self, table_or_tile: Table | Tile, what_to_embed: set[str]) -> None:
        parts_to_embed = set()
        for key in what_to_embed:
            for weight_key in self.aggregation_weights[key]:
                parts_to_embed.add(weight_key)

        parts = {}
        if "cell-values" in parts_to_embed and table_or_tile.cell_values_embedding_by is not self:
            parts["cell-values"] = self.get_cell_values(table_or_tile)
        if "col-names" in parts_to_embed and table_or_tile.col_embeddings_by is not self:
            parts["col-names"] = self.get_column_names(table_or_tile)
        if "table-name" in parts_to_embed and table_or_tile.table_name_embedding_by is not self:
            parts["table-name"] = self.get_table_name(table_or_tile)

        current = 0
        part_indices = {}
        part_strings = []
        for part_key, part_string in parts.items():
            part_strings += part_string
            part_indices[part_key] = (current, current + len(part_string))
            current += len(part_string)

        all_embeddings = self.sentence_transformer.encode(part_strings, show_progress_bar=False)

        part_embeddings = {}
        for part_key, (left, right) in part_indices.items():
            part_embeddings[part_key] = all_embeddings[left:right]

        num_rows, num_cols = len(table_or_tile.row_index), len(table_or_tile.col_index)
        dim = self.sentence_transformer.get_sentence_embedding_dimension()
        if "cell-values" in part_embeddings.keys():
            part_embeddings["cell-values"] = np.reshape(part_embeddings["cell-values"], (num_cols, num_rows, dim))
            part_embeddings["cell-values"] = np.swapaxes(part_embeddings["cell-values"], 0, 1)
            table_or_tile.cell_values_embeddings = part_embeddings["cell-values"]
            table_or_tile.cell_values_embedding_by = self
        if "col-names" in part_embeddings.keys():
            table_or_tile.col_names_embeddings = part_embeddings["col-names"].reshape((num_cols, dim))
            table_or_tile.col_names_embedding_by = self
        if "table-name" in part_embeddings.keys():
            table_or_tile.table_name_embedding = part_embeddings["table-name"].reshape((dim,))
            table_or_tile.table_name_embedding_by = self

    @staticmethod
    def _aggregate_embeddings(embeddings: list[np.ndarray], weights: list[float]) -> np.ndarray:
        if embeddings == []:
            raise ValueError("Cannot aggregate empty list of embeddings!")

        if len(embeddings) == 1:
            return embeddings[0]

        embeddings = np.stack(embeddings)
        weights = np.stack(weights)

        weights_shape = (weights.shape[0],) + (1,) * (embeddings.ndim - 1)
        embeddings = np.multiply(embeddings, weights.reshape(weights_shape))  # TODO: make sure this works
        normalizer = np.sum(weights)
        return np.divide(np.sum(embeddings, axis=0), normalizer)

    @staticmethod
    def _transform_cells_to_cols(embedding: np.ndarray) -> np.ndarray:
        return np.mean(embedding, axis=0)

    @staticmethod
    def _transform_cells_to_rows(embedding: np.ndarray) -> np.ndarray:
        return np.mean(embedding, axis=1)

    @staticmethod
    def _transform_cells_to_tab(embedding: np.ndarray) -> np.ndarray:
        embedding = np.mean(embedding, axis=0)
        return np.mean(embedding, axis=0)

    @staticmethod
    def _transform_cols_to_tab(embedding: np.ndarray) -> np.ndarray:
        return np.mean(embedding, axis=0)

    def _transform_cols_to_cells(self, embedding: np.ndarray, table_or_tile: Table | Tile) -> np.ndarray:
        dim = self.sentence_transformer.get_sentence_embedding_dimension()
        return np.broadcast_to(embedding, (len(table_or_tile.row_index), len(table_or_tile.col_index), dim))

    def _transform_cols_to_rows(self, embedding: np.ndarray, table_or_tile: Table | Tile) -> np.ndarray:
        dim = self.sentence_transformer.get_sentence_embedding_dimension()
        embedding = np.mean(embedding, axis=0)
        return np.broadcast_to(embedding, (len(table_or_tile.row_index), dim))

    @staticmethod
    def _transform_rows_to_tab(embedding: np.ndarray) -> np.ndarray:
        return np.mean(embedding, axis=0)

    def _transform_rows_to_cols(self, embedding: np.ndarray, table_or_tile: Table | Tile) -> np.ndarray:
        dim = self.sentence_transformer.get_sentence_embedding_dimension()
        embedding = np.mean(embedding, 0)
        embedding = np.broadcast_to(embedding, (len(table_or_tile.col_index), dim))
        return embedding

    def _transform_rows_to_cells(self, embedding: np.ndarray, table_or_tile: Table | Tile) -> np.ndarray:
        dim = self.sentence_transformer.get_sentence_embedding_dimension()
        embedding = np.broadcast_to(embedding, (len(table_or_tile.col_index), len(table_or_tile.row_index), dim))
        embedding = np.swapaxes(embedding, 0, 1)
        return embedding

    def _transform_tab_to_cells(self, embedding: np.ndarray, table_or_tile: Table | Tile) -> np.ndarray:
        dim = self.sentence_transformer.get_sentence_embedding_dimension()
        return np.broadcast_to(embedding, (len(table_or_tile.row_index), len(table_or_tile.col_index), dim))

    def _transform_tab_to_cols(self, embedding: np.ndarray, table_or_tile: Table | Tile) -> np.ndarray:
        dim = self.sentence_transformer.get_sentence_embedding_dimension()
        return np.broadcast_to(embedding, (len(table_or_tile.col_index), dim))

    def _transform_tab_to_rows(self, embedding: np.ndarray, table_or_tile: Table | Tile) -> np.ndarray:
        dim = self.sentence_transformer.get_sentence_embedding_dimension()
        return np.broadcast_to(embedding, (len(table_or_tile.row_index), dim))
