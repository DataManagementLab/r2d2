import abc
import heapq
import logging
import math

import attrs
import numpy as np
import omegaconf
import pandas as pd
import sklearn
from hydra.core.config_store import ConfigStore
from sklearn.cluster import AgglomerativeClustering

from lib.lake import DataLake, Table
from lib.utils import tqdm_silent

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()


@attrs.define
class BaseSlicerConfig(abc.ABC):
    _target_: str = omegaconf.MISSING
    tiles_per_table: int = 10  # average number of tiles per table
    tile_distribution: str = "cell-based"  # "cell-based" or "table-based" distribution of tiles per table


@attrs.define
class BaseSlicer(abc.ABC):
    """Slice tables into tiles."""
    tiles_per_table: int | None
    tile_distribution: str

    def _compute_tiles_for_table(self, table: Table, lake: DataLake) -> int:
        if self.tile_distribution == "table-based":
            return self.tiles_per_table
        elif self.tile_distribution == "cell-based":
            return 1 + int((self.tiles_per_table - 1) * len(lake.tables) * table.num_cells / lake.num_cells)
        else:
            raise AssertionError(f"Invalid tile distribution mode!")

    def __call__(self, lake: DataLake, *, silent: bool = False) -> None:
        """Slice the data lake's tables into tiles.

        Args:
            lake: A data lake with tables to slice.
            silent: Whether to disable progress bars.
        """
        for table in tqdm_silent(lake.get_all_tables(), silent, desc="slice tables", leave=False):
            if table.num_cells == 0:
                raise AssertionError("Cannot slice empty table!")
            tiles_per_table = self._compute_tiles_for_table(table, lake)
            self._slice_table(table, tiles_per_table)

    @abc.abstractmethod
    def _slice_table(self, table: Table, tiles_for_table: int) -> None:
        """Slice the given table.

        This method is implemented by the subclasses of BaseSlicer.

        Args:
            table: The table to slice.
            tiles_for_table: The number of tiles to create from the table.
        """
        raise NotImplementedError


@attrs.define
class NoSlicerConfig(BaseSlicerConfig):
    _target_: str = "lib.slice.NoSlicer"


cs.store(name="no_slicer", node=NoSlicerConfig, group="slicer")


@attrs.define
class NoSlicer(BaseSlicer):
    """Create one tile per table. Ignores `tiles_per_table`."""

    def _slice_table(self, table: Table, tiles_for_table: int) -> None:
        table.create_tile(table.row_index.copy(), table.col_index.copy(), "slice-tile")


@attrs.define
class CellSlicerConfig(BaseSlicerConfig):
    _target_: str = "lib.slice.CellSlicer"


cs.store(name="cell_slicer", node=CellSlicerConfig, group="slicer")


@attrs.define
class CellSlicer(BaseSlicer):
    """Create one tile per cell. Ignores `tiles_per_table`."""

    def _slice_table(self, table: Table, tiles_for_table: int) -> None:
        for row in table.row_index:
            for col in table.col_index:
                table.create_tile(pd.Index([row]), pd.Index([col]), "slice-tile")


@attrs.define
class RowWiseSlicerConfig(BaseSlicerConfig):
    _target_: str = "lib.slice.RowWiseSlicer"


cs.store(name="rowwise_slicer", node=RowWiseSlicerConfig, group="slicer")


@attrs.define
class RowWiseSlicer(BaseSlicer):
    """Create row-wise tiles."""

    def _slice_table(self, table: Table, tiles_for_table: int) -> None:
        rows_per_tile = math.ceil(len(table.row_index) / tiles_for_table)

        for row_idx in range(0, len(table.row_index), rows_per_tile):
            row_index = table.row_index.copy()[row_idx:row_idx + rows_per_tile]
            col_index = table.col_index.copy()
            table.create_tile(row_index, col_index, "slice-tile")


@attrs.define
class ColWiseSlicerConfig(BaseSlicerConfig):
    _target_: str = "lib.slice.ColWiseSlicer"


cs.store(name="colwise_slicer", node=ColWiseSlicerConfig, group="slicer")


@attrs.define
class ColWiseSlicer(BaseSlicer):
    """Create column-wise tiles."""

    def _slice_table(self, table: Table, tiles_for_table: int) -> None:
        cols_per_tile = math.ceil(len(table.col_index) / tiles_for_table)

        for col_idx in range(0, len(table.col_index), cols_per_tile):
            row_index = table.row_index.copy()
            col_index = table.col_index.copy()[col_idx:col_idx + cols_per_tile]
            table.create_tile(row_index, col_index, "slice-tile")


class _Pair:
    row_index: np.ndarray
    col_index: np.ndarray
    cell_embeddings: np.ndarray
    row_embeddings: np.ndarray
    col_embeddings: np.ndarray
    row_distance: float
    col_distance: float
    row_penalty_base: float | None
    col_penalty_base: float | None

    def __init__(self, row_index: np.ndarray, col_index: np.ndarray, cell_embeddings: np.ndarray, row_penalty_base: float | None, col_penalty_base: float | None) -> None:
        self.row_index = row_index
        self.col_index = col_index
        self.cell_embeddings = cell_embeddings
        self.row_penalty_base = row_penalty_base
        self.col_penalty_base = col_penalty_base

        selected_cell_embeddings = cell_embeddings[self.row_index][:, self.col_index]  # get tile
        self.row_embeddings = np.mean(selected_cell_embeddings, axis=1)
        self.col_embeddings = np.mean(selected_cell_embeddings, axis=0)

        row_distances = sklearn.metrics.pairwise_distances(self.row_embeddings, metric="euclidean")
        col_distances = sklearn.metrics.pairwise_distances(self.col_embeddings, metric="euclidean")
        row_penalty = 0 if self.row_penalty_base is None else math.log(self.row_embeddings.shape[0], self.row_penalty_base)
        col_penalty = 0 if self.col_penalty_base is None else math.log(self.col_embeddings.shape[0], self.col_penalty_base)
        self.row_distance = float(np.max(np.abs(row_distances))) * (1 + row_penalty)
        self.col_distance = float(np.max(np.abs(col_distances))) * (1 + col_penalty)

    @property
    def split_distance(self) -> float:
        return max(self.row_distance, self.col_distance)

    @property
    def split_embeddings(self) -> np.ndarray:
        if self.row_distance > self.col_distance:
            return self.row_embeddings
        else:
            return self.col_embeddings

    @property
    def split_index(self) -> np.ndarray:
        if self.row_distance > self.col_distance:
            return self.row_index
        else:
            return self.col_index

    def after_split(self, split_index) -> "_Pair":
        if self.row_distance > self.col_distance:
            return _Pair(split_index, self.col_index, self.cell_embeddings, self.row_penalty_base, self.col_penalty_base)
        else:
            return _Pair(self.row_index, split_index, self.cell_embeddings, self.row_penalty_base, self.col_penalty_base)

    def __lt__(self, other):
        return self.split_distance < other.split_distance


@attrs.define
class DivisiveSlicerConfig(BaseSlicerConfig):
    _target_: str = "lib.slice.DivisiveSlicer"
    clusters_per_step: int = 2  # number of tiles that should result from splitting a tile
    row_penalty_base: float | None = None
    col_penalty_base: float | None = None


cs.store(name="divisive_slicer", node=DivisiveSlicerConfig, group="slicer")


@attrs.define
class DivisiveSlicer(BaseSlicer):
    clusters_per_step: int
    row_penalty_base: float | None
    col_penalty_base: float | None

    def _slice_table(self, table: Table, tiles_for_table: int) -> None:
        initial_cell_embeddings = table.cell_embeddings

        l2_norms = np.linalg.norm(initial_cell_embeddings, axis=2, keepdims=True)
        initial_cell_embeddings = initial_cell_embeddings / l2_norms

        initial_pair = _Pair(
            row_index=np.arange(0, initial_cell_embeddings.shape[0]),
            col_index=np.arange(0, initial_cell_embeddings.shape[1]),
            cell_embeddings=initial_cell_embeddings,
            row_penalty_base=self.row_penalty_base,
            col_penalty_base=self.col_penalty_base
        )
        pairs_heap = [(-initial_pair.split_distance, initial_pair)]  # heap has smallest first
        finished_pairs = []

        while True:
            if len(pairs_heap) + len(finished_pairs) + self.clusters_per_step - 1 > tiles_for_table:
                logger.debug("Stopped slicing due to reaching the maximum number of tiles per table.")
                break
            if len(pairs_heap) == 0:
                logger.debug("Stopped slicing due to no more pairs to slice.")
                break

            _, pair = heapq.heappop(pairs_heap)

            if self.clusters_per_step > pair.split_embeddings.shape[0]:
                # skip pair because it cannot be split further
                finished_pairs.append(pair)
                continue

            clustering = AgglomerativeClustering(n_clusters=self.clusters_per_step, metric="euclidean")
            cluster_labels = clustering.fit_predict(pair.split_embeddings)

            new_pairs = []
            for idx in range(self.clusters_per_step):
                split_index = pair.split_index[cluster_labels == idx]
                if len(split_index) > 0:  # some clustering algorithms can create empty clusters
                    new_pair = pair.after_split(split_index)
                    new_pairs.append(new_pair)

            if len(new_pairs) == 1:
                # this prevents an infinite loop when the clustering algorithm does not split a tile
                finished_pairs.append(new_pairs[0])
            else:
                for new_pair in new_pairs:
                    heapq.heappush(pairs_heap, (-new_pair.split_distance, new_pair))

        all_pairs = [pair for _, pair in pairs_heap] + finished_pairs
        for pair in all_pairs:
            row_index = pd.Index(table.row_index.to_numpy()[pair.row_index]).copy()
            col_index = pd.Index(table.col_index.to_numpy()[pair.col_index]).copy()
            table.create_tile(row_index, col_index, "slice-tile")
