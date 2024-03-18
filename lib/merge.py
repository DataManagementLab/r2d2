import abc
import collections
import logging

import attrs
import omegaconf
from hydra.core.config_store import ConfigStore

from lib.lake import Tile

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()


@attrs.define
class BaseMergerConfig(abc.ABC):
    _target_: str = omegaconf.MISSING


@attrs.define
class BaseMerger(abc.ABC):
    """Compose the retrieved tiles to construct a result."""

    @abc.abstractmethod
    def __call__(self, tiles: list[Tile], *, silent: bool = False) -> list[Tile]:
        """Compose the given tiles to construct a result.

        Args:
            tiles: A list of Tiles to merge.
            silent: Whether to disable progress bars.
        """
        raise NotImplementedError()


@attrs.define
class NoMergerConfig(BaseMergerConfig):
    _target_: str = "lib.merge.NoMerger"


cs.store(name="no_merger", node=NoMergerConfig, group="merger")


@attrs.define
class NoMerger(BaseMerger):
    """Do not merge any tiles."""

    def __call__(self, tiles: list[Tile], *, silent: bool = False) -> list[Tile]:
        return tiles


@attrs.define
class ClosureMergerConfig(BaseMergerConfig):
    _target_: str = "lib.merge.ClosureMerger"


cs.store(name="closure_merger", node=ClosureMergerConfig, group="merger")


@attrs.define
class ClosureMerger(BaseMerger):
    """Merge tiles into one tile per table that includes the cells from all tiles."""

    def __call__(self, tiles: list[Tile], *, silent: bool = False) -> list[Tile]:

        tiles_by_table = collections.defaultdict(list)
        for tile in tiles:
            tiles_by_table[tile.table].append(tile)

        result = []
        for table, tiles in tiles_by_table.items():
            row_index = tiles[0].row_index.copy()
            col_index = tiles[0].col_index.copy()
            if len(tiles) > 1:
                for tile in tiles[1:]:
                    row_index = row_index.union(tile.row_index)
                    col_index = col_index.union(tile.col_index)

            # this is done so that the row/column order matches the original table
            row_index = table.data.index.intersection(row_index)
            col_index = table.data.columns.intersection(col_index)

            result_tile = table.create_tile(row_index, col_index, "result-tile")
            result.append(result_tile)

        return result
