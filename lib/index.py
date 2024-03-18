import abc
import logging
import operator

import attrs
import numpy as np
import omegaconf
from hydra.core.config_store import ConfigStore
from sklearn.metrics.pairwise import cosine_distances

from lib.lake import DataLake, Tile
from lib.utils import TileSet

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()


@attrs.define
class BaseIndexConfig(abc.ABC):
    _target_: str = omegaconf.MISSING


@attrs.define
class BaseIndex(abc.ABC):
    """Retrieves relevant tiles from a data lake."""

    @abc.abstractmethod
    def __call__(
            self,
            lake: DataLake,
            embedding: np.ndarray,
            *,
            tile_kinds: list[str] | None = None,
            threshold: float | None = None,
            sort: bool = False,
            return_distances: bool = False,
            silent: bool = False
    ) -> list[Tile] | list[tuple[Tile, float]]:
        """Retrieve tiles that are close to the given embedding vector.

        Args:
            lake: Data lake to retrieve from.
            embedding: Embedding vector of the query.
            tile_kinds: Which tile kinds to retrieve.
            threshold: Maximum distance between query and returned tiles.
            sort: Whether to sort the returned entries by distance.
            return_distances: Whether to return the distances together with the entries.
            silent: Whether to disable progress bars.

        Returns:
            List of tiles or list of tiles and distances.

        Raises:
            ValueError if the given embedding's shape does not match the shape of the index entries.
        """
        raise NotImplementedError()


@attrs.define
class ListIndexConfig(BaseIndexConfig):
    _target_: str = "lib.index.ListIndex"
    distance_function: str = "cosine"


cs.store(name="list_index", node=ListIndexConfig, group="index")


@attrs.define
class ListIndex(BaseIndex):
    """Simple index implementation that scans a list."""
    distance_function: str

    def __call__(
            self,
            lake: DataLake,
            embedding: np.ndarray,
            *,
            tile_kinds: list[str] | None = None,
            threshold: float | None = None,
            sort: bool = False,
            return_distances: bool = False,
            silent: bool = False
    ) -> list[Tile] | list[tuple[Tile, float]]:
        tiles = lake.get_all_tiles(tile_kind=tile_kinds)
        if tiles == []:
            return []

        embeddings = [tile.tile_embedding for tile in tiles]
        stacked_embeddings = np.stack(embeddings)
        if embedding.shape != stacked_embeddings.shape[1:]:
            raise ValueError(
                f"The given embedding's shape ({embedding.shape}) does not match the shape of the "
                f"tile embeddings ({stacked_embeddings.shape[1:]})!"
            )

        if self.distance_function == "cosine":
            distances = cosine_distances(stacked_embeddings, embedding.reshape((1, *embedding.shape)))
        elif self.distance_function == "inner-product":
            inner_products = stacked_embeddings @ embedding
            inner_products = inner_products - np.mean(inner_products)
            inner_products = inner_products / np.max(np.abs(inner_products)) / 2
            inner_products = inner_products + 0.5
            distances = 1 - inner_products  # TODO: this means that the distances depend on what is in the data lake
        else:
            raise KeyError(f"Unknown distance function '{self.distance_function}'!")
        distances = distances.reshape((distances.shape[0],))
        pairs = [(tile, float(distance)) for tile, distance in zip(tiles, distances.tolist())]

        if sort:
            pairs.sort(key=operator.itemgetter(1))

        results = []
        for tile, distance in pairs:
            if threshold is None or distance <= threshold:
                if return_distances:
                    results.append((tile, distance))
                else:
                    results.append(tile)

        return results


@attrs.define
class PerfectIndexConfig(BaseIndexConfig):
    _target_: str = "lib.index.PerfectIndex"


cs.store(name="perfect_index", node=PerfectIndexConfig, group="index")


@attrs.define
class PerfectIndex(BaseIndex):
    """Sanity check index implementation that retrieves the ground truth best-fitting tiles."""

    # noinspection PyMethodOverriding
    def __call__(
            self,
            lake: DataLake,
            embedding: np.ndarray,
            gt_tile_kind: str,  # the perfect index needs the ground truths
            *,
            tile_kinds: list[str] | None = None,
            threshold: float | None = None,
            sort: bool = False,
            return_distances: bool = False,
            silent: bool = False
    ) -> list[Tile] | list[tuple[Tile, float]]:
        tiles = lake.get_all_tiles(tile_kind=tile_kinds)
        if tiles == []:
            return []

        gt_tile_set = TileSet.from_lake(lake, tile_kind=gt_tile_kind)

        all_distances = []
        for tile in tiles:
            t_tile_set = TileSet({tile})

            in_other, not_in_other = t_tile_set.compute_overlap(gt_tile_set)
            distance = 1 - (in_other / (in_other + not_in_other))
            all_distances.append(distance)

        pairs = [(tile, distance) for tile, distance in zip(tiles, all_distances)]

        if sort:
            pairs.sort(key=operator.itemgetter(1))

        results = []
        for tile, distance in pairs:
            if threshold is None or distance <= threshold:
                if return_distances:
                    results.append((tile, distance))
                else:
                    results.append(tile)

        return results
