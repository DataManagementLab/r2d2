import collections
import functools
import itertools
import json
import logging
import pathlib
import pickle
import sqlite3
import tempfile
import zipfile
from typing import Mapping, Sequence

import attrs
import numpy as np
import pandas as pd
import tqdm

logger = logging.getLogger(__name__)

LAKE_COMPONENTS = {"table", "table-rows", "table-cols", "table-cells", "tile", "tile-rows", "tile-cols", "tile-cells"}


# noinspection PyProtectedMember
@attrs.define(eq=True, hash=True, slots=False)
class Table:
    """Table in a DataLake."""

    ####################################################################################################################
    # lake data model
    ####################################################################################################################

    _lake: "DataLake" = attrs.field(
        init=True,
        eq=True,
        hash=True,
        repr=True
    )
    _name: str = attrs.field(
        init=True,
        validator=attrs.validators.instance_of(str),
        eq=True,
        hash=True,
        repr=True
    )
    _data: pd.DataFrame = attrs.field(
        init=True,
        validator=attrs.validators.instance_of(pd.DataFrame),
        eq=False,
        hash=False,
        repr=False
    )
    _tiles_dict: dict[str, list["Tile"]] = attrs.field(
        init=False,
        factory=lambda: collections.defaultdict(list),
        eq=False,
        hash=False,
        repr=False
    )

    def __attrs_post_init__(self) -> None:
        if not attrs.validators.get_disabled():
            if self._name in self._lake._tables_dict.keys():
                raise KeyError(f"A table with the name '{self._name}' already exists in '{self._lake}'!")

        self._lake._tables_dict[self._name] = self

    # noinspection PyUnresolvedReferences
    @_lake.validator
    def _validate_lake(self, _, value) -> None:
        if value is not None and not isinstance(value, DataLake):
            raise TypeError(f"The data lake '{value}' is not a DataLake!")

    @functools.cached_property
    def lake(self) -> "DataLake":
        return self._lake

    @functools.cached_property
    def name(self) -> str:
        return self._name

    @functools.cached_property
    def data(self) -> pd.DataFrame:
        return self._data

    @functools.cached_property
    def row_index(self) -> pd.Index:
        return self._data.index

    @functools.cached_property
    def col_index(self) -> pd.Index:
        return self._data.columns

    @functools.cached_property
    def num_cells(self) -> int:
        return len(self._data.index) * len(self._data.columns)

    @functools.cached_property
    def tiles(self) -> Mapping[str, Sequence["Tile"]]:
        return self._tiles_dict

    def get_all_tiles(self, *, tile_kind: str | list[str] | None = None) -> Sequence["Tile"]:
        """Get all Tiles in the Table.

        Args:
            tile_kind: An optional string or list of strings specifying the kind of the Tiles to return.

        Returns:
            A list of Tiles in the Table.
        """
        if tile_kind is not None:
            if isinstance(tile_kind, list):
                return list(itertools.chain.from_iterable(self._tiles_dict[tk] for tk in tile_kind))
            else:
                return list(self._tiles_dict[tile_kind])
        else:
            return list(itertools.chain.from_iterable(self._tiles_dict.values()))

    def create_tile(self, row_index: pd.Index, col_index: pd.Index, tile_kind: str) -> "Tile":
        """Create a Tile that includes parts of the Table.

        Args:
            row_index: A pd.Index specifying the rows to include.
            col_index: A pd.Index specifying the columns to include.
            tile_kind: A str specifying the kind of the Tile.

        Returns:
            The Tile that includes the specified parts of the Table.

        Raises:
            IndexError if one of the indices is not completely included in the table.
        """
        return Tile(self, row_index, col_index, tile_kind)

    def remove_tile(self, tile: "Tile") -> None:
        """Remove the given Tile from the Table.

        This destroys the Tile, which should only exist if registered with the Table.

        Args:
            tile: A Tile to remove from the Table.
        """
        self._tiles_dict[tile._kind].remove(tile)
        self._lake._tiles_dict[tile._kind].remove(tile)
        tile._table = None

    def remove_all_tiles(self, *, tile_kind: str | list[str] | None = None) -> None:
        """Removes Tiles from a Table.

        This destroys the Tiles, which should only exist if registered with the Table.

        Args:
            tile_kind: An optional string or list of strings specifying the kind of the Tiles to remove.
        """
        for tile in list(self.get_all_tiles(tile_kind=tile_kind)):
            self.remove_tile(tile)

    @classmethod
    def create_from_file(cls, lake: "DataLake", path: pathlib.Path) -> "Table":
        """Create a Table from the file with the given path.

        Args:
            lake: A DataLake the Table should be a part of.
            path: A pathlib.Path to the data file.

        Returns:
            The Table created from the file with the given path.
        """
        if path.name.endswith(".parquet"):
            data = pd.read_parquet(path)
        elif path.name.endswith(".csv"):
            data = pd.read_csv(path)
        else:
            raise TypeError(f"Unknown file type '{path.name}'!")
        table_name = path.name[:path.name.rindex(".")]
        return cls(lake, table_name, data)

    ####################################################################################################################
    # other attributes
    ####################################################################################################################

    table_name_embedding: np.ndarray | None = attrs.field(init=False, default=None, eq=False, hash=False, repr=False)
    table_name_embedding_by: object | None = attrs.field(init=False, default=None, eq=False, hash=False, repr=False)
    col_names_embeddings: np.ndarray | None = attrs.field(init=False, default=None, eq=False, hash=False, repr=False)
    col_names_embedding_by: object | None = attrs.field(init=False, default=None, eq=False, hash=False, repr=False)
    cell_values_embeddings: np.ndarray | None = attrs.field(init=False, default=None, eq=False, hash=False, repr=False)
    cell_values_embedding_by: object | None = attrs.field(init=False, default=None, eq=False, hash=False, repr=False)

    table_embedding: np.ndarray | None = attrs.field(init=False, default=None, eq=False, hash=False, repr=False)
    table_embedding_by: object | None = attrs.field(init=False, default=None, eq=False, hash=False, repr=False)
    col_embeddings: np.ndarray | None = attrs.field(init=False, default=None, eq=False, hash=False, repr=False)
    col_embeddings_by: object | None = attrs.field(init=False, default=None, eq=False, hash=False, repr=False)
    row_embeddings: np.ndarray | None = attrs.field(init=False, default=None, eq=False, hash=False, repr=False)
    row_embeddings_by: object | None = attrs.field(init=False, default=None, eq=False, hash=False, repr=False)
    cell_embeddings: np.ndarray | None = attrs.field(init=False, default=None, eq=False, hash=False, repr=False)
    cell_embeddings_by: object | None = attrs.field(init=False, default=None, eq=False, hash=False, repr=False)

    @table_name_embedding.validator
    def _validate_table_name_embedding(self, _, value) -> None:
        if value is not None:
            if not isinstance(value, np.ndarray):
                raise TypeError(f"The table name embedding '{value}' is not a np.ndarray!")
            if value.ndim != 1:
                raise IndexError(f"The table name embedding '{value}' has incorrect dimensions!")

    @col_names_embeddings.validator
    def _validate_col_names_embeddings(self, _, value) -> None:
        if value is not None:
            if not isinstance(value, np.ndarray):
                raise TypeError(f"The column name embeddings '{value}' are not a np.ndarray!")
            if value.ndim != 2 or value.shape[0] != len(self._data.columns):
                raise IndexError(f"The column name embeddings '{value}' have incorrect dimensions!")

    @cell_values_embeddings.validator
    def _validate_cell_values_embeddings(self, _, value) -> None:
        if value is not None:
            if not isinstance(value, np.ndarray):
                raise TypeError(f"The cell values embeddings '{value}' are not a np.ndarray!")
            if value.ndim != 3 or value.shape[0] != len(self._data.index) or value.shape[1] != len(self._data.columns):
                raise IndexError(f"The cell values embeddings '{value}' have incorrect dimensions!")

    @table_embedding.validator
    def _validate_table_embedding(self, _, value) -> None:
        if value is not None:
            if not isinstance(value, np.ndarray):
                raise TypeError(f"The table embedding '{value}' is not a np.ndarray!")
            if value.ndim != 1:
                raise IndexError(f"The table embedding '{value}' has incorrect dimensions!")

    @row_embeddings.validator
    def _validate_row_embeddings(self, _, value) -> None:
        if value is not None:
            if not isinstance(value, np.ndarray):
                raise TypeError(f"The row embeddings '{value}' are not a np.ndarray!")
            if value.ndim != 2 or value.shape[0] != len(self._data.index):
                raise IndexError(f"The row embeddings '{value}' have incorrect dimensions!")

    @col_embeddings.validator
    def _validate_col_embeddings(self, _, value) -> None:
        if value is not None:
            if not isinstance(value, np.ndarray):
                raise TypeError(f"The column embeddings '{value}' are not a np.ndarray!")
            if value.ndim != 2 or value.shape[0] != len(self._data.columns):
                raise IndexError(f"The column embeddings '{value}' have incorrect dimensions!")

    @cell_embeddings.validator
    def _validate_cell_embeddings(self, _, value) -> None:
        if value is not None:
            if not isinstance(value, np.ndarray):
                raise TypeError(f"The cell embeddings '{value}' are not a np.ndarray!")
            if value.ndim != 3 or value.shape[0] != len(self._data.index) or value.shape[1] != len(self._data.columns):
                raise IndexError(f"The cell embeddings '{value}' have incorrect dimensions!")

    ####################################################################################################################
    # serialization
    ####################################################################################################################

    def _save(self, table_ix: int, dir_path: pathlib.Path) -> None:
        table_info = {
            "name": self._name,
            "num_tiles": len(self.get_all_tiles())
        }
        table_info_path = dir_path.joinpath(f"table_{table_ix}_info.json")
        with open(table_info_path, "w", encoding="utf-8") as file:
            json.dump(table_info, file)

        data_path = dir_path.joinpath(f"table_{table_ix}_data.pickle")
        self._data.to_pickle(data_path)

        if self.table_name_embedding is not None:
            np.save(str(dir_path.joinpath(f"table_{table_ix}_table_name_embedding.npy")), self.table_name_embedding)

        if self.col_names_embeddings is not None:
            np.save(str(dir_path.joinpath(f"table_{table_ix}_col_names_embeddings.npy")), self.col_names_embeddings)

        if self.cell_values_embeddings is not None:
            np.save(str(dir_path.joinpath(f"table_{table_ix}_cell_values_embeddings.npy")), self.cell_values_embeddings)

        if self.table_embedding is not None:
            np.save(str(dir_path.joinpath(f"table_{table_ix}_table_embedding.npy")), self.table_embedding)

        if self.row_embeddings is not None:
            np.save(str(dir_path.joinpath(f"table_{table_ix}_row_embeddings.npy")), self.row_embeddings)

        if self.col_embeddings is not None:
            np.save(str(dir_path.joinpath(f"table_{table_ix}_col_embeddings.npy")), self.col_embeddings)

        if self.cell_embeddings is not None:
            np.save(str(dir_path.joinpath(f"table_{table_ix}_cell_embeddings.npy")), self.cell_embeddings)

        for tile_ix, tile in enumerate(self.get_all_tiles()):
            tile._save(table_ix, tile_ix, dir_path)

    @classmethod
    def _load(cls, table_ix: int, zip_file: zipfile.ZipFile, name_set: set[str], lake: "DataLake") -> None:
        with zip_file.open(f"table_{table_ix}_info.json") as file:
            table_info = json.load(file)

        with zip_file.open(f"table_{table_ix}_data.pickle") as file:
            data = pd.read_pickle(file)

        table = cls(lake, table_info["name"], data)

        name = f"table_{table_ix}_table_name_embedding.npy"
        if name in name_set:
            with zip_file.open(name) as file:
                table.table_name_embedding = np.load(file)

        name = f"table_{table_ix}_col_names_embeddings.npy"
        if name in name_set:
            with zip_file.open(name) as file:
                table.col_names_embeddings = np.load(file)

        name = f"table_{table_ix}_cell_values_embeddings.npy"
        if name in name_set:
            with zip_file.open(name) as file:
                table.cell_values_embeddings = np.load(file)

        name = f"table_{table_ix}_table_embedding.npy"
        if name in name_set:
            with zip_file.open(name) as file:
                table.table_embedding = np.load(file)

        name = f"table_{table_ix}_row_embeddings.npy"
        if f"table_{table_ix}_row_embeddings.npy" in name_set:
            with zip_file.open(name) as file:
                table.row_embeddings = np.load(file)

        name = f"table_{table_ix}_col_embeddings.npy"
        if name in name_set:
            with zip_file.open(name) as file:
                table.col_embeddings = np.load(file)

        name = f"table_{table_ix}_cell_embeddings.npy"
        if name in name_set:
            with zip_file.open(name) as file:
                table.cell_embeddings = np.load(file)

        for tile_ix in range(table_info["num_tiles"]):
            Tile._load(table_ix, tile_ix, zip_file, name_set, table)


# noinspection PyProtectedMember
@attrs.define(eq=False, hash=False, slots=False)
class Tile:
    """A chunk of a Table."""

    ####################################################################################################################
    # lake data model
    ####################################################################################################################

    _table: Table = attrs.field(
        init=True,
        validator=attrs.validators.optional(attrs.validators.instance_of(Table)),
        repr=True
    )
    _row_index: pd.Index = attrs.field(
        init=True,
        validator=attrs.validators.instance_of(pd.Index),
        repr=True
    )
    _col_index: pd.Index = attrs.field(
        init=True,
        validator=attrs.validators.instance_of(pd.Index),
        repr=True
    )
    _kind: str = attrs.field(
        init=True,
        validator=attrs.validators.instance_of(str),
        repr=True
    )

    def __attrs_post_init__(self) -> None:
        if not attrs.validators.get_disabled():
            row_diff = self._row_index.difference(self._table._data.index)
            if not row_diff.empty:
                raise IndexError("The row index is not completely included in this table's row index!")

            col_diff = self._col_index.difference(self._table._data.columns)
            if not col_diff.empty:
                raise IndexError("The column index is not completely included in this table's column index!")

        self._table._tiles_dict[self._kind].append(self)
        self._table._lake._tiles_dict[self._kind].append(self)

    @functools.cached_property
    def table(self) -> Table:
        return self._table

    @functools.cached_property
    def lake(self) -> "DataLake":
        return self._table._lake

    @functools.cached_property
    def row_index(self) -> pd.Index:
        return self._row_index

    @functools.cached_property
    def col_index(self) -> pd.Index:
        return self._col_index

    @functools.cached_property
    def kind(self) -> str:
        return self._kind

    @functools.cached_property
    def num_cells(self) -> int:
        return len(self._row_index) * len(self._col_index)

    @functools.cached_property
    def data(self) -> pd.DataFrame:
        return self._table._data.loc[self._row_index, self._col_index]

    @functools.cached_property
    def row_coverage(self) -> np.ndarray:
        return self._table._data.index.isin(self._row_index)

    @functools.cached_property
    def col_coverage(self) -> np.ndarray:
        return self._table._data.columns.isin(self._col_index)

    @functools.cached_property
    def coverage(self) -> np.ndarray:
        row_coverage = self.row_coverage
        col_coverage = self.col_coverage
        return row_coverage.reshape((row_coverage.shape[0], 1)) @ col_coverage.reshape((1, col_coverage.shape[0]))

    ####################################################################################################################
    # other attributes
    ####################################################################################################################

    table_name_embedding: np.ndarray | None = attrs.field(init=False, default=None, eq=False, hash=False, repr=False)
    table_name_embedding_by: object | None = attrs.field(init=False, default=None, eq=False, hash=False, repr=False)
    col_names_embeddings: np.ndarray | None = attrs.field(init=False, default=None, eq=False, hash=False, repr=False)
    col_names_embedding_by: object | None = attrs.field(init=False, default=None, eq=False, hash=False, repr=False)
    cell_values_embeddings: np.ndarray | None = attrs.field(init=False, default=None, eq=False, hash=False, repr=False)
    cell_values_embedding_by: object | None = attrs.field(init=False, default=None, eq=False, hash=False, repr=False)

    tile_embedding: np.ndarray | None = attrs.field(init=False, default=None, repr=False)
    tile_embedding_by: object | None = attrs.field(init=False, default=None, eq=False, hash=False, repr=False)
    col_embeddings: np.ndarray | None = attrs.field(init=False, default=None, repr=False)
    col_embeddings_by: object | None = attrs.field(init=False, default=None, eq=False, hash=False, repr=False)
    row_embeddings: np.ndarray | None = attrs.field(init=False, default=None, repr=False)
    row_embeddings_by: object | None = attrs.field(init=False, default=None, eq=False, hash=False, repr=False)
    cell_embeddings: np.ndarray | None = attrs.field(init=False, default=None, repr=False)
    cell_embeddings_by: object | None = attrs.field(init=False, default=None, eq=False, hash=False, repr=False)

    @table_name_embedding.validator
    def _validate_table_name_embedding(self, _, value) -> None:
        if value is not None:
            if not isinstance(value, np.ndarray):
                raise TypeError(f"The table name embedding '{value}' is not a np.ndarray!")
            if value.ndim != 1:
                raise IndexError(f"The table name embedding '{value}' has incorrect dimensions!")

    @col_names_embeddings.validator
    def _validate_col_names_embeddings(self, _, value) -> None:
        if value is not None:
            if not isinstance(value, np.ndarray):
                raise TypeError(f"The column name embeddings '{value}' are not a np.ndarray!")
            if value.ndim != 2 or value.shape[0] != len(self._col_index):
                raise IndexError(f"The column name embeddings '{value}' have incorrect dimensions!")

    @cell_values_embeddings.validator
    def _validate_cell_values_embeddings(self, _, value) -> None:
        if value is not None:
            if not isinstance(value, np.ndarray):
                raise TypeError(f"The cell values embeddings '{value}' are not a np.ndarray!")
            if value.ndim != 3 or value.shape[0] != len(self._row_index) or value.shape[1] != len(self._col_index):
                raise IndexError(f"The cell values embeddings '{value}' have incorrect dimensions!")

    @tile_embedding.validator
    def _validate_tile_embedding(self, _, value) -> None:
        if value is not None:
            if not isinstance(value, np.ndarray):
                raise TypeError(f"The tile embedding '{value}' is not a np.ndarray!")
            if value.ndim != 1:
                raise IndexError(f"The tile embedding '{value}' has incorrect dimensions!")

    @row_embeddings.validator
    def _validate_row_embeddings(self, _, value) -> None:
        if value is not None:
            if not isinstance(value, np.ndarray):
                raise TypeError(f"The row embeddings '{value}' are not a np.ndarray!")
            if value.ndim != 2 or value.shape[0] != len(self._row_index):
                raise IndexError(f"The row embeddings '{value}' have incorrect dimensions!")

    @col_embeddings.validator
    def _validate_col_embeddings(self, _, value) -> None:
        if value is not None:
            if not isinstance(value, np.ndarray):
                raise TypeError(f"The column embeddings '{value}' are not a np.ndarray!")
            if value.ndim != 2 or value.shape[0] != len(self._col_index):
                raise IndexError(f"The column embeddings '{value}' have incorrect dimensions!")

    @cell_embeddings.validator
    def _validate_cell_embeddings(self, _, value) -> None:
        if value is not None:
            if not isinstance(value, np.ndarray):
                raise TypeError(f"The cell embeddings '{value}' are not a np.ndarray!")
            if value.ndim != 3 or value.shape[0] != len(self._row_index) or value.shape[1] != len(self._col_index):
                raise IndexError(f"The cell embeddings '{value}' have incorrect dimensions!")

    ####################################################################################################################
    # serialization
    ####################################################################################################################

    def _save(self, table_ix: int, tile_ix: int, dir_path: pathlib.Path) -> None:
        tile_info = {
            "kind": self._kind
        }
        tile_info_path = dir_path.joinpath(f"table_{table_ix}_tile_{tile_ix}_info.json")
        with open(tile_info_path, "w", encoding="utf-8") as file:
            json.dump(tile_info, file)

        row_index_path = dir_path.joinpath(f"table_{table_ix}_tile_{tile_ix}_row_index.pickle")
        with open(row_index_path, "wb") as file:
            pickle.dump(self._row_index, file)
        col_index_path = dir_path.joinpath(f"table_{table_ix}_tile_{tile_ix}_col_index.pickle")
        with open(col_index_path, "wb") as file:
            pickle.dump(self._col_index, file)

        if self.table_name_embedding is not None:
            np.save(str(dir_path.joinpath(f"table_{table_ix}_tile_{tile_ix}_table_name_embedding.npy")), self.table_name_embedding)

        if self.col_names_embeddings is not None:
            np.save(str(dir_path.joinpath(f"table_{table_ix}_tile_{tile_ix}_col_names_embeddings.npy")), self.col_names_embeddings)

        if self.cell_values_embeddings is not None:
            np.save(str(dir_path.joinpath(f"table_{table_ix}_tile_{tile_ix}_cell_values_embeddings.npy")), self.cell_values_embeddings)

        if self.tile_embedding is not None:
            tile_embedding_path = dir_path.joinpath(f"table_{table_ix}_tile_{tile_ix}_tile_embedding.npy")
            np.save(str(tile_embedding_path), self.tile_embedding)

        if self.row_embeddings is not None:
            row_embeddings_path = dir_path.joinpath(f"table_{table_ix}_tile_{tile_ix}_row_embeddings.npy")
            np.save(str(row_embeddings_path), self.row_embeddings)

        if self.col_embeddings is not None:
            col_embeddings_path = dir_path.joinpath(f"table_{table_ix}_tile_{tile_ix}_col_embeddings.npy")
            np.save(str(col_embeddings_path), self.col_embeddings)

        if self.cell_embeddings is not None:
            cell_embeddings_path = dir_path.joinpath(f"table_{table_ix}_tile_{tile_ix}_cell_embeddings.npy")
            np.save(str(cell_embeddings_path), self.cell_embeddings)

    @classmethod
    def _load(cls, table_ix: int, tile_ix: int, zip_file: zipfile.ZipFile, name_set: set[str], table: Table) -> None:
        with zip_file.open(f"table_{table_ix}_tile_{tile_ix}_info.json") as file:
            tile_info = json.load(file)

        with zip_file.open(f"table_{table_ix}_tile_{tile_ix}_row_index.pickle") as file:
            row_index = pickle.load(file)

        with zip_file.open(f"table_{table_ix}_tile_{tile_ix}_col_index.pickle") as file:
            col_index = pickle.load(file)

        tile = table.create_tile(row_index, col_index, tile_info["kind"])

        name = f"table_{table_ix}_tile_{tile_ix}_table_name_embedding.npy"
        if name in name_set:
            with zip_file.open(name) as file:
                tile.table_name_embedding = np.load(file)

        name = f"table_{table_ix}_tile_{tile_ix}_col_names_embeddings.npy"
        if name in name_set:
            with zip_file.open(name) as file:
                tile.col_names_embeddings = np.load(file)

        name = f"table_{table_ix}_tile_{tile_ix}_cell_values_embeddings.npy"
        if name in name_set:
            with zip_file.open(name) as file:
                tile.cell_values_embeddings = np.load(file)

        name = f"table_{table_ix}_tile_{tile_ix}_tile_embedding.npy"
        if name in name_set:
            with zip_file.open(name) as file:
                tile.tile_embedding = np.load(file)

        name = f"table_{table_ix}_tile_{tile_ix}_row_embeddings.npy"
        if name in name_set:
            with zip_file.open(name) as file:
                tile.row_embeddings = np.load(file)

        name = f"table_{table_ix}_tile_{tile_ix}_col_embeddings.npy"
        if name in name_set:
            with zip_file.open(name) as file:
                tile.col_embeddings = np.load(file)

        name = f"table_{table_ix}_tile_{tile_ix}_cell_embeddings.npy"
        if name in name_set:
            with zip_file.open(name) as file:
                tile.cell_embeddings = np.load(file)


# noinspection PyProtectedMember
@attrs.define(eq=True, hash=True, slots=False)
class DataLake:
    """A data lake."""
    _name: str = attrs.field(
        init=True,
        validator=attrs.validators.instance_of(str),
        eq=True,
        hash=True,
        repr=True
    )
    _tables_dict: dict[str, Table] = attrs.field(
        init=False,
        factory=dict,
        eq=False,
        hash=False,
        repr=False
    )
    _tiles_dict: dict[str, list[Tile]] = attrs.field(
        init=False,
        factory=lambda: collections.defaultdict(list),
        eq=False,
        hash=False,
        repr=False
    )

    @functools.cached_property
    def name(self) -> str:
        return self._name

    @property  # this cannot be cached
    def num_cells(self) -> int:
        return sum(table.num_cells for table in self._tables_dict.values())

    @functools.cached_property
    def tables(self) -> Mapping[str, Table]:
        return self._tables_dict

    @functools.cached_property
    def tiles(self) -> Mapping[str, Sequence[Tile]]:
        return self._tiles_dict

    def get_all_tables(self) -> Sequence[Table]:
        """Get all Tables in the DataLake.

        Returns:
            A list of Tables in the DataLake.
        """
        return list(self._tables_dict.values())

    def get_all_tiles(self, *, tile_kind: str | list[str] | None = None) -> Sequence[Tile]:
        """Get all Tiles in the DataLake.

        Args:
            tile_kind: An optional string or list of strings specifying the kind of the Tiles.

        Returns:
            A list of Tiles in the DataLake.
        """
        if tile_kind is not None:
            if isinstance(tile_kind, list):
                return list(itertools.chain.from_iterable(self._tiles_dict[tk] for tk in tile_kind))
            else:
                return list(self._tiles_dict[tile_kind])
        else:
            return list(itertools.chain.from_iterable(self._tiles_dict.values()))

    def remove_table(self, table: Table) -> None:
        """Remove the given Table from the DataLake.

        This destroys the Table, which should only exist if registered with the DataLake.

        Args:
            table: A Table to remove from the DataLake.
        """
        del self._tables_dict[table._name]

        for tile in table.get_all_tiles():
            self._tiles_dict[tile._kind].remove(tile)
            tile._table = None

        table._lake = None
        table._tiles_dict = collections.defaultdict(list)

    def remove_all_tables(self) -> None:
        """Removes all Tables in the DataLake.

        This destroys the Tables, which should only exist if registered with the DataLake.
        """
        for table in self.get_all_tables():
            self.remove_table(table)

    def remove_all_tiles(self, *, tile_kind: str | list[str] | None = None) -> None:
        """Removes all Tiles from all Tables in the DataLake.

        This destroys the Tiles, which should only exist if registered with the Table.

        Args:
            tile_kind: An optional string or list of strings specifying the kind of Tiles to remove.
        """
        for table in self._tables_dict.values():
            table.remove_all_tiles(tile_kind=tile_kind)

    @classmethod
    def create_from_directory(cls, name: str, path: pathlib.Path) -> "DataLake":
        """Create a DataLake from the fies in the given directory.

        Args:
            name: A string name for the DataLake.
            path: A pathlib.Path to the directory.

        Returns:
            The created DataLake.

        Raises:
            ValueError if the specified path is not a directory.
            Other Errors if the initialization of the DataLake or its Tables fails.
        """
        logger.debug(f"Create data lake from directory '{path}'.")

        if not path.is_dir():
            raise ValueError(f"The specified path '{path}' is not a directory.")

        csv_paths = list(sorted(path.glob("*.csv")))
        parquet_paths = list(sorted(path.glob("*.parquet")))
        paths = parquet_paths + csv_paths

        lake = cls(name)

        for path in tqdm.tqdm(paths, desc="Create tables"):
            Table.create_from_file(lake, path)

        logger.debug("Created the data lake.")
        return lake

    # noinspection SqlDialectInspection,SqlNoDataSourceInspection
    @classmethod
    def create_from_sqlite(cls, name: str, path: pathlib.Path) -> "DataLake":
        logger.debug(f"Create data lake from sqlite '{path}'.")

        lake = cls(name)

        with sqlite3.connect(path) as connection:
            # This was necessary to circumvent `sqlite3.OperationalError: Could not decode to UTF-8 column ...`
            connection.text_factory = lambda b: b.decode(errors="ignore")

            table_names = connection.execute("SELECT name FROM sqlite_master WHERE type='table';")
            table_names = table_names.fetchall()
            table_names = [table_name[0] for table_name in table_names]

            for table_name in table_names:
                table_data = pd.read_sql_query(f"SELECT * FROM \"{table_name}\"", connection)

                Table(lake, table_name, table_data)

        logger.debug("Created the data lake.")
        return lake

    ####################################################################################################################
    # serialization
    ####################################################################################################################

    @classmethod
    def load(cls, path: pathlib.Path, *, silent: bool = False) -> "DataLake":
        """Load a DataLake from a ZIP archive.

        Args:
            path: The pathlib.Path to the ZIP archive.
            silent: Whether to disable progress bars.

        Returns:
            The DataLake.

        Raises:
            ValueError if the specified path does not exist.
        """
        logger.debug(f"Load the data lake from '{path}'.")

        if not path.is_file():
            raise ValueError(f"The specified path '{path}' does not exist.")

        with attrs.validators.disabled():
            with zipfile.ZipFile(path, "r") as zip_file:
                name_set = set(zip_file.namelist())
                with zip_file.open("lake_info.json") as file:
                    lake_info = json.load(file)

                lake = cls(lake_info["name"])

                iterable = range(lake_info["num_tables"])
                if not silent:
                    iterable = tqdm.tqdm(iterable, desc="load tables", leave=False, total=lake_info["num_tables"])
                for table_ix in iterable:
                    Table._load(table_ix, zip_file, name_set, lake)

        logger.debug("Loaded the data lake.")
        return lake

    def save(self, path: pathlib.Path, *, silent: bool = False) -> None:
        """Save the DataLake to a ZIP archive.

        Args:
            path: The pathlib.Path to the ZIP archive.
            silent: Whether to disable progress bars.

        Raises:
            ValueError if the specified path is invalid.
        """
        logger.debug(f"Save the data lake to '{path}'.")

        if not path.parent.is_dir():
            raise ValueError(f"The specified path '{path}' is invalid.")
        with tempfile.TemporaryDirectory(dir=path.parent) as tmp_path:
            tmp_path = pathlib.Path(tmp_path)
            lake_info = {
                "name": self._name,
                "num_tables": len(self._tables_dict)
            }
            lake_info_path = tmp_path / "lake_info.json"
            with open(lake_info_path, "w", encoding="utf-8") as file:
                json.dump(lake_info, file)

            iterable = self._tables_dict.values()
            if not silent:
                iterable = tqdm.tqdm(iterable, desc="save tables", leave=False)
            for table_ix, table in enumerate(iterable):
                table._save(table_ix, tmp_path)

            with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zip_file:
                for path in sorted(tmp_path.glob("*.*")):
                    zip_file.write(path, arcname=path.name)

        logger.debug("Saved the data lake.")
