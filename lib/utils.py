import collections
import logging
import os
import pathlib
import random
from typing import Iterable, Any

import attrs
import numpy as np
import pandas as pd
import tqdm

from lib.colors import LIST_A, LIST_B, LIST_C, LIST_D
from lib.lake import Tile, Table, DataLake

logger = logging.getLogger(__name__)


def get_data_path() -> pathlib.Path:
    """Get the absolute path of the data directory.

    Returns:
        A pathlib.Path to the data directory.
    """
    path = pathlib.Path(os.path.dirname(__file__)).joinpath("../data").resolve()
    os.makedirs(path, exist_ok=True)
    return path


def get_models_path() -> pathlib.Path:
    """Get the path of the directory in which to store all models.

    Returns:
        A pathlib.Path of the directory in which to store all models.
    """
    path = get_data_path() / "models"
    os.makedirs(path, exist_ok=True)
    return path


def tqdm_silent(iterable: Iterable, silent: bool = False, **kwargs) -> Iterable:
    if silent:
        return iterable
    else:
        return tqdm.tqdm(iterable, **kwargs)


def _sort_table_by_tiles(
        table: Table, *,
        tile_kind: str | None = None, data: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Return the data of the given Table with the rows and columns sorted by Tile.

    The method uses the values of the first row to sort the columns and the values of the first column to sort the rows.

    Args:
        table: The Table that determines the indices and potentially the data.
        tile_kind: The kind of Tile to consider.
        data: The Data that should be used instead of the Table's data.

    Returns:
        The data sorted by Tile.
    """
    tile_frame = pd.DataFrame(index=table.data.index, columns=table.data.columns)

    for tile_ix, tile in enumerate(table.get_all_tiles(tile_kind=tile_kind)):
        tile_frame.loc[tile.row_index, tile.col_index] = tile_ix

    tile_frame = tile_frame.sort_values(tile_frame.columns[0], axis=0)
    tile_frame = tile_frame.sort_values(tile_frame.index[0], axis=1)

    if data is None:
        data = table.data.copy()
    data = data.loc[tile_frame.index, tile_frame.columns]
    return data


def _dataframe_to_html(
        value_df: pd.DataFrame,
        *,
        color_df: pd.DataFrame | None = None,
        font_style_df: pd.DataFrame | None = None
) -> str:
    random.seed(323)
    a, b, c, d = list(LIST_A), list(LIST_B), list(LIST_C), list(LIST_D)
    random.shuffle(a)
    random.shuffle(b)
    random.shuffle(c)
    random.shuffle(d)
    colors = a + b + c + d

    def idx_to_color(idx: int) -> str:
        idx = idx % len(colors)
        return colors[idx]

    parts = []
    parts.append(
        """
                <style>
                    table {
                        border: 1px solid;
                        border-collapse: collapse;
                    }
                    td {
                        padding: 5px 20px 5px 5px;
                        text-align: left;
                        border: 1px solid;
                    }
                    th {
                        padding: 5px 20px 5px 5px;
                        text-align: left;
                        border: 1px solid;
                    }
                </style>
            """
    )
    parts.append("<table>")

    parts.append("<tr>")
    for column_name in color_df.columns:
        parts.append(f"<th>{column_name}</th>")
    parts.append("</tr>")

    for row_idx in color_df.index:
        parts.append("<tr>")
        for col_idx in color_df.columns:
            parts.append("<td>")
            value = value_df.at[row_idx, col_idx]
            color = color_df.at[row_idx, col_idx] if color_df is not None else None
            if isinstance(color, list) and len(color) == 0 or not isinstance(color, list) and pd.isna(color):
                color = None
            font_style = font_style_df.at[row_idx, col_idx] if font_style_df is not None else None
            if not isinstance(font_style, str):
                font_style = None

            if color is not None:
                if font_style is None:
                    font_style = ""
                if len(color) == 1:
                    c = idx_to_color(color[0])
                    parts.append(f'<span style="color:{c};{font_style}">{value}</span>')
                else:
                    for char_ix, char in enumerate(list(str(value))):
                        c = idx_to_color(color[char_ix % len(color)])
                        parts.append(f'<span style="color:{c};{font_style}">{char}</span>')
            elif font_style is not None:
                parts.append(f'<span style="{font_style}">{value}</span>')
            else:
                parts.append(str(value))

            parts.append("</td>")

        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


def make_navigation_links(*args) -> str:
    if len(args) % 2 != 0:
        raise AssertionError(f"The number of arguments ({len(args)}) not divisible by two!")

    params = []
    for n in range(0, len(args), 2):
        name = args[n]
        if not isinstance(name, str):
            raise AssertionError(f"The name parameter '{name}' is not a string!")
        idx = args[n + 1]
        if not isinstance(idx, int):
            raise AssertionError(f"The index parameter '{idx}' is not an integer!")
        params.append((name, idx))

    lines = []
    for prev_name, _ in params:
        line_parts = []
        line_parts.append("<a href=\"")
        file_path_parts = []
        for name, idx in params:
            if name == prev_name:
                file_path_parts.append(f"{name}-{idx - 1}")
            else:
                file_path_parts.append(f"{name}-{idx}")
        line_parts.append("-".join(file_path_parts))
        line_parts.append(f".html\">Previous {prev_name}</a>")
        lines.append("".join(line_parts))

    for next_name, _ in reversed(params):
        line_parts = []
        line_parts.append("<a href=\"")
        file_path_parts = []
        for name, idx in params:
            if name == next_name:
                file_path_parts.append(f"{name}-{idx + 1}")
            else:
                file_path_parts.append(f"{name}-{idx}")
        line_parts.append("-".join(file_path_parts))
        line_parts.append(f".html\">Next {next_name}</a>")
        lines.append("".join(line_parts))

    return " ".join(lines)


@attrs.define
class TileSet:
    tiles: set[Tile] = attrs.field(converter=set)

    @classmethod
    def from_lake(cls, lake: DataLake, *, tile_kind: str | None = None) -> "TileSet":
        return cls(set(lake.get_all_tiles(tile_kind=tile_kind)))

    @classmethod
    def from_table(cls, table: Table, *, tile_kind: str | None = None) -> "TileSet":
        return cls(set(table.get_all_tiles(tile_kind=tile_kind)))

    def get_all_lakes(self, *, tile_kind: str | None = None) -> set[DataLake]:
        all_lakes = set()
        for tile in self.tiles:
            if tile_kind is None or tile.kind == tile_kind:
                all_lakes.add(tile.lake)
        return all_lakes

    def get_all_tables(self, *, lake: DataLake | None = None, tile_kind: str | None = None) -> set[Table]:
        all_tables = set()
        for tile in self.tiles:
            if (tile_kind is None or tile.kind == tile_kind) and (lake is None or tile.lake == lake):
                all_tables.add(tile.table)
        return all_tables

    def get_all_tiles(
            self, *, lake: DataLake | None = None,
            table: Table | None = None, tile_kind: str | None = None
    ) -> set[Tile]:
        return set(
            tile for tile in self.tiles if (tile_kind is None or tile.kind == tile_kind) \
            and (lake is None or tile.lake == lake) and (table is None or tile.table == table)
        )

    def get_all_tile_kinds(
            self, *, lake: DataLake | None = None,
            table: Table | None = None, tile_kind: str | None = None
    ) -> set[str]:
        return set(
            tile.kind for tile in self.tiles if (tile_kind is None or tile.kind == tile_kind) \
            and (lake is None or tile.lake == lake) and (table is None or tile.table == table)
        )

    def _table_coverage(self, *, tile_kind: str | None = None) -> dict[Table, np.ndarray]:
        table_coverage = {}
        for tile in self.get_all_tiles(tile_kind=tile_kind):
            if tile.table not in table_coverage.keys():
                table_coverage[tile.table] = np.zeros((len(tile.table.row_index), len(tile.table.col_index)), dtype=np.bool_)
            table_coverage[tile.table] = np.logical_or(table_coverage[tile.table], tile.coverage)
        return table_coverage

    def _num_covered_cells(self, coverage: np.ndarray) -> int:
        return int(np.sum(coverage))

    def compute_num_cells(self, *, tile_kind: str | None = None) -> int:
        # this is fairly tricky since tiles in the tile set can overlap
        table_coverage = self._table_coverage(tile_kind=tile_kind)
        return sum(self._num_covered_cells(coverage) for coverage in table_coverage.values())

    def compute_overlap(self, other: "TileSet", *, tile_kind: str | None = None) -> tuple[int, int]:
        num_in_other = 0
        num_not_in_other = 0

        self_table_coverage = self._table_coverage(tile_kind=tile_kind)
        other_table_coverage = other._table_coverage(tile_kind=tile_kind)

        for table, self_coverage in self_table_coverage.items():
            if table not in other_table_coverage.keys():
                num_not_in_other += self._num_covered_cells(self_coverage)
            else:
                other_coverage = other_table_coverage[table]
                combined_coverage = np.multiply(self_coverage, other_coverage)
                num_combined_coverage = self._num_covered_cells(combined_coverage)
                num_in_other += num_combined_coverage
                num_not_in_other += self._num_covered_cells(self_coverage) - num_combined_coverage

        return num_in_other, num_not_in_other

    def visualize_as_html(
            self,
            path: pathlib.Path,
            *,
            tile_kind: str | None = None,
            show_all_tables: bool = False,
            sort: bool = False,
            title: str | None = None,
            max_value_len: int | None = None,
            color_tile_kinds: list[str] | None = None,
            tile_kind_to_font_style: dict[str, str] | None = None
    ) -> None:
        tables_with_tiles = collections.defaultdict(list)
        if show_all_tables:
            for lake in self.get_all_lakes():
                for table in lake.get_all_tables():
                    tables_with_tiles[table] = []
        for tile in self.get_all_tiles(tile_kind=tile_kind):
            tables_with_tiles[tile.table].append(tile)

        parts = []
        parts.append(
            """
                        <style>
                            * {
                                font-family: monospace, monospace;
                            }
                            .table-div {
                                padding: 30px;
                            }
                            .outer-div {
                                display:flex;
                                flex-wrap:wrap;
                            }
                        </style>
                    """
        )

        if title is not None:
            parts.append(f"<h2>{title}</h2>")

        parts.append('<div class="outer-div">')
        for table, tiles in tables_with_tiles.items():
            parts.append('<div class="table-div">')
            parts.append(f"<h3>{table.name}</h3>")

            value_df = table.data
            if max_value_len is not None:
                value_df = value_df.map(lambda x: str(x)[:max_value_len])

            color_df = pd.DataFrame(index=table.row_index, columns=table.col_index)
            if color_tile_kinds is not None:
                for tile_ix, tile in enumerate(tiles):
                    if tile.kind in color_tile_kinds:
                        for row_idx in tile.row_index:
                            for col_idx in tile.col_index:
                                if pd.isna(color_df.at[row_idx, col_idx]):
                                    color_df.at[row_idx, col_idx] = {tile_ix}
                                else:
                                    color_df.at[row_idx, col_idx].add(tile_ix)
                color_df = color_df.map(lambda x: list(x) if isinstance(x, set) else x)

            font_style_df = pd.DataFrame(index=table.row_index, columns=table.col_index)
            if tile_kind_to_font_style is not None:
                for tile in tiles:
                    if tile.kind in tile_kind_to_font_style.keys():
                        font_style_df.loc[tile.row_index, tile.col_index] = tile_kind_to_font_style[tile.kind]

            if sort:
                value_df = _sort_table_by_tiles(table, tile_kind=tile_kind)
                color_df = _sort_table_by_tiles(table, tile_kind=tile_kind, data=color_df)
                font_style_df = _sort_table_by_tiles(table, tile_kind=tile_kind, data=font_style_df)

            parts.append(_dataframe_to_html(value_df, color_df=color_df, font_style_df=font_style_df))
            parts.append("</div>")
        parts.append("</div class=\"outer-div\">")

        with open(path, "w", encoding="utf-8") as file:
            file.write("".join(parts))


@attrs.define
class GroundTruth:
    name: str
    lake_name: str
    partition: str
    information_need: str
    sql_query: str
    answer_str: str | None
    answer_list: list[str] | None
    answer_table: dict[str, list[Any]] | None
    data: dict


@attrs.define
class Accuracy:
    """Accuracy metric."""
    correct: int
    incorrect: int

    @property
    def total(self) -> int:
        """Total number of instances.

        Returns:
            The total number of instances.

        >>> Accuracy(1, 1).total
        2
        """
        return self.correct + self.incorrect

    @property
    def accuracy(self) -> float:
        """Accuracy score.

        Returns:
            The accuracy score.

        >>> Accuracy(1, 1).accuracy
        0.5
        """
        return self.correct / self.total

    @classmethod
    def empty(cls) -> "Accuracy":
        """Create an empty accuracy object.

        Returns:
            An empty accuracy object.

        >>> Accuracy.empty()
        Accuracy(correct=0, incorrect=0)
        """
        return cls(0, 0)

    def push(self, is_correct: bool) -> None:
        """Include the given instance in the accuracy.

        Args:
            is_correct: Whether the instance is correct.

        >>> acc = Accuracy(0, 0)
        >>> acc.push(True)
        >>> acc
        Accuracy(correct=1, incorrect=0)
        """
        if is_correct:
            self.correct += 1
        else:
            self.incorrect += 1

    def __add__(self, other: "Accuracy") -> "Accuracy":
        return Accuracy(self.correct + other.correct, self.incorrect + other.incorrect)

    def __radd__(self, other: "Accuracy") -> None:
        self.correct += other.correct
        self.incorrect += other.incorrect

    def as_dict(self) -> dict[str, int | float]:
        return {
            "correct": self.correct,
            "incorrect": self.incorrect,
            "accuracy": self.accuracy,
            "total": self.total
        }


@attrs.define
class ConfusionMatrix:
    """Confusion matrix with precision, recall, and F1 score."""
    TP: int
    FP: int
    TN: int
    FN: int

    @property
    def total(self) -> int:
        """Total number of instances.

        Returns:
            The total number of instances.

        >>> ConfusionMatrix(1, 1, 0, 1).total
        3
        """
        return self.TN + self.FP + self.FN + self.TP

    @property
    def precision(self) -> float:
        """Precision score.

        Returns:
            The precision score.

        >>> ConfusionMatrix(1, 1, 0, 1).precision
        0.5
        """
        if self.TP + self.FP == 0:
            return 1
        return self.TP / (self.TP + self.FP)

    @property
    def recall(self) -> float:
        """Recall score.

        Returns:
            The recall score.

        >>> ConfusionMatrix(1, 1, 0, 1).recall
        0.5
        """
        if self.TP + self.FN == 0:
            return 0
        return self.TP / (self.TP + self.FN)

    @property
    def f1_score(self) -> float:
        """F1 score.

        Returns:
            The F1 score.

        >>> ConfusionMatrix(1, 1, 0, 1).f1_score
        0.5
        """
        if self.precision + self.recall == 0:
            return 0
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    @classmethod
    def empty(cls) -> "ConfusionMatrix":
        """Create an empty confusion matrix object.

        Returns:
            An empty confusion matrix object.

        >>> ConfusionMatrix.empty()
        Confusion(TP=0, FP=0, TN=0, FN=0)
        """
        return cls(0, 0, 0, 0)

    def __add__(self, other: "ConfusionMatrix") -> "ConfusionMatrix":
        return ConfusionMatrix(self.TP + other.TP, self.FP + other.FP, self.TN + other.TN, self.FN + other.FN)

    def __radd__(self, other: "ConfusionMatrix") -> None:
        self.TP += other.TP
        self.FP += other.FP
        self.TN += other.TN
        self.FN += other.FN

    def as_dict(self) -> dict[str, int | float]:
        return {
            "TN": self.TN,
            "TP": self.TP,
            "FN": self.FN,
            "FP": self.FP,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "total": self.total
        }


def compute_slicing_and_retrieval_confusion_matrix(
        lake: DataLake, result: TileSet,
        ground_truth: TileSet
) -> ConfusionMatrix:
    confusion = ConfusionMatrix.empty()

    confusion.TP, confusion.FP = result.compute_overlap(ground_truth)
    _, confusion.FN = ground_truth.compute_overlap(result)

    confusion.TN = lake.num_cells - confusion.FP - confusion.FN - confusion.TP

    return confusion
