"""Disk-backed exact external-node ID lookup.

The optional DuckDB dependency is imported only when an index is queried.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Literal

IdDomain = Literal["int64", "uint64", "string"]


class ExternalNodeIndex:
    """One type-local exact bijection stored in a DuckDB database.

    Instances retain only paths and scalar metadata. Point queries are executed
    against the on-disk ``mapping`` table; no Python forward or reverse map is
    constructed.
    """

    __slots__ = (
        "node_type",
        "id_dtype",
        "row_count",
        "root",
        "node_ids_path",
        "lookup_path",
        "completion_path",
        "_connection",
    )

    def __init__(
        self,
        *,
        node_type: str,
        id_dtype: IdDomain,
        row_count: int,
        root: str | Path,
    ) -> None:
        self.node_type = node_type
        self.id_dtype = id_dtype
        self.row_count = row_count
        self.root = Path(root)
        self.node_ids_path = self.root / "node_ids.parquet"
        self.lookup_path = self.root / "lookup.duckdb"
        self.completion_path = self.root / "mapping.complete.json"
        self._connection = None

    def __len__(self) -> int:
        return self.row_count

    def __enter__(self) -> ExternalNodeIndex:
        self._connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _connect(self):
        if self._connection is None:
            import duckdb

            self._connection = duckdb.connect(
                str(self.lookup_path), read_only=True
            )
        return self._connection

    def close(self) -> None:
        """Close the lazy database handle, if one was opened."""
        connection = self._connection
        if connection is not None:
            self._connection = None
            connection.close()

    def lookup(self, external_id: int | str) -> int:
        """Resolve one exact external ID to its type-local dense ordinal."""
        value = self._validated_external_id(external_id)
        row = (
            self._connect()
            .execute(
                "SELECT local_ordinal FROM mapping WHERE external_id = ?",
                [value],
            )
            .fetchone()
        )
        if row is None:
            raise KeyError(external_id)
        return int(row[0])

    def external_id(self, local_ordinal: int) -> int | str:
        """Resolve one dense ordinal to its exact external ID."""
        if (
            isinstance(local_ordinal, bool)
            or not isinstance(local_ordinal, int)
            or local_ordinal < 0
            or local_ordinal >= self.row_count
        ):
            raise IndexError(local_ordinal)
        row = (
            self._connect()
            .execute(
                "SELECT external_id FROM mapping WHERE local_ordinal = ?",
                [local_ordinal],
            )
            .fetchone()
        )
        if row is None:
            raise IndexError(local_ordinal)
        return row[0]

    def _validated_external_id(self, value: object) -> int | str:
        if self.id_dtype == "string":
            if not isinstance(value, str):
                raise KeyError(value)
            return value
        if isinstance(value, bool) or not isinstance(value, int):
            raise KeyError(value)
        if self.id_dtype == "int64":
            if value < -(2**63) or value >= 2**63:
                raise KeyError(value)
        elif value < 0 or value >= 2**64:
            raise KeyError(value)
        return value


__all__ = ["ExternalNodeIndex", "IdDomain"]
