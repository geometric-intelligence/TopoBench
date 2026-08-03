"""Immutable, row-aligned prediction export contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

PredictionColumn = torch.Tensor | np.ndarray


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _column_length(name: str, value: object) -> int:
    if not isinstance(value, (torch.Tensor, np.ndarray)):
        raise TypeError(
            f"prediction column {name!r} must be a tensor or ndarray"
        )
    if value.ndim == 0:
        raise ValueError(
            f"prediction column {name!r} must have a leading dimension"
        )
    if isinstance(value, np.ndarray) and value.dtype.hasobject:
        raise TypeError(
            f"prediction column {name!r} must not have object dtype"
        )
    return int(value.shape[0])


def _snapshot_identity_column(
    name: str,
    value: object,
) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        snapshot = value.detach().cpu().numpy().copy()
    elif isinstance(value, np.ndarray):
        snapshot = np.array(value, copy=True)
    else:
        raise TypeError(
            f"prediction column {name!r} must be a tensor or ndarray"
        )
    snapshot.setflags(write=False)
    return snapshot


def _validate_identity_column(name: str, value: PredictionColumn) -> None:
    if value.ndim != 1:
        raise ValueError(f"identity column {name!r} must be rank one")
    if isinstance(value, torch.Tensor):
        if (
            value.dtype == torch.bool
            or value.is_floating_point()
            or value.is_complex()
        ):
            raise TypeError(
                f"identity column {name!r} must contain integer or string values"
            )
        return
    if value.dtype.kind not in {"i", "u", "U", "S"}:
        raise TypeError(
            f"identity column {name!r} must contain integer or string values"
        )
    if value.dtype.kind in {"U", "S"} and bool(np.any(value == "")):
        raise ValueError(f"identity column {name!r} has a missing value")


def _identity_values(
    name: str,
    value: PredictionColumn,
) -> tuple[int | str, ...]:
    _validate_identity_column(name, value)
    raw_values = (
        value.detach().cpu().tolist()
        if isinstance(value, torch.Tensor)
        else value.tolist()
    )
    normalized: list[int | str] = []
    for row, item in enumerate(raw_values):
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        if isinstance(item, str):
            if not item:
                raise ValueError(
                    f"identity column {name!r} has a missing value at row {row}"
                )
            normalized.append(item)
        elif isinstance(item, Integral) and not isinstance(item, bool):
            normalized.append(int(item))
        else:
            raise TypeError(
                f"identity column {name!r} has an invalid value at row {row}"
            )
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class PredictionIdentity:
    """Canonical composite identity columns for prediction rows."""

    columns: Mapping[str, PredictionColumn]
    key: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.columns, Mapping) or not self.columns:
            raise ValueError("identity columns must be a non-empty mapping")
        if not isinstance(self.key, tuple) or not self.key:
            raise ValueError("identity key must be a non-empty tuple")
        if any(not isinstance(name, str) or not name for name in self.key):
            raise ValueError("identity key names must be non-empty strings")
        if len(set(self.key)) != len(self.key):
            raise ValueError("identity key must not contain duplicate columns")

        normalized: dict[str, PredictionColumn] = {}
        row_count: int | None = None
        for name, column in self.columns.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "identity column names must be non-empty strings"
                )
            snapshot = _snapshot_identity_column(name, column)
            count = _column_length(name, snapshot)
            _validate_identity_column(name, snapshot)
            if row_count is None:
                row_count = count
            elif count != row_count:
                raise ValueError("identity columns must be row-aligned")
            normalized[name] = snapshot

        missing_keys = set(self.key).difference(normalized)
        if missing_keys:
            raise ValueError(
                f"identity key references missing columns: {sorted(missing_keys)}"
            )

        values = {
            name: _identity_values(name, normalized[name]) for name in self.key
        }
        rows = tuple(
            tuple(values[name][row] for name in self.key)
            for row in range(row_count or 0)
        )
        duplicate = len(rows) != len(set(rows))
        if duplicate:
            raise ValueError("identity key contains duplicate rows")
        object.__setattr__(self, "columns", MappingProxyType(normalized))

    @property
    def num_rows(self) -> int:
        """Return the exact aligned row count."""
        first = next(iter(self.columns.values()))
        return int(first.shape[0])

    @property
    def rows(self) -> tuple[tuple[int | str, ...], ...]:
        """Return canonical key tuples in payload order."""
        values = {
            name: _identity_values(name, self.columns[name])
            for name in self.key
        }
        return tuple(
            tuple(values[name][row] for name in self.key)
            for row in range(self.num_rows)
        )


@dataclass(frozen=True, slots=True)
class PredictionPayload:
    """Selected model rows plus explicit export semantics and schema metadata."""

    identity: PredictionIdentity
    prediction: PredictionColumn
    columns: Mapping[str, PredictionColumn]
    column_metadata: Mapping[str, Mapping[str, Any]]
    output_semantics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, PredictionIdentity):
            raise TypeError("identity must be a PredictionIdentity")
        if not isinstance(self.columns, Mapping):
            raise TypeError("columns must be a mapping")
        required = {"target", "raw_output"}
        missing = required.difference(self.columns)
        if missing:
            raise ValueError(
                f"payload is missing required columns: {sorted(missing)}"
            )
        collisions = set(self.columns).intersection(self.identity.columns)
        if collisions:
            raise ValueError(
                "payload columns collide with identity columns: "
                f"{sorted(collisions)}"
            )

        normalized: dict[str, PredictionColumn] = {}
        for name, column in self.columns.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "payload column names must be non-empty strings"
                )
            count = _column_length(name, column)
            if count != self.identity.num_rows:
                raise ValueError(
                    "payload columns must be row-aligned with identity"
                )
            normalized[name] = column
        if (
            _column_length("prediction", self.prediction)
            != self.identity.num_rows
        ):
            raise ValueError("prediction must be row-aligned with identity")

        if not isinstance(self.column_metadata, Mapping):
            raise TypeError("column_metadata must be a mapping")
        declared = set(normalized) | {"prediction"}
        metadata_names = set(self.column_metadata)
        if metadata_names != declared:
            names = sorted(metadata_names.symmetric_difference(declared))
            raise ValueError(
                f"undeclared payload columns or metadata: {names}"
            )
        for name, metadata in self.column_metadata.items():
            if not isinstance(metadata, Mapping):
                raise TypeError(
                    f"column metadata for {name!r} must be a mapping"
                )
            role = metadata.get("role")
            if not isinstance(role, str) or not role:
                raise ValueError(
                    f"column metadata for {name!r} requires a role"
                )

        if not isinstance(self.output_semantics, Mapping):
            raise TypeError("output_semantics must be a mapping")
        task = self.output_semantics.get("task")
        if task not in {"classification", "regression"}:
            raise ValueError(
                "output_semantics task must be classification or regression"
            )
        target = normalized["target"]
        raw_output = normalized["raw_output"]
        if task == "classification":
            if target.ndim != 1:
                raise ValueError("classification target must have shape [N]")
            if raw_output.ndim != 2:
                raise ValueError(
                    "classification raw_output must have shape [N, C]"
                )
            if (
                self.prediction.ndim != 2
                or self.prediction.shape != raw_output.shape
            ):
                raise ValueError(
                    "classification prediction and raw_output shapes must "
                    "match exactly"
                )
            vocabulary = self.output_semantics.get("class_vocabulary")
            if not isinstance(vocabulary, (tuple, list)):
                raise TypeError(
                    "classification class_vocabulary must be a sequence"
                )
            if not vocabulary:
                raise ValueError(
                    "classification class_vocabulary must be non-empty"
                )
            try:
                unique_classes = len(set(vocabulary))
            except TypeError as error:
                raise ValueError(
                    "classification class_vocabulary must contain unique "
                    "hashable values"
                ) from error
            if unique_classes != len(vocabulary):
                raise ValueError(
                    "classification class_vocabulary must contain unique values"
                )
            if len(vocabulary) != int(raw_output.shape[1]):
                raise ValueError(
                    "classification class_vocabulary must match raw_output width"
                )
        else:
            if target.ndim != 2 or int(target.shape[1]) != 1:
                raise ValueError("regression target must have shape [N, 1]")
            if raw_output.shape != target.shape:
                raise ValueError(
                    "regression raw_output shape must match target exactly"
                )
            if self.prediction.shape != target.shape:
                raise ValueError(
                    "regression prediction shape must match target exactly"
                )
            vocabulary = self.output_semantics.get("class_vocabulary", ())
            if vocabulary not in ((), []):
                raise ValueError("regression class_vocabulary must be empty")

        object.__setattr__(self, "columns", MappingProxyType(normalized))
        object.__setattr__(
            self,
            "column_metadata",
            _freeze(self.column_metadata),
        )
        object.__setattr__(
            self,
            "output_semantics",
            _freeze(self.output_semantics),
        )

    @property
    def num_rows(self) -> int:
        """Return the exact aligned prediction row count."""
        return self.identity.num_rows

    @property
    def rows(self) -> tuple[tuple[int | str, ...], ...]:
        """Return canonical identity rows in payload order."""
        return self.identity.rows


__all__ = ["PredictionColumn", "PredictionIdentity", "PredictionPayload"]
