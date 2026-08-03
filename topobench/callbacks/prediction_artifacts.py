"""Bounded publication of selected-checkpoint prediction artifacts."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import quote

import numpy as np
import torch
from lightning import Callback

from topobench.evaluator import (
    AbstractEvaluator,
    EvaluationBatch,
    EvaluationContext,
    EvaluationResult,
)

_METRICS_SCHEMA = "topobench-selected-checkpoint-metrics-v1"
_MANIFEST_SCHEMA = "topobench-selected-checkpoint-predictions-v1"
_REQUIRED_COLUMNS = ("target", "raw_output", "prediction")
_REQUIRED_PROVENANCE = (
    "source_fingerprint",
    "dataset_fingerprint",
    "split_fingerprint",
    "model_fingerprint",
    "transform_fingerprint",
)
_RESERVED_COLUMN_SCHEMA_KEYS = frozenset({"name", "dtype", "shape_tail"})
_ALLOWED_ARRAY_KINDS = frozenset({"b", "i", "u", "f", "S", "U"})
_HEX_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    """Content-addressed descriptor for one immutable local artifact file."""

    path: Path
    sha256: str
    byte_size: int
    registration_name: str

    def __post_init__(self) -> None:
        path = Path(self.path)
        _require_digest("sha256", self.sha256)
        if (
            isinstance(self.byte_size, bool)
            or not isinstance(self.byte_size, int)
            or self.byte_size < 0
        ):
            raise ValueError("byte_size must be a non-negative integer")
        if (
            not isinstance(self.registration_name, str)
            or not self.registration_name
        ):
            raise ValueError("registration_name must be a non-empty string")
        object.__setattr__(self, "path", path)


@dataclass(frozen=True, slots=True)
class SplitPublication:
    """The complete, immutable publication for one evaluation split."""

    split: str
    metrics_file: ArtifactFile
    manifest_file: ArtifactFile
    shard_files: tuple[ArtifactFile, ...]
    num_examples: int
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        if self.split not in {"val", "test"}:
            raise ValueError("split publication must be for 'val' or 'test'")
        if not isinstance(self.metrics_file, ArtifactFile) or not isinstance(
            self.manifest_file, ArtifactFile
        ):
            raise TypeError(
                "metrics_file and manifest_file must be ArtifactFile values"
            )
        shards = tuple(self.shard_files)
        if not all(isinstance(shard, ArtifactFile) for shard in shards):
            raise TypeError(
                "shard_files must contain only ArtifactFile values"
            )
        if (
            isinstance(self.num_examples, bool)
            or not isinstance(self.num_examples, int)
            or self.num_examples < 0
        ):
            raise ValueError("num_examples must be a non-negative integer")
        _require_digest("checkpoint_sha256", self.checkpoint_sha256)
        descriptors = (self.metrics_file, self.manifest_file, *shards)
        paths = [descriptor.path for descriptor in descriptors]
        names = [descriptor.registration_name for descriptor in descriptors]
        if len(paths) != len(set(paths)):
            raise ValueError("publication file paths must be distinct")
        if len(names) != len(set(names)):
            raise ValueError("artifact registration names must be distinct")
        object.__setattr__(self, "shard_files", shards)


@dataclass(frozen=True, slots=True)
class _ColumnSpec:
    name: str
    dtype: np.dtype[Any]
    shape_tail: tuple[int, ...]
    metadata: Mapping[str, Any]

    @property
    def row_bytes(self) -> int:
        return self.dtype.itemsize * math.prod(self.shape_tail)


@dataclass(frozen=True, slots=True)
class _SliceSpec:
    max_categories: int
    min_rows: int
    vocabulary: tuple[str, ...] | None


@dataclass(slots=True)
class _CaptureState:
    context: EvaluationContext
    checkpoint_path: Path
    checkpoint_relative_path: str
    checkpoint_sha256: str
    checkpoint_epoch: int
    checkpoint_global_step: int
    staging_root: Path
    prediction_root: Path
    lock_path: Path
    lock_descriptor: int | None
    identity_db: sqlite3.Connection | None
    identity_db_path: Path
    column_specs: tuple[_ColumnSpec, ...] = ()
    identity_columns: tuple[str, ...] = ()
    identity_key: tuple[str, ...] = ()
    output_semantics: Mapping[str, Any] | None = None
    buffer_capacity: int = 0
    buffer: dict[str, np.ndarray[Any, Any]] | None = None
    buffered_rows: int = 0
    observed_rows: int = 0
    flushed_rows: int = 0
    shard_records: list[dict[str, Any]] = field(default_factory=list)
    shard_descriptors: list[ArtifactFile] = field(default_factory=list)
    slice_context: EvaluationContext | None = None
    slice_evaluators: dict[str, dict[str, AbstractEvaluator]] = field(
        default_factory=dict
    )
    slice_counts: dict[str, dict[str, int]] = field(default_factory=dict)


class SelectedCheckpointArtifactCallback(Callback):
    """Model-bound sink for authoritative selected-checkpoint evaluation rows.

    The model calls :meth:`begin`, :meth:`update`, and :meth:`finalize` around
    the already-existing selected-checkpoint evaluator lifecycle. This callback
    deliberately defines no Lightning batch-output hooks: it consumes the same
    canonical :class:`EvaluationBatch` that owns evaluator participation.
    """

    def __init__(
        self,
        *,
        run_root: str | Path,
        root: str | Path,
        shard_rows: int,
        shard_bytes: int,
        metadata_fields: tuple[str, ...] = (),
        evaluation_slices: Mapping[str, object] | None = None,
        distributed_policy: str = "reject",
        existing_artifact_policy: str = "verify_identical",
    ) -> None:
        super().__init__()
        self.run_root = Path(run_root).expanduser().resolve()
        self.root = Path(root).expanduser().resolve()
        try:
            self.root.relative_to(self.run_root)
        except ValueError as error:
            raise ValueError(
                "artifact root must be inside run_root"
            ) from error
        if self.root == self.run_root:
            raise ValueError("artifact root must not equal run_root")
        self.shard_rows = _positive_int("shard_rows", shard_rows)
        self.shard_bytes = _positive_int("shard_bytes", shard_bytes)
        if not isinstance(metadata_fields, tuple):
            metadata_fields = tuple(metadata_fields)
        if any(
            not isinstance(name, str) or not name for name in metadata_fields
        ):
            raise ValueError("metadata_fields must contain non-empty strings")
        if len(metadata_fields) != len(set(metadata_fields)):
            raise ValueError("metadata_fields must not contain duplicates")
        if any(name in _REQUIRED_COLUMNS for name in metadata_fields):
            raise ValueError(
                "metadata_fields must not contain required output columns"
            )
        if distributed_policy != "reject":
            raise ValueError("distributed_policy must be 'reject'")
        if existing_artifact_policy != "verify_identical":
            raise ValueError(
                "existing_artifact_policy must be 'verify_identical'"
            )
        self.metadata_fields = metadata_fields
        self.evaluation_slices = _normalize_slice_specs(
            evaluation_slices,
            metadata_fields,
        )
        self.distributed_policy = distributed_policy
        self.existing_artifact_policy = existing_artifact_policy
        self._slice_evaluator_factory: (
            Callable[[], AbstractEvaluator] | None
        ) = None
        self._state: _CaptureState | None = None
        self._publications: dict[str, SplitPublication] = {}
        self._publication_view: Mapping[str, SplitPublication] = (
            MappingProxyType(self._publications)
        )
        self._selected_checkpoint: tuple[str, str, int, int] | None = None

    @property
    def publications(self) -> Mapping[str, SplitPublication]:
        """Completed split publications, exposed as an immutable live view."""

        return self._publication_view

    def configure_slice_evaluator_factory(
        self,
        factory: Callable[[], AbstractEvaluator],
    ) -> None:
        """Bind the evaluator constructor used by bounded metric slices."""

        if self._state is not None:
            raise RuntimeError(
                "slice evaluator factory cannot change during capture"
            )
        if not callable(factory):
            raise TypeError("slice evaluator factory must be callable")
        self._slice_evaluator_factory = factory

    def begin(
        self,
        context: EvaluationContext,
        *,
        checkpoint_path: str | Path,
        checkpoint_sha256: str,
        checkpoint_epoch: int,
        checkpoint_global_step: int,
        world_size: int,
        global_rank: int,
    ) -> None:
        """Begin one selected-checkpoint split capture after full preflight."""

        if self._state is not None:
            raise RuntimeError("an artifact capture is already active")
        if not isinstance(context, EvaluationContext):
            raise TypeError("context must be an EvaluationContext")
        if context.pass_kind != "selected_checkpoint":
            raise ValueError("artifacts require a selected_checkpoint context")
        if context.split not in {"val", "test"}:
            raise ValueError(
                "selected-checkpoint artifacts support val and test"
            )
        if context.expected_num_examples is None:
            raise ValueError(
                "expected_num_examples is required for artifact coverage"
            )
        _positive_int("expected_num_examples", context.expected_num_examples)
        if self.evaluation_slices and self._slice_evaluator_factory is None:
            raise RuntimeError(
                "configured evaluation slices require an evaluator factory"
            )
        actual_world_size = _positive_int("world_size", world_size)
        actual_rank = _non_negative_int("global_rank", global_rank)
        if actual_world_size != 1 or actual_rank != 0:
            raise RuntimeError(
                "distributed multi-rank artifact capture is unsupported; "
                f"world_size={actual_world_size}, global_rank={actual_rank}"
            )
        epoch = _non_negative_int("checkpoint_epoch", checkpoint_epoch)
        global_step = _non_negative_int(
            "checkpoint_global_step", checkpoint_global_step
        )
        _require_digest("checkpoint_sha256", checkpoint_sha256)
        checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
        if not checkpoint.is_file():
            raise ValueError("checkpoint_path must identify a readable file")
        try:
            checkpoint_relative = checkpoint.relative_to(
                self.run_root
            ).as_posix()
        except ValueError as error:
            raise ValueError(
                "checkpoint_path must be inside run_root"
            ) from error
        actual_sha256 = _sha256(checkpoint)
        if actual_sha256 != checkpoint_sha256:
            raise ValueError(
                "checkpoint SHA-256 does not match checkpoint contents"
            )
        if context.checkpoint_id != checkpoint_sha256:
            raise ValueError(
                "evaluation context checkpoint identity does not match"
            )

        checkpoint_identity = (
            checkpoint_relative,
            checkpoint_sha256,
            epoch,
            global_step,
        )
        if (
            self._selected_checkpoint is not None
            and self._selected_checkpoint != checkpoint_identity
        ):
            raise RuntimeError(
                "val and test artifacts must use the same selected checkpoint"
            )

        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.run_root / (
            f".best-checkpoint-{context.split}.capture.lock"
        )
        lock_descriptor = _acquire_capture_lock(lock_path, context.split)
        identity_db: sqlite3.Connection | None = None
        try:
            staging_root = Path(
                tempfile.mkdtemp(
                    prefix=f".{context.split}.staging-",
                    dir=self.root,
                )
            )
            prediction_root = staging_root / "predictions"
            prediction_root.mkdir()
            identity_db_path = staging_root / ".identity-uniqueness.sqlite3"
            identity_db = sqlite3.connect(identity_db_path)
            identity_db.execute("PRAGMA cache_size = -1024")
            identity_db.execute("PRAGMA temp_store = FILE")
            identity_db.execute("PRAGMA journal_mode = DELETE")
            identity_db.execute(
                "CREATE TABLE identities (identity BLOB PRIMARY KEY) WITHOUT ROWID"
            )
            identity_db.commit()
        except BaseException:
            if identity_db is not None:
                identity_db.close()
            _release_lock_descriptor(lock_descriptor)
            if "staging_root" in locals():
                shutil.rmtree(staging_root, ignore_errors=True)
            raise

        self._state = _CaptureState(
            context=context,
            checkpoint_path=checkpoint,
            checkpoint_relative_path=checkpoint_relative,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_epoch=epoch,
            checkpoint_global_step=global_step,
            staging_root=staging_root,
            prediction_root=prediction_root,
            lock_path=lock_path,
            lock_descriptor=lock_descriptor,
            identity_db=identity_db,
            identity_db_path=identity_db_path,
        )
        self._selected_checkpoint = checkpoint_identity

    def update(self, batch: EvaluationBatch) -> None:
        """Copy one canonical evaluator batch to the bounded CPU shard buffer."""

        state = self._require_state()
        if not isinstance(batch, EvaluationBatch):
            raise TypeError("batch must be an EvaluationBatch")
        if batch.context != state.context:
            raise ValueError(
                "evaluation batch context does not match active capture"
            )
        payload = getattr(batch, "prediction_payload", None)
        if payload is None:
            raise ValueError("evaluation batch has no prediction_payload")
        if (
            state.observed_rows + batch.num_examples
            > state.context.expected_num_examples
        ):
            raise ValueError("prediction rows exceed expected identity count")

        arrays, specs, identity_columns, identity_key, semantics = (
            self._batch_arrays(batch, payload)
        )
        if not state.column_specs:
            self._initialize_schema(
                state,
                specs,
                identity_columns,
                identity_key,
                semantics,
            )
            self._initialize_slices(state)
        else:
            self._validate_schema(
                state,
                specs,
                identity_columns,
                identity_key,
                semantics,
            )

        ordinal = arrays["split_ordinal"]
        expected_ordinals = np.arange(
            state.observed_rows,
            state.observed_rows + batch.num_examples,
            dtype=ordinal.dtype,
        )
        if not np.array_equal(ordinal, expected_ordinals):
            raise ValueError(
                "split_ordinal must preserve exact canonical evaluation order"
            )
        self._update_slices(state, batch, arrays)
        self._record_identities(state, arrays, batch.num_examples)

        cursor = 0
        while cursor < batch.num_examples:
            if state.buffer is None:
                state.buffer = {
                    spec.name: np.empty(
                        (state.buffer_capacity, *spec.shape_tail),
                        dtype=spec.dtype,
                    )
                    for spec in state.column_specs
                }
            available = state.buffer_capacity - state.buffered_rows
            take = min(available, batch.num_examples - cursor)
            destination = slice(
                state.buffered_rows, state.buffered_rows + take
            )
            source = slice(cursor, cursor + take)
            for spec in state.column_specs:
                state.buffer[spec.name][destination] = arrays[spec.name][
                    source
                ]
            state.buffered_rows += take
            state.observed_rows += take
            cursor += take
            if state.buffered_rows == state.buffer_capacity:
                self._flush(state)

    def finalize(self, result: EvaluationResult) -> SplitPublication:
        """Validate and atomically promote the complete split publication."""

        state = self._require_state()
        if not isinstance(result, EvaluationResult):
            raise TypeError("result must be an EvaluationResult")
        if result.context != state.context:
            raise ValueError(
                "evaluation result context does not match active capture"
            )
        if _sha256(state.checkpoint_path) != state.checkpoint_sha256:
            raise RuntimeError(
                "selected checkpoint contents changed during evaluation"
            )
        if state.buffered_rows:
            self._flush(state)

        expected_rows = state.context.expected_num_examples
        assert expected_rows is not None
        if result.num_examples != state.observed_rows:
            raise ValueError(
                "EvaluationResult num_examples does not equal observed prediction count"
            )
        if state.observed_rows != expected_rows:
            raise ValueError(
                "observed identity count does not equal expected_num_examples"
            )
        if not state.column_specs:
            raise ValueError("no prediction rows were captured")
        database_count = self._identity_count(state)
        if database_count != state.observed_rows:
            raise ValueError(
                "identity uniqueness evidence has the wrong row count"
            )

        provenance = _validated_provenance(
            result.provenance, result.num_examples
        )
        aggregate_metrics = _result_metric_document(result)
        slices = self._finalize_slices(state)
        checkpoint_document = {
            "path": state.checkpoint_relative_path,
            "sha256": state.checkpoint_sha256,
            "epoch": state.checkpoint_epoch,
            "global_step": state.checkpoint_global_step,
        }
        metrics_document = {
            "schema_version": _METRICS_SCHEMA,
            "split": state.context.split,
            "checkpoint": checkpoint_document,
            **aggregate_metrics,
            "provenance": provenance,
            **({"slices": slices} if slices else {}),
        }
        manifest_document = {
            "schema_version": _MANIFEST_SCHEMA,
            "split": state.context.split,
            "task": state.context.task,
            "checkpoint": checkpoint_document,
            "output_semantics": dict(state.output_semantics),
            "provenance": provenance,
            "identity": {
                "key": list(state.identity_key),
                "unique": True,
                "order": "split_ordinal",
            },
            "columns": [
                self._column_document(spec) for spec in state.column_specs
            ],
            "expected_rows": expected_rows,
            "observed_rows": state.observed_rows,
            "shards": state.shard_records,
            "status": "complete",
            "writer": {
                "format": "npz",
                "compression": "none",
                "shard_rows": self.shard_rows,
                "shard_bytes": self.shard_bytes,
                "metadata_fields": list(self.metadata_fields),
                "evaluation_slices": {
                    field: {
                        "max_categories": spec.max_categories,
                        "min_rows": spec.min_rows,
                        "vocabulary": list(
                            state.slice_evaluators.get(field, {})
                        ),
                    }
                    for field, spec in self.evaluation_slices.items()
                },
                "distributed_policy": self.distributed_policy,
                "existing_artifact_policy": self.existing_artifact_policy,
            },
        }

        self._close_identity_database(state, remove=True)
        self._validate_staged_shards(state)
        metrics_path = state.staging_root / "metrics.json"
        manifest_path = state.prediction_root / "manifest.json"
        _write_json(metrics_path, metrics_document)
        _write_json(manifest_path, manifest_document)

        publication = SplitPublication(
            split=state.context.split,
            metrics_file=_descriptor(
                metrics_path,
                f"best-checkpoint-{state.context.split}-metrics",
            ),
            manifest_file=_descriptor(
                manifest_path,
                f"best-checkpoint-{state.context.split}-predictions-manifest",
            ),
            shard_files=tuple(state.shard_descriptors),
            num_examples=result.num_examples,
            checkpoint_sha256=state.checkpoint_sha256,
        )
        completed = self._promote_or_verify(state, publication)
        self._publications[state.context.split] = completed
        self._state = None
        return completed

    def abort(self) -> None:
        """Discard the active split staging tree without exposing partial output."""

        state = self._state
        if state is None:
            return
        try:
            self._abort_slices(state)
            self._close_identity_database(state, remove=False)
            shutil.rmtree(state.staging_root, ignore_errors=True)
        finally:
            self._release_capture_lock(state)
            self._state = None

    def _batch_arrays(
        self,
        batch: EvaluationBatch,
        payload: Any,
    ) -> tuple[
        dict[str, np.ndarray[Any, Any]],
        tuple[_ColumnSpec, ...],
        tuple[str, ...],
        tuple[str, ...],
        Mapping[str, Any],
    ]:
        identity = getattr(payload, "identity", None)
        identity_mapping = getattr(identity, "columns", None)
        identity_key_value = getattr(identity, "key", None)
        payload_mapping = getattr(payload, "columns", None)
        metadata_mapping = getattr(payload, "column_metadata", None)
        semantics_value = getattr(payload, "output_semantics", None)
        if not isinstance(identity_mapping, Mapping):
            raise TypeError("prediction identity columns must be a mapping")
        if not isinstance(payload_mapping, Mapping):
            raise TypeError("prediction payload columns must be a mapping")
        if not isinstance(metadata_mapping, Mapping):
            raise TypeError("prediction column_metadata must be a mapping")
        if not isinstance(semantics_value, Mapping):
            raise TypeError("prediction output_semantics must be a mapping")

        identity_columns = tuple(identity_mapping)
        identity_key = tuple(identity_key_value or ())
        if "split_ordinal" not in identity_columns:
            raise ValueError("prediction identity must declare split_ordinal")
        if not identity_key or any(
            name not in identity_columns for name in identity_key
        ):
            raise ValueError(
                "prediction identity key must reference identity columns"
            )
        if len(identity_key) != len(set(identity_key)):
            raise ValueError(
                "prediction identity key must not contain duplicates"
            )
        if set(identity_columns).intersection(payload_mapping) or (
            "prediction" in identity_columns
        ):
            raise ValueError(
                "identity and prediction column names must be distinct"
            )
        if "prediction" in payload_mapping:
            raise ValueError("prediction must use the dedicated payload field")
        for required in ("target", "raw_output"):
            if required not in payload_mapping:
                raise ValueError(
                    f"prediction payload is missing required {required!r}"
                )

        declared = set(payload_mapping) | {"prediction"}
        if set(metadata_mapping) != declared:
            raise ValueError(
                "column_metadata must declare exactly every prediction column"
            )
        metadata_documents: dict[str, Mapping[str, Any]] = {}
        for name in metadata_mapping:
            value = metadata_mapping[name]
            if not isinstance(value, Mapping):
                raise TypeError(
                    f"column metadata for {name!r} must be a mapping"
                )
            if _RESERVED_COLUMN_SCHEMA_KEYS.intersection(value):
                raise ValueError(
                    f"column metadata for {name!r} overrides reserved schema keys"
                )
            document = _json_mapping(value, f"column metadata for {name!r}")
            metadata_documents[name] = document

        expected_roles = {
            "target": "target",
            "raw_output": "raw_output",
            "prediction": "prediction",
        }
        for name, role in expected_roles.items():
            if metadata_documents[name].get("role") != role:
                raise ValueError(f"column {name!r} must declare role {role!r}")
        declared_metadata = {
            name
            for name, document in metadata_documents.items()
            if document.get("role") == "metadata"
        }
        if declared_metadata != set(self.metadata_fields):
            raise ValueError(
                "payload metadata columns must exactly match configured metadata_fields"
            )

        optional_columns = tuple(
            name
            for name in metadata_mapping
            if name not in _REQUIRED_COLUMNS
            and name not in self.metadata_fields
        )
        ordered_names = (
            *identity_columns,
            *_REQUIRED_COLUMNS,
            *optional_columns,
            *self.metadata_fields,
        )
        if len(ordered_names) != len(set(ordered_names)):
            raise ValueError("prediction column names must be unique")

        raw_columns: dict[str, Any] = dict(identity_mapping)
        raw_columns.update(payload_mapping)
        raw_columns["prediction"] = getattr(payload, "prediction", None)
        arrays: dict[str, np.ndarray[Any, Any]] = {}
        specs: list[_ColumnSpec] = []
        for name in ordered_names:
            array = _to_cpu_array(name, raw_columns[name], batch.num_examples)
            arrays[name] = array
            if name in identity_columns:
                role = (
                    "ordinal"
                    if name == "split_ordinal"
                    else "identity"
                    if name in identity_key
                    else "identity_auxiliary"
                )
                column_metadata: Mapping[str, Any] = MappingProxyType(
                    {"role": role}
                )
            else:
                column_metadata = metadata_documents[name]
            specs.append(
                _ColumnSpec(
                    name=name,
                    dtype=array.dtype,
                    shape_tail=tuple(array.shape[1:]),
                    metadata=column_metadata,
                )
            )

        semantics = _json_mapping(semantics_value, "output_semantics")
        if semantics.get("task") != batch.context.task:
            raise ValueError(
                "output_semantics task does not match evaluation context"
            )
        self._validate_task_columns(batch.context, arrays)
        self._validate_values(arrays, identity_columns)
        return arrays, tuple(specs), identity_columns, identity_key, semantics

    def _validate_task_columns(
        self,
        context: EvaluationContext,
        arrays: Mapping[str, np.ndarray[Any, Any]],
    ) -> None:
        raw_shape = arrays["raw_output"].shape[1:]
        prediction_shape = arrays["prediction"].shape[1:]
        if raw_shape != prediction_shape:
            raise ValueError(
                "raw_output and prediction shapes must match exactly"
            )
        if context.task == "classification" and (
            not raw_shape or raw_shape[-1] != context.num_classes
        ):
            raise ValueError(
                "classification outputs must expose the exact class dimension"
            )
        if (
            context.task == "regression"
            and arrays["target"].shape[1:] != raw_shape
        ):
            raise ValueError(
                "regression target and output shapes must match without broadcasting"
            )

    def _validate_values(
        self,
        arrays: Mapping[str, np.ndarray[Any, Any]],
        identity_columns: tuple[str, ...],
    ) -> None:
        for name, array in arrays.items():
            if array.dtype.kind in {"f", "c"} and not np.isfinite(array).all():
                raise ValueError(f"column {name!r} contains non-finite values")
        for name in identity_columns:
            array = arrays[name]
            if array.dtype.kind in {"S", "U"} and np.any(
                np.char.str_len(array) == 0
            ):
                raise ValueError(
                    f"identity column {name!r} contains missing values"
                )
        ordinal = arrays["split_ordinal"]
        if ordinal.dtype.kind not in {"i", "u"} or ordinal.ndim != 1:
            raise ValueError(
                "split_ordinal must be a one-dimensional integer column"
            )

    def _initialize_slices(self, state: _CaptureState) -> None:
        if not self.evaluation_slices:
            return
        factory = self._slice_evaluator_factory
        if factory is None:
            raise RuntimeError(
                "configured evaluation slices require an evaluator factory"
            )
        columns = {spec.name: spec for spec in state.column_specs}
        context = replace(state.context, expected_num_examples=None)
        created: list[AbstractEvaluator] = []
        seen_evaluators: set[int] = set()
        try:
            for field, slice_spec in self.evaluation_slices.items():
                column = columns[field]
                vocabulary = _slice_vocabulary(
                    column.metadata.get("vocabulary"),
                    field,
                )
                if len(vocabulary) > slice_spec.max_categories:
                    raise ValueError(
                        f"evaluation slice {field!r} vocabulary exceeds "
                        "max_categories"
                    )
                if (
                    slice_spec.vocabulary is not None
                    and vocabulary != slice_spec.vocabulary
                ):
                    raise ValueError(
                        f"evaluation slice {field!r} vocabulary does not "
                        "match configuration"
                    )
                evaluators: dict[str, AbstractEvaluator] = {}
                for value in vocabulary:
                    evaluator = factory()
                    if not isinstance(evaluator, AbstractEvaluator):
                        raise TypeError(
                            "slice evaluator factory must return an "
                            "AbstractEvaluator"
                        )
                    if id(evaluator) in seen_evaluators:
                        raise ValueError(
                            "slice evaluator factory must return a fresh "
                            "evaluator"
                        )
                    seen_evaluators.add(id(evaluator))
                    evaluator.begin(context)
                    created.append(evaluator)
                    evaluators[value] = evaluator
                state.slice_evaluators[field] = evaluators
                state.slice_counts[field] = {value: 0 for value in vocabulary}
        except BaseException:
            for evaluator in reversed(created):
                evaluator.abort()
            raise
        state.slice_context = context

    def _update_slices(
        self,
        state: _CaptureState,
        batch: EvaluationBatch,
        arrays: Mapping[str, np.ndarray[Any, Any]],
    ) -> None:
        context = state.slice_context
        if context is None:
            return
        for field_name, evaluators in state.slice_evaluators.items():
            values = arrays[field_name]
            if values.ndim != 1 or values.dtype.kind not in {"S", "U"}:
                raise ValueError(
                    f"evaluation slice {field_name!r} must be a string vector"
                )
            observed = {str(value) for value in np.unique(values)}
            unknown = observed.difference(evaluators)
            if unknown:
                raise ValueError(
                    f"evaluation slice {field_name!r} contains undeclared "
                    f"categories: {sorted(unknown)!r}"
                )
            for value, evaluator in evaluators.items():
                mask_array = values == value
                count = int(mask_array.sum())
                if count == 0:
                    continue
                mask = torch.as_tensor(
                    mask_array,
                    dtype=torch.bool,
                    device=batch.outputs.device,
                )
                evaluator.update(
                    EvaluationBatch(
                        outputs=batch.outputs[mask],
                        targets=batch.targets[mask],
                        num_examples=count,
                        context=context,
                        sequence_id=batch.sequence_id,
                    )
                )
                state.slice_counts[field_name][value] += count

    def _finalize_slices(
        self,
        state: _CaptureState,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        document: dict[str, dict[str, dict[str, Any]]] = {}
        for field_name, evaluators in state.slice_evaluators.items():
            field_document: dict[str, dict[str, Any]] = {}
            minimum = self.evaluation_slices[field_name].min_rows
            for value, evaluator in evaluators.items():
                count = state.slice_counts[field_name][value]
                if count < minimum:
                    evaluator.abort()
                    continue
                result = evaluator.finalize()
                if result.num_examples != count:
                    raise RuntimeError(
                        f"evaluation slice {field_name!r}/{value!r} count "
                        "disagrees with its evaluator"
                    )
                field_document[_slice_token(value)] = {
                    "value": value,
                    **_result_metric_document(result),
                }
            document[field_name] = field_document
        return document

    @staticmethod
    def _abort_slices(state: _CaptureState) -> None:
        for evaluators in state.slice_evaluators.values():
            for evaluator in evaluators.values():
                if getattr(evaluator, "state", None) == "idle":
                    continue
                evaluator.abort()

    def _initialize_schema(
        self,
        state: _CaptureState,
        specs: tuple[_ColumnSpec, ...],
        identity_columns: tuple[str, ...],
        identity_key: tuple[str, ...],
        semantics: Mapping[str, Any],
    ) -> None:
        bytes_per_row = sum(spec.row_bytes for spec in specs)
        if bytes_per_row <= 0:
            raise ValueError(
                "prediction rows must have positive in-memory size"
            )
        state.column_specs = specs
        state.identity_columns = identity_columns
        state.identity_key = identity_key
        state.output_semantics = semantics
        state.buffer_capacity = min(
            self.shard_rows,
            max(1, self.shard_bytes // bytes_per_row),
        )

    def _validate_schema(
        self,
        state: _CaptureState,
        specs: tuple[_ColumnSpec, ...],
        identity_columns: tuple[str, ...],
        identity_key: tuple[str, ...],
        semantics: Mapping[str, Any],
    ) -> None:
        if (
            identity_columns != state.identity_columns
            or identity_key != state.identity_key
        ):
            raise ValueError(
                "prediction identity schema changed between batches"
            )
        if semantics != state.output_semantics:
            raise ValueError("output_semantics changed between batches")
        if len(specs) != len(state.column_specs):
            raise ValueError(
                "prediction column schema changed between batches"
            )
        for expected, observed in zip(state.column_specs, specs, strict=True):
            if expected != observed:
                raise ValueError(
                    f"prediction column schema changed for {expected.name!r}"
                )

    def _record_identities(
        self,
        state: _CaptureState,
        arrays: Mapping[str, np.ndarray[Any, Any]],
        row_count: int,
    ) -> None:
        database = state.identity_db
        if database is None:
            raise RuntimeError("identity uniqueness database is closed")

        def encoded_rows():
            for row in range(row_count):
                key = bytearray()
                for name in state.identity_key:
                    value = np.ascontiguousarray(arrays[name][row : row + 1])
                    key.extend(value.tobytes(order="C"))
                yield (sqlite3.Binary(bytes(key)),)

        try:
            database.executemany(
                "INSERT INTO identities (identity) VALUES (?)",
                encoded_rows(),
            )
            database.commit()
        except sqlite3.IntegrityError as error:
            database.rollback()
            raise ValueError(
                "duplicate prediction identity violates uniqueness"
            ) from error

    def _identity_count(self, state: _CaptureState) -> int:
        database = state.identity_db
        if database is None:
            raise RuntimeError("identity uniqueness database is closed")
        row = database.execute("SELECT COUNT(*) FROM identities").fetchone()
        if row is None:
            raise RuntimeError(
                "identity uniqueness database returned no count"
            )
        return int(row[0])

    def _flush(self, state: _CaptureState) -> None:
        if state.buffer is None or state.buffered_rows == 0:
            return
        arrays = {
            spec.name: state.buffer[spec.name][: state.buffered_rows]
            for spec in state.column_specs
        }
        self._write_bounded_shards(state, arrays, state.buffered_rows)
        state.buffer = None
        state.buffered_rows = 0

    def _write_bounded_shards(
        self,
        state: _CaptureState,
        arrays: Mapping[str, np.ndarray[Any, Any]],
        row_count: int,
    ) -> None:
        index = len(state.shard_records)
        name = f"part-{index:05d}.npz"
        path = state.prediction_root / name
        _write_npz(path, arrays)
        byte_size = path.stat().st_size
        if byte_size > self.shard_bytes:
            path.unlink()
            if row_count == 1:
                raise ValueError(
                    "single encoded prediction row exceeds configured "
                    f"shard_bytes byte bound: {byte_size} > {self.shard_bytes}"
                )
            midpoint = row_count // 2
            self._write_bounded_shards(
                state,
                {name: array[:midpoint] for name, array in arrays.items()},
                midpoint,
            )
            self._write_bounded_shards(
                state,
                {name: array[midpoint:] for name, array in arrays.items()},
                row_count - midpoint,
            )
            return

        descriptor = _descriptor(
            path,
            f"best-checkpoint-{state.context.split}-predictions-part-{index:05d}",
        )
        row_start = state.flushed_rows
        row_stop = row_start + row_count
        state.shard_records.append(
            {
                "path": name,
                "row_start": row_start,
                "row_stop": row_stop,
                "byte_size": descriptor.byte_size,
                "sha256": descriptor.sha256,
            }
        )
        state.shard_descriptors.append(descriptor)
        state.flushed_rows = row_stop

    def _validate_staged_shards(self, state: _CaptureState) -> None:
        if state.flushed_rows != state.observed_rows:
            raise ValueError("shard row ranges do not cover all observed rows")
        expected_start = 0
        expected_names = [spec.name for spec in state.column_specs]
        for record, descriptor in zip(
            state.shard_records,
            state.shard_descriptors,
            strict=True,
        ):
            if record["row_start"] != expected_start:
                raise ValueError("shard row ranges are not contiguous")
            row_count = record["row_stop"] - record["row_start"]
            if not 0 < row_count <= self.shard_rows:
                raise ValueError("shard row count violates shard_rows")
            if descriptor.byte_size > self.shard_bytes:
                raise ValueError("staged shard byte size violates shard_bytes")
            if descriptor.byte_size != descriptor.path.stat().st_size or (
                descriptor.sha256 != _sha256(descriptor.path)
            ):
                raise RuntimeError(
                    "staged shard checksum or byte size changed"
                )
            with zipfile.ZipFile(descriptor.path) as archive:
                if any(
                    item.compress_type != zipfile.ZIP_STORED
                    for item in archive.infolist()
                ):
                    raise RuntimeError(
                        "prediction shards must be uncompressed"
                    )
            with np.load(descriptor.path, allow_pickle=False) as shard:
                if shard.files != expected_names:
                    raise ValueError("staged shard column order changed")
                for spec in state.column_specs:
                    array = shard[spec.name]
                    if array.dtype != spec.dtype or array.shape != (
                        row_count,
                        *spec.shape_tail,
                    ):
                        raise ValueError(
                            f"staged shard schema changed for {spec.name!r}"
                        )
                    if array.dtype.kind == "O":
                        raise ValueError(
                            "object arrays are forbidden in prediction shards"
                        )
                    if (
                        array.dtype.kind in {"f", "c"}
                        and not np.isfinite(array).all()
                    ):
                        raise ValueError(
                            f"staged shard {spec.name!r} contains non-finite values"
                        )
                ordinal = shard["split_ordinal"]
                expected_ordinal = np.arange(
                    record["row_start"],
                    record["row_stop"],
                    dtype=ordinal.dtype,
                )
                if not np.array_equal(ordinal, expected_ordinal):
                    raise ValueError(
                        "staged shard canonical row order changed"
                    )
            expected_start = record["row_stop"]
        if expected_start != state.observed_rows:
            raise ValueError("shard row coverage is incomplete")

    def _column_document(self, spec: _ColumnSpec) -> dict[str, Any]:
        return {
            "name": spec.name,
            "dtype": spec.dtype.str,
            "shape_tail": list(spec.shape_tail),
            **dict(spec.metadata),
        }

    def _promote_or_verify(
        self,
        state: _CaptureState,
        staged: SplitPublication,
    ) -> SplitPublication:
        final_root = self.root / state.context.split
        if final_root.exists():
            if not final_root.is_dir() or not _identical_trees(
                state.staging_root, final_root
            ):
                self._preserve_conflicting_staging(state)
                raise RuntimeError(
                    f"immutable artifact conflict for split {state.context.split!r}"
                )
            completed = _relocate_publication(
                staged,
                state.staging_root,
                final_root,
            )
            try:
                shutil.rmtree(state.staging_root)
            finally:
                self._release_capture_lock(state)
            return completed

        _sync_populated_directories(state.staging_root)
        try:
            state.staging_root.rename(final_root)
        except FileExistsError as error:
            self._preserve_conflicting_staging(state)
            raise RuntimeError(
                f"immutable artifact collision for split {state.context.split!r}"
            ) from error
        _sync_directory(self.root)
        self._release_capture_lock(state)
        return _relocate_publication(staged, state.staging_root, final_root)

    def _close_identity_database(
        self,
        state: _CaptureState,
        *,
        remove: bool,
    ) -> None:
        if state.identity_db is not None:
            state.identity_db.close()
            state.identity_db = None
        if remove:
            for suffix in ("", "-journal", "-shm", "-wal"):
                path = Path(f"{state.identity_db_path}{suffix}")
                path.unlink(missing_ok=True)

    def _release_capture_lock(self, state: _CaptureState) -> None:
        descriptor = state.lock_descriptor
        if descriptor is None:
            return
        state.lock_descriptor = None
        _release_lock_descriptor(descriptor)

    def _preserve_conflicting_staging(self, state: _CaptureState) -> None:
        self._close_identity_database(state, remove=True)
        self._release_capture_lock(state)
        if self._state is state:
            self._state = None

    def _require_state(self) -> _CaptureState:
        if self._state is None:
            raise RuntimeError(
                "no selected-checkpoint artifact capture is active"
            )
        return self._state


def _acquire_capture_lock(path: Path, split: str) -> int:
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(descriptor)
        raise RuntimeError(
            f"artifact capture for split {split!r} is already locked"
        ) from error
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _release_lock_descriptor(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _normalize_slice_specs(
    value: Mapping[str, object] | None,
    metadata_fields: tuple[str, ...],
) -> Mapping[str, _SliceSpec]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise TypeError("evaluation_slices must be a mapping")
    specs: dict[str, _SliceSpec] = {}
    for field_name, raw_spec in value.items():
        if not isinstance(field_name, str) or not field_name:
            raise ValueError(
                "evaluation_slices keys must be non-empty strings"
            )
        if field_name not in metadata_fields:
            raise ValueError(
                f"evaluation slice {field_name!r} is not an allowlisted metadata field"
            )
        if not isinstance(raw_spec, Mapping):
            raise TypeError(
                f"evaluation slice {field_name!r} must be a mapping"
            )
        required = {"max_categories", "min_rows"}
        optional = {"vocabulary"}
        if not required.issubset(raw_spec) or set(raw_spec).difference(
            required | optional
        ):
            raise ValueError(
                f"evaluation slice {field_name!r} must define max_categories "
                "and min_rows, with optional vocabulary"
            )
        maximum = _positive_int(
            f"evaluation_slices.{field_name}.max_categories",
            raw_spec["max_categories"],
        )
        minimum = _positive_int(
            f"evaluation_slices.{field_name}.min_rows",
            raw_spec["min_rows"],
        )
        configured = raw_spec.get("vocabulary")
        vocabulary = (
            None
            if configured is None
            else _slice_vocabulary(configured, field_name)
        )
        if vocabulary is not None and len(vocabulary) > maximum:
            raise ValueError(
                f"evaluation slice {field_name!r} vocabulary exceeds "
                "max_categories"
            )
        specs[field_name] = _SliceSpec(maximum, minimum, vocabulary)
    return MappingProxyType(specs)


def _slice_vocabulary(value: object, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f"evaluation slice {field!r} requires a declared finite vocabulary"
        )
    vocabulary = tuple(value)
    if not vocabulary or any(
        not isinstance(item, str) or not item for item in vocabulary
    ):
        raise ValueError(
            f"evaluation slice {field!r} vocabulary must contain "
            "non-empty strings"
        )
    if len(vocabulary) != len(set(vocabulary)):
        raise ValueError(
            f"evaluation slice {field!r} vocabulary contains duplicates"
        )
    return vocabulary


def _slice_token(value: str) -> str:
    return quote(value, safe="._-")


def _non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_digest(name: str, value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _to_cpu_array(
    name: str,
    value: Any,
    expected_rows: int,
) -> np.ndarray[Any, Any]:
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise TypeError(f"column {name!r} must be a dense tensor")
        try:
            array = value.detach().to(device="cpu").numpy()
        except (RuntimeError, TypeError) as error:
            raise TypeError(
                f"column {name!r} cannot be represented as a NumPy array"
            ) from error
    elif isinstance(value, np.ndarray):
        array = value
    else:
        raise TypeError(f"column {name!r} must be a tensor or NumPy array")
    array = np.asarray(array)
    if array.ndim == 0 or array.shape[0] != expected_rows:
        raise ValueError(
            f"column {name!r} must have leading dimension {expected_rows}"
        )
    if any(dimension <= 0 for dimension in array.shape[1:]):
        raise ValueError(f"column {name!r} has an empty shape dimension")
    if array.dtype.kind not in _ALLOWED_ARRAY_KINDS or array.dtype.hasobject:
        raise TypeError(
            f"column {name!r} has unsupported or object dtype {array.dtype}"
        )
    return np.ascontiguousarray(array)


def _validated_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for name, value in metrics.items():
        scalar = _json_value(value)
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
            raise TypeError(f"metric {name!r} must serialize as a real scalar")
        if isinstance(scalar, float) and not math.isfinite(scalar):
            raise ValueError(f"metric {name!r} must be finite")
        document[name] = scalar
    return document


def _result_metric_document(result: EvaluationResult) -> dict[str, Any]:
    return {
        "num_examples": result.num_examples,
        "metrics": _validated_metrics(result.metrics),
        "metric_metadata": {
            name: {
                "status": _json_value(result.status.get(name)),
                "support": _json_value(result.support.get(name)),
                "reason": _json_value(result.reason.get(name)),
            }
            for name in result.metrics
        },
    }


def _validated_provenance(
    provenance: Mapping[str, Any],
    num_examples: int,
) -> dict[str, Any]:
    document = dict(_json_mapping(provenance, "provenance"))
    if (
        document.get("num_examples") != num_examples
        or type(document.get("num_examples")) is not int
    ):
        raise ValueError(
            "provenance num_examples must equal EvaluationResult count"
        )
    missing = [name for name in _REQUIRED_PROVENANCE if name not in document]
    if missing:
        raise ValueError(
            f"provenance is missing required fingerprints: {missing}"
        )
    for name in _REQUIRED_PROVENANCE:
        _require_digest(name, document[name])
    return document


def _json_mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    document: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise TypeError(f"{name} keys must be non-empty strings")
        document[key] = _json_value(item)
    return MappingProxyType(document)


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise TypeError("JSON tensor values must be scalar")
        return _json_value(value.detach().cpu().item())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Real):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("JSON numeric values must be finite")
        return result
    if isinstance(value, Mapping):
        document: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and key:
                encoded_key = key
            elif isinstance(key, int) and not isinstance(key, bool):
                encoded_key = str(key)
            else:
                raise TypeError(
                    "JSON mapping keys must be non-empty strings or integers"
                )
            if encoded_key in document:
                raise ValueError(
                    f"JSON mapping key collision after encoding: {encoded_key!r}"
                )
            document[encoded_key] = _json_value(item)
        return document
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise TypeError(
        f"unsupported deterministic JSON value {type(value).__name__}"
    )


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    with path.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _write_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray[Any, Any]],
) -> None:
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for name, array in arrays.items():
            information = zipfile.ZipInfo(
                filename=f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            information.compress_type = zipfile.ZIP_STORED
            information.create_system = 3
            information.external_attr = 0o600 << 16
            with archive.open(
                information, mode="w", force_zip64=True
            ) as member:
                np.lib.format.write_array(
                    member,
                    np.ascontiguousarray(array),
                    allow_pickle=False,
                )
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _descriptor(path: Path, registration_name: str) -> ArtifactFile:
    return ArtifactFile(
        path=path,
        sha256=_sha256(path),
        byte_size=path.stat().st_size,
        registration_name=registration_name,
    )


def _identical_trees(left: Path, right: Path) -> bool:
    left_files = {
        path.relative_to(left): path
        for path in left.rglob("*")
        if path.is_file()
    }
    right_files = {
        path.relative_to(right): path
        for path in right.rglob("*")
        if path.is_file()
    }
    if left_files.keys() != right_files.keys():
        return False
    for relative, left_path in left_files.items():
        right_path = right_files[relative]
        if left_path.stat().st_size != right_path.stat().st_size:
            return False
        if _sha256(left_path) != _sha256(right_path):
            return False
    return True


def _relocate_publication(
    publication: SplitPublication,
    previous_root: Path,
    final_root: Path,
) -> SplitPublication:
    def relocate(descriptor: ArtifactFile) -> ArtifactFile:
        return ArtifactFile(
            path=final_root / descriptor.path.relative_to(previous_root),
            sha256=descriptor.sha256,
            byte_size=descriptor.byte_size,
            registration_name=descriptor.registration_name,
        )

    return SplitPublication(
        split=publication.split,
        metrics_file=relocate(publication.metrics_file),
        manifest_file=relocate(publication.manifest_file),
        shard_files=tuple(
            relocate(shard) for shard in publication.shard_files
        ),
        num_examples=publication.num_examples,
        checkpoint_sha256=publication.checkpoint_sha256,
    )


def _sync_populated_directories(root: Path) -> None:
    for directory, child_directories, file_names in os.walk(
        root, topdown=False
    ):
        if child_directories or file_names:
            _sync_directory(Path(directory))


def _sync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "ArtifactFile",
    "SelectedCheckpointArtifactCallback",
    "SplitPublication",
]
