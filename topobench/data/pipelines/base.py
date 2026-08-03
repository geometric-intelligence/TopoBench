"""Shared interfaces for configuration-driven data pipelines."""

from __future__ import annotations

import hashlib
import math
import struct
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    TypeAlias,
    runtime_checkable,
)

import hydra
import numpy as np
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch_geometric.data import Data, HeteroData

from topobench.data.capabilities import RuntimeDataCapability, qualify_dataset
from topobench.data.heterogeneous import HeterogeneousDataSpec
from topobench.data.loaders.parquet import (
    ParquetTypedGraphLoader,
    ParquetTypedGraphSource,
    ProfilingSpec,
    ReproducibilitySpec,
)
from topobench.data.preprocessor import PreProcessor
from topobench.data.stores.qualification_checks import QualificationReport
from topobench.data.stores.typed_graph_store import (
    TypedGraphStore,
    TypedGraphStoreState,
)
from topobench.transforms.fittable import FittableTransform

if TYPE_CHECKING:
    from topobench.evaluator.prediction import (
        PredictionIdentity,
        PredictionPayload,
    )

Phase: TypeAlias = Literal["train", "val", "test"]
CanonicalPredictionIdentity: TypeAlias = tuple[str, int] | tuple[str, str, int]
_PHASES: tuple[Phase, ...] = ("train", "val", "test")
_PARQUET_LOADER_TARGET = (
    "topobench.data.loaders.parquet.ParquetTypedGraphLoader"
)

_NATIVE_SPLIT_FIELDS = frozenset({"train_mask", "val_mask", "test_mask"})


def _observed_classification_count(
    label_tensors: Iterable[Tensor | np.ndarray],
) -> int:
    """Return the observed zero-based class vocabulary size."""
    maximum: int | None = None
    for labels in label_tensors:
        if isinstance(labels, Tensor):
            if labels.numel() == 0:
                continue
            current = int(labels.max().item())
        elif isinstance(labels, np.ndarray):
            if labels.size == 0:
                continue
            current = int(np.max(labels))
        else:
            raise TypeError(
                "classification labels must be tensors or NumPy arrays"
            )
        maximum = current if maximum is None else max(maximum, current)
    if maximum is None:
        raise ValueError("classification labels must not be empty")
    return maximum + 1


def _fingerprint_frame(
    digest: Any,
    label: str,
    payload: bytes | memoryview,
) -> None:
    """Append one unambiguous labeled field to a streaming digest."""
    encoded_label = label.encode("utf-8")
    digest.update(len(encoded_label).to_bytes(4, "big"))
    digest.update(encoded_label)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _fingerprint_text(digest: Any, label: str, value: str) -> None:
    _fingerprint_frame(digest, label, value.encode("utf-8"))


def _fingerprint_tensor(digest: Any, value: Tensor) -> None:
    """Stream one tensor without constructing a whole-tensor byte string."""
    _fingerprint_text(digest, "value.type", "tensor")
    _fingerprint_text(digest, "tensor.dtype", str(value.dtype))
    _fingerprint_text(digest, "tensor.layout", str(value.layout))
    _fingerprint_frame(
        digest,
        "tensor.shape",
        b"".join(int(size).to_bytes(8, "big") for size in value.shape),
    )
    if value.device.type == "meta":
        raise ValueError("native provenance cannot fingerprint meta tensors")
    if value.is_quantized:
        _fingerprint_text(digest, "tensor.qscheme", str(value.qscheme()))
        if value.qscheme() in {
            torch.per_channel_affine,
            torch.per_channel_affine_float_qparams,
            torch.per_channel_symmetric,
        }:
            _fingerprint_value(digest, value.q_per_channel_scales())
            _fingerprint_value(digest, value.q_per_channel_zero_points())
            _fingerprint_value(digest, value.q_per_channel_axis())
        else:
            _fingerprint_value(digest, value.q_scale())
            _fingerprint_value(digest, value.q_zero_point())
        _fingerprint_value(digest, value.int_repr())
        return
    if value.layout != torch.strided:
        sparse = value.detach().to(device="cpu").to_sparse_coo().coalesce()
        _fingerprint_value(digest, sparse.indices())
        _fingerprint_value(digest, sparse.values())
        return
    contiguous = (
        value.detach()
        .to(device="cpu")
        .resolve_conj()
        .resolve_neg()
        .contiguous()
    )
    if type(contiguous) is not Tensor:
        contiguous = contiguous.as_subclass(Tensor)
    raw = contiguous.reshape(-1).view(torch.uint8).numpy()
    _fingerprint_frame(digest, "tensor.bytes", memoryview(raw).cast("B"))


def _mapping_key_order(value: object) -> tuple[object, ...]:
    """Return a deterministic order for supported structured mapping keys."""
    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", int(value))
    if isinstance(value, Integral):
        return ("int", int(value))
    if isinstance(value, Real):
        return ("float", struct.pack(">d", float(value)))
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, tuple):
        return ("tuple", *(_mapping_key_order(item) for item in value))
    raise TypeError(
        "native provenance mapping keys must be scalar values or tuples"
    )


def _fingerprint_value(digest: Any, value: object) -> None:
    """Stream one deterministic structured value into ``digest``."""
    if isinstance(value, Tensor):
        _fingerprint_tensor(digest, value)
        return
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(
                "native provenance cannot fingerprint object arrays"
            )
        _fingerprint_text(digest, "value.type", "ndarray")
        _fingerprint_text(digest, "array.dtype", value.dtype.str)
        _fingerprint_frame(
            digest,
            "array.shape",
            b"".join(int(size).to_bytes(8, "big") for size in value.shape),
        )
        contiguous = np.ascontiguousarray(value)
        _fingerprint_frame(
            digest,
            "array.bytes",
            memoryview(contiguous).cast("B"),
        )
        return
    if isinstance(value, np.generic):
        _fingerprint_value(digest, value.item())
        return
    if value is None:
        _fingerprint_text(digest, "value.type", "none")
        return
    if isinstance(value, bool):
        _fingerprint_text(digest, "value.type", "bool")
        _fingerprint_frame(digest, "bool", b"\x01" if value else b"\x00")
        return
    if isinstance(value, Integral):
        _fingerprint_text(digest, "value.type", "int")
        _fingerprint_text(digest, "int", str(int(value)))
        return
    if isinstance(value, Real):
        _fingerprint_text(digest, "value.type", "float")
        _fingerprint_frame(digest, "float", struct.pack(">d", float(value)))
        return
    if isinstance(value, str):
        _fingerprint_text(digest, "value.type", "str")
        _fingerprint_text(digest, "str", value)
        return
    if isinstance(value, bytes):
        _fingerprint_text(digest, "value.type", "bytes")
        _fingerprint_frame(digest, "bytes", value)
        return
    if isinstance(value, Mapping):
        _fingerprint_text(digest, "value.type", "mapping")
        items = sorted(
            value.items(), key=lambda item: _mapping_key_order(item[0])
        )
        _fingerprint_value(digest, len(items))
        for key, item in items:
            _fingerprint_value(digest, key)
            _fingerprint_value(digest, item)
        return
    if isinstance(value, Sequence):
        _fingerprint_text(digest, "value.type", "sequence")
        _fingerprint_value(digest, len(value))
        for item in value:
            _fingerprint_value(digest, item)
        return
    raise TypeError(
        "native provenance cannot fingerprint value of type "
        f"{type(value).__name__}"
    )


def _fingerprint_store(
    digest: Any,
    *,
    store_kind: str,
    store_key: object,
    store: object,
) -> None:
    """Hash one PyG store with split masks omitted from content identity."""
    _fingerprint_text(digest, "store.kind", store_kind)
    _fingerprint_value(digest, store_key)
    keys = sorted(
        str(key)
        for key in store.keys()  # type: ignore[attr-defined]  # noqa: SIM118
        if str(key) not in _NATIVE_SPLIT_FIELDS
    )
    _fingerprint_value(digest, len(keys))
    for key in keys:
        _fingerprint_text(digest, "store.key", key)
        _fingerprint_value(
            digest,
            store[key],  # type: ignore[index]
        )


def _fingerprint_data(digest: Any, data: Data | HeteroData) -> None:
    """Hash the sorted public schema and content of one native PyG value."""
    if isinstance(data, HeteroData):
        _fingerprint_text(digest, "data.type", "HeteroData")
        stores: list[tuple[str, object, object]] = []
        for store in data.stores:
            key = getattr(store, "_key", None)
            if key is None:
                stores.append(("global", (), store))
            elif isinstance(key, str):
                stores.append(("node", key, store))
            elif isinstance(key, tuple):
                stores.append(
                    ("edge", tuple(str(item) for item in key), store)
                )
            else:
                raise TypeError(
                    "native heterogeneous stores require canonical string keys"
                )
        stores.sort(key=lambda item: (item[0], _mapping_key_order(item[1])))
        _fingerprint_value(digest, len(stores))
        for store_kind, store_key, store in stores:
            _fingerprint_store(
                digest,
                store_kind=store_kind,
                store_key=store_key,
                store=store,
            )
        return
    if not isinstance(data, Data):
        raise TypeError("native provenance requires PyG Data or HeteroData")
    _fingerprint_text(digest, "data.type", "Data")
    _fingerprint_store(
        digest,
        store_kind="data",
        store_key=(),
        store=data,
    )


def _canonical_identity(
    value: object,
    *,
    field_name: str,
    expected_size: int | None = None,
) -> Tensor:
    """Return one canonical integer identity vector on the CPU."""
    if (
        not isinstance(value, Tensor)
        or value.dtype == torch.bool
        or value.is_floating_point()
        or value.is_complex()
        or value.ndim != 1
        or (expected_size is not None and value.numel() != expected_size)
    ):
        raise ValueError(
            f"{field_name} must be an aligned rank-one integer tensor"
        )
    identities = value.detach().to(device="cpu", dtype=torch.long)
    return identities


def _native_provenance_fingerprints(
    phase_datasets: Mapping[str, object],
    *,
    output_kind: str,
    target_node_type: str = "",
) -> Mapping[str, str]:
    """Fingerprint observed native content and canonical split membership."""
    if frozenset(phase_datasets) != frozenset(_PHASES):
        raise ValueError(
            f"native phase datasets must contain exactly {_PHASES!r}"
        )
    content_digest = hashlib.sha256()
    split_digest = hashlib.sha256()
    _fingerprint_text(content_digest, "convention", "native-content-v1")
    _fingerprint_text(content_digest, "output.kind", output_kind)
    _fingerprint_text(split_digest, "convention", "native-split-v1")
    _fingerprint_text(split_digest, "output.kind", output_kind)

    if output_kind == "graph":
        records: list[tuple[int, bytes]] = []
        for phase in _PHASES:
            dataset = phase_datasets[phase]
            if dataset is None:
                raise ValueError(f"{phase} split must not be empty")
            _fingerprint_text(split_digest, "phase", phase)
            dataset_size = len(dataset)  # type: ignore[arg-type]
            _fingerprint_value(split_digest, dataset_size)
            for index in range(dataset_size):
                data = dataset[index]  # type: ignore[index]
                if not isinstance(data, Data):
                    raise TypeError(
                        "native graph provenance requires PyG Data samples"
                    )
                sample_id = _canonical_identity(
                    getattr(data, "sample_id", None),
                    field_name="sample_id",
                    expected_size=1,
                )
                identity = int(sample_id.item())
                _fingerprint_value(split_digest, identity)
                record = hashlib.sha256()
                _fingerprint_text(record, "convention", "native-record-v1")
                _fingerprint_data(record, data)
                records.append((identity, record.digest()))
        records.sort(key=lambda item: item[0])
        _fingerprint_value(content_digest, len(records))
        for identity, observed_data in records:
            _fingerprint_value(content_digest, identity)
            _fingerprint_frame(
                content_digest,
                "observed.data",
                observed_data,
            )
    else:
        train = phase_datasets["train"]
        if train is None or len(train) != 1:  # type: ignore[arg-type]
            raise ValueError(
                "native node provenance requires one shared graph"
            )
        data = train[0]  # type: ignore[index]
        if output_kind == "heterogeneous":
            if not isinstance(data, HeteroData):
                raise TypeError(
                    "heterogeneous provenance requires native HeteroData"
                )
            if target_node_type not in data.node_types:
                raise ValueError(
                    f"missing target node type {target_node_type!r}"
                )
            target_store = data[target_node_type]
            mask_store = target_store
            count = int(target_store.num_nodes)
            identities = _canonical_identity(
                target_store.get("n_id"),
                field_name=f"{target_node_type}.n_id",
                expected_size=count,
            )
        else:
            if not isinstance(data, Data):
                raise TypeError(
                    f"{output_kind} provenance requires native PyG Data"
                )
            count = int(data.num_nodes)
            mask_store = data
            identities = _canonical_identity(
                getattr(data, "global_nid", None),
                field_name="global_nid",
                expected_size=count,
            )
        _fingerprint_value(content_digest, 1)
        _fingerprint_data(content_digest, data)
        for phase in _PHASES:
            mask = mask_store.get(f"{phase}_mask")
            if (
                not isinstance(mask, Tensor)
                or mask.dtype != torch.bool
                or mask.ndim != 1
                or mask.numel() != identities.numel()
            ):
                raise ValueError(
                    f"{phase}_mask must align to canonical identities"
                )
            members = identities[mask.detach().to(device="cpu")]
            _fingerprint_text(split_digest, "phase", phase)
            _fingerprint_value(split_digest, members)

    observed_content = content_digest.digest()
    source = hashlib.sha256(b"topobench-native-source-v1\0")
    source.update(observed_content)
    dataset = hashlib.sha256(b"topobench-native-dataset-v1\0")
    dataset.update(observed_content)
    return MappingProxyType(
        {
            "source_fingerprint": source.hexdigest(),
            "dataset_fingerprint": dataset.hexdigest(),
            "split_fingerprint": split_digest.hexdigest(),
        }
    )


def is_parquet_typed_graph_config(cfg: DictConfig) -> bool:
    """Select only the exact packaged immutable Parquet descriptor."""
    return (
        OmegaConf.select(cfg, "dataset.loader._target_", default=None)
        == _PARQUET_LOADER_TARGET
    )


def native_prediction_row_adapter(
    cfg: DictConfig,
    *,
    output_kind: str,
    target_node_type: str = "",
    sampling_strategy: str,
) -> tuple[str, PredictionRowAdapter]:
    """Build one deterministic native adapter from declared dataset semantics."""
    source_name = OmegaConf.select(
        cfg,
        "dataset.loader.parameters.data_name",
        default=None,
    )
    if not isinstance(source_name, str) or not source_name:
        source_name = OmegaConf.select(cfg, "dataset.selector", default=None)
    if not isinstance(source_name, str) or not source_name:
        raise ValueError(
            "native prediction rows require a declared non-empty data_name"
        )
    requested_metadata = tuple(
        OmegaConf.select(
            cfg,
            "evaluation_artifacts.metadata_fields",
            default=(),
        )
        or ()
    )
    if set(requested_metadata).difference({"source"}):
        raise ValueError(
            "native prediction metadata is restricted to the source allowlist"
        )
    parameters = cfg.dataset.parameters
    adapter = PredictionRowAdapter(
        source_graph_id=source_name,
        output_kind=output_kind,
        target_node_type=target_node_type,
        sampling_strategy=sampling_strategy,
        task=str(parameters.get("task", "classification")),
        task_level=str(parameters.get("task_level", "node")),
        class_vocabulary=tuple(parameters.get("class_vocabulary", ())),
        units=parameters.get("units"),
        metadata_fields=requested_metadata,
        source_metadata=(
            {"source": source_name} if requested_metadata else {}
        ),
    )
    return source_name, adapter


def _phase(value: object) -> Phase:
    if not isinstance(value, str) or value not in _PHASES:
        raise ValueError(f"phase must be one of {_PHASES!r}")
    return value


def _identity_tensor(value: object, *, field_name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(
            f"prediction identity field {field_name} must be a tensor"
        )
    if value.ndim != 1:
        raise ValueError(
            f"prediction identity field {field_name} must be rank one"
        )
    if value.dtype == torch.bool or value.is_floating_point():
        raise TypeError(
            f"prediction identity field {field_name} must contain integer ordinals"
        )
    return value


@dataclass(frozen=True, slots=True)
class PredictionRowAdapter:
    """Convert the one supervised selection into canonical prediction rows.

    Disk-backed instances also retain the qualified store resolver needed to
    restore external identifiers at export. Native and disk paths otherwise
    share this one adapter contract.
    """

    source_graph_id: str
    output_kind: str
    target_node_type: str = ""
    sampling_strategy: str = ""
    task: str = "classification"
    task_level: str = "node"
    class_vocabulary: tuple[str, ...] = ()
    units: str | None = None
    metadata_fields: tuple[str, ...] = ()
    source_metadata: Mapping[str, int | str] = field(default_factory=dict)
    store_path: Path | None = None
    store_state: TypedGraphStoreState | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_graph_id, str)
            or not self.source_graph_id
        ):
            raise ValueError("source_graph_id must be a non-empty string")
        if self.output_kind not in {
            "graph",
            "homogeneous",
            "heterogeneous",
            "hypergraph",
        }:
            raise ValueError(
                "output_kind must be graph, homogeneous, heterogeneous, or "
                "hypergraph"
            )
        if self.task not in {"classification", "regression"}:
            raise ValueError("task must be classification or regression")
        if self.task_level not in {"graph", "node", "node_inductive"}:
            raise ValueError(
                "task_level must be graph, node, or node_inductive"
            )
        if self.output_kind == "heterogeneous":
            if (
                not isinstance(self.target_node_type, str)
                or not self.target_node_type
            ):
                raise ValueError(
                    "heterogeneous prediction rows require target_node_type"
                )
        elif self.target_node_type and not isinstance(
            self.target_node_type, str
        ):
            raise TypeError("target_node_type must be a string")
        if not isinstance(self.sampling_strategy, str):
            raise TypeError("sampling_strategy must be a string")
        if self.store_path is None and self.store_state is not None:
            raise ValueError("store_state requires store_path")
        if self.store_path is not None:
            object.__setattr__(self, "store_path", Path(self.store_path))
            if not isinstance(self.store_state, TypedGraphStoreState):
                raise TypeError(
                    "disk prediction rows require TypedGraphStoreState"
                )
            if Path(self.store_state.root) != self.store_path:
                raise ValueError(
                    "prediction identity store state/path mismatch"
                )
            expected_prefix = {
                "homogeneous": "homogeneous-",
                "heterogeneous": "heterogeneous-",
            }.get(self.output_kind)
            if (
                expected_prefix is None
                or not self.sampling_strategy.startswith(expected_prefix)
            ):
                raise ValueError(
                    "prediction identity output/strategy mismatch: "
                    f"output_kind={self.output_kind!r}, "
                    f"strategy={self.sampling_strategy!r}"
                )
        if not isinstance(self.class_vocabulary, tuple) or any(
            not isinstance(label, str) or not label
            for label in self.class_vocabulary
        ):
            raise TypeError(
                "class_vocabulary must be a tuple of non-empty strings"
            )
        if self.units is not None and (
            not isinstance(self.units, str) or not self.units
        ):
            raise ValueError("units must be a non-empty string or None")
        if (
            not isinstance(self.metadata_fields, tuple)
            or len(set(self.metadata_fields)) != len(self.metadata_fields)
            or any(
                not isinstance(name, str) or not name
                for name in self.metadata_fields
            )
        ):
            raise ValueError(
                "metadata_fields must contain unique non-empty names"
            )
        if not isinstance(self.source_metadata, Mapping):
            raise TypeError("source_metadata must be a mapping")
        unknown_metadata = set(self.source_metadata).difference(
            self.metadata_fields
        )
        missing_metadata = set(self.metadata_fields).difference(
            self.source_metadata
        )
        if unknown_metadata or missing_metadata:
            raise ValueError(
                "source_metadata must exactly match allowlisted metadata_fields"
            )
        normalized_metadata: dict[str, int | str] = {}
        for name, value in self.source_metadata.items():
            if (
                not isinstance(name, str)
                or not name
                or isinstance(value, bool)
                or not isinstance(value, (Integral, str))
                or (isinstance(value, str) and not value)
            ):
                raise TypeError(
                    "source metadata values must be non-empty strings or integers"
                )
            normalized_metadata[name] = (
                int(value) if isinstance(value, Integral) else value
            )
        object.__setattr__(
            self,
            "source_metadata",
            MappingProxyType(normalized_metadata),
        )

    @staticmethod
    def _row_indices(supervised: object) -> Tensor:
        row_indices = getattr(supervised, "row_indices", None)
        count = getattr(supervised, "num_examples", None)
        if (
            not isinstance(row_indices, Tensor)
            or row_indices.dtype != torch.long
            or row_indices.ndim != 1
            or type(count) is not int
            or row_indices.numel() != count
        ):
            raise TypeError(
                "supervised rows require aligned rank-one torch.long row_indices"
            )
        return row_indices

    @staticmethod
    def _take_rows(
        value: object,
        row_indices: Tensor,
        *,
        count: int,
        field_name: str,
        require_exact_count: bool = False,
    ) -> Tensor | np.ndarray:
        if isinstance(value, Tensor):
            value = _identity_tensor(value, field_name=field_name)
            if require_exact_count and value.numel() != count:
                raise ValueError(
                    f"prediction identity field {field_name} is not aligned "
                    "to supervised rows"
                )
            if value.numel() == count:
                return value
            indices = row_indices.to(device=value.device)
            if indices.numel() and int(indices.max().item()) >= value.numel():
                raise ValueError(
                    f"prediction identity field {field_name} is not aligned "
                    "to supervised rows"
                )
            return value.index_select(0, indices)
        if isinstance(value, np.ndarray):
            if value.ndim != 1 or value.dtype.hasobject:
                raise ValueError(
                    f"prediction identity field {field_name} must be a "
                    "non-object rank-one array"
                )
            if require_exact_count and len(value) != count:
                raise ValueError(
                    f"prediction identity field {field_name} is not aligned "
                    "to supervised rows"
                )
            if len(value) == count:
                return value
            indices = row_indices.detach().cpu().numpy()
            if indices.size and int(indices.max()) >= len(value):
                raise ValueError(
                    f"prediction identity field {field_name} is not aligned "
                    "to supervised rows"
                )
            return value[indices]
        if isinstance(value, (tuple, list)):
            array = np.asarray(value)
            return PredictionRowAdapter._take_rows(
                array,
                row_indices,
                count=count,
                field_name=field_name,
                require_exact_count=require_exact_count,
            )
        raise TypeError(
            f"prediction identity field {field_name} must be a tensor or array"
        )

    @staticmethod
    def _external_id_column(
        values: Sequence[int | str],
        *,
        count: int,
    ) -> np.ndarray:
        if len(values) != count:
            raise ValueError(
                "restored external_id values must be aligned to canonical rows"
            )
        if len(values) == 0:
            column = np.empty(0, dtype=np.int64)
        elif all(
            isinstance(value, Integral) and not isinstance(value, bool)
            for value in values
        ) or all(isinstance(value, str) and bool(value) for value in values):
            column = np.asarray(values)
        else:
            raise TypeError(
                "restored external_id values must contain only integers or "
                "non-empty strings"
            )
        if (
            column.ndim != 1
            or len(column) != count
            or column.dtype.kind not in {"i", "u", "U", "S"}
        ):
            raise TypeError(
                "restored external_id values must form a typed rank-one array"
            )
        return column

    def _identity(
        self,
        batch: Data | HeteroData,
        row_indices: Tensor,
        count: int,
        split_ordinal_start: int,
    ) -> PredictionIdentity:
        from topobench.evaluator.prediction import PredictionIdentity

        split_ordinals = np.arange(
            split_ordinal_start,
            split_ordinal_start + count,
            dtype=np.int64,
        )

        if self.output_kind == "graph":
            if not isinstance(batch, Data):
                raise TypeError("graph prediction rows require native Data")
            sample_id = self._take_rows(
                getattr(batch, "sample_id", None),
                row_indices,
                count=count,
                field_name="sample_id",
                require_exact_count=True,
            )
            return PredictionIdentity(
                columns={
                    "split_ordinal": split_ordinals,
                    "sample_id": sample_id,
                },
                key=("sample_id",),
            )

        if self.output_kind in {"homogeneous", "hypergraph"}:
            if not isinstance(batch, Data):
                raise TypeError(
                    f"{self.output_kind} prediction rows require native Data"
                )
            global_nid = self._take_rows(
                getattr(batch, "global_nid", None),
                row_indices,
                count=count,
                field_name="global_nid",
            )
            identity = PredictionIdentity(
                columns={
                    "split_ordinal": split_ordinals,
                    "source_graph_id": np.full(count, self.source_graph_id),
                    "global_nid": global_nid,
                },
                key=("source_graph_id", "global_nid"),
            )
        else:
            if not isinstance(batch, HeteroData):
                raise TypeError(
                    "heterogeneous prediction rows require HeteroData"
                )
            if self.target_node_type not in batch.node_types:
                raise ValueError(
                    f"batch has no target node type {self.target_node_type!r}"
                )
            n_id = self._take_rows(
                batch[self.target_node_type].get("n_id"),
                row_indices,
                count=count,
                field_name=f"{self.target_node_type}.n_id",
            )
            identity = PredictionIdentity(
                columns={
                    "split_ordinal": split_ordinals,
                    "source_graph_id": np.full(count, self.source_graph_id),
                    "target_node_type": np.full(
                        count,
                        self.target_node_type,
                    ),
                    "n_id": n_id,
                },
                key=("source_graph_id", "target_node_type", "n_id"),
            )

        if self.store_path is None:
            return identity
        external_ids = self.restore_external_ids(identity.rows)
        external_id = self._external_id_column(external_ids, count=count)
        return PredictionIdentity(
            columns={**identity.columns, "external_id": external_id},
            key=identity.key,
        )

    def adapt(
        self,
        batch: Data | HeteroData,
        supervised: object,
        *,
        phase: Phase,
        split_ordinal_start: int = 0,
    ) -> PredictionPayload:
        """Build one payload from already-selected logits and targets."""
        _phase(phase)
        if isinstance(split_ordinal_start, bool) or not isinstance(
            split_ordinal_start, Integral
        ):
            raise TypeError("split_ordinal_start must be an integer")
        split_ordinal_start = int(split_ordinal_start)
        if split_ordinal_start < 0:
            raise ValueError("split_ordinal_start must be non-negative")
        from topobench.evaluator.prediction import PredictionPayload

        logits = getattr(supervised, "logits", None)
        targets = getattr(supervised, "targets", None)
        count = getattr(supervised, "num_examples", None)
        if (
            not isinstance(logits, Tensor)
            or not isinstance(targets, Tensor)
            or type(count) is not int
            or logits.ndim == 0
            or targets.ndim == 0
            or logits.shape[0] != count
            or targets.shape[0] != count
        ):
            raise TypeError(
                "supervised logits and targets must be aligned tensors"
            )
        row_indices = self._row_indices(supervised)
        identity = self._identity(
            batch,
            row_indices,
            count,
            split_ordinal_start,
        )
        prediction = (
            torch.softmax(logits, dim=-1)
            if self.task == "classification"
            else logits
        )
        vocabulary = self.class_vocabulary
        if self.task == "classification" and not vocabulary:
            if logits.ndim != 2:
                raise ValueError(
                    "classification logits must have shape [N, C]"
                )
            vocabulary = tuple(str(index) for index in range(logits.shape[1]))
        columns: dict[str, Tensor | np.ndarray] = {
            "target": targets,
            "raw_output": logits,
        }
        metadata: dict[str, Mapping[str, object]] = {
            "target": {"role": "target"},
            "raw_output": {"role": "raw_output"},
            "prediction": {"role": "prediction"},
        }
        for name in self.metadata_fields:
            columns[name] = np.full(count, self.source_metadata[name])
            metadata[name] = {
                "role": "metadata",
                "vocabulary": (self.source_metadata[name],),
            }
        return PredictionPayload(
            identity=identity,
            prediction=prediction,
            columns=columns,
            column_metadata=metadata,
            output_semantics={
                "task": self.task,
                "class_vocabulary": vocabulary,
                "units": self.units,
            },
        )

    def resolve(
        self,
        batch: Data | HeteroData,
        *,
        phase: Phase,
    ) -> tuple[CanonicalPredictionIdentity, ...]:
        """Compatibility API for existing qualified disk identity callers."""
        if self.store_path is None:
            raise RuntimeError(
                "resolve is only available for disk-backed adapters"
            )
        selected_phase = _phase(phase)
        descriptor = getattr(batch, "sampling_descriptor", None)
        descriptor_content = getattr(descriptor, "content_sha256", None)
        descriptor_phase = getattr(descriptor, "phase", None)
        descriptor_strategy = getattr(descriptor, "strategy", None)
        if (
            descriptor_content != self.source_graph_id
            or descriptor_phase != selected_phase
            or descriptor_strategy != self.sampling_strategy
        ):
            raise ValueError(
                "prediction batch descriptor mismatch: "
                f"content={descriptor_content!r}, phase={descriptor_phase!r}, "
                f"strategy={descriptor_strategy!r}"
            )

        if self.output_kind == "homogeneous":
            if type(batch) is not Data:
                raise TypeError(
                    "homogeneous prediction identity requires native Data"
                )
            ordinals = _identity_tensor(
                getattr(batch, "global_nid", None),
                field_name="global_nid",
            )
            mask = self._phase_mask(batch, selected_phase, len(ordinals))
            selected = ordinals[mask]
            return tuple(
                (self.source_graph_id, int(ordinal))
                for ordinal in selected.tolist()
            )

        if not isinstance(batch, HeteroData):
            raise TypeError(
                "heterogeneous prediction identity requires native HeteroData"
            )
        if self.target_node_type not in batch.node_types:
            raise ValueError(
                "prediction batch has no target node type "
                f"{self.target_node_type!r}"
            )
        target = batch[self.target_node_type]
        ordinals = _identity_tensor(
            target.get("n_id"),
            field_name=f"{self.target_node_type}.n_id",
        )
        if self.sampling_strategy == "heterogeneous-neighbor":
            raw_count = target.get("batch_size")
            if (
                isinstance(raw_count, bool)
                or not isinstance(raw_count, Integral)
                or int(raw_count) < 1
                or int(raw_count) > len(ordinals)
            ):
                raise ValueError(
                    "heterogeneous neighbor prediction identity requires a "
                    "positive exact target batch_size"
                )
            selected = ordinals[: int(raw_count)]
        else:
            mask = self._phase_mask(target, selected_phase, len(ordinals))
            selected = ordinals[mask]
        return tuple(
            (
                self.source_graph_id,
                self.target_node_type,
                int(ordinal),
            )
            for ordinal in selected.tolist()
        )

    @staticmethod
    def _phase_mask(store: Any, phase: Phase, count: int) -> Tensor:
        mask = store.get(f"{phase}_mask")
        if (
            not isinstance(mask, Tensor)
            or mask.dtype != torch.bool
            or mask.ndim != 1
            or len(mask) != count
        ):
            raise ValueError(
                f"prediction batch requires rank-one boolean {phase}_mask "
                "aligned to canonical identities"
            )
        return mask

    def restore_external_ids(
        self,
        identities: Sequence[Sequence[int | str]],
    ) -> tuple[int | str, ...]:
        """Restore external IDs only after canonical rows reach export."""
        if isinstance(identities, (str, bytes)) or not isinstance(
            identities,
            Sequence,
        ):
            raise TypeError("identities must be an ordered sequence")
        if self.output_kind == "graph":
            external_ids: list[int | str] = []
            for index, identity in enumerate(identities):
                if (
                    isinstance(identity, (str, bytes))
                    or not isinstance(identity, Sequence)
                    or len(identity) != 1
                ):
                    raise ValueError(
                        f"canonical identity at index {index} has the wrong shape"
                    )
                value = identity[0]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (Integral, str))
                    or (isinstance(value, str) and not value)
                ):
                    raise TypeError(
                        f"canonical identity at index {index} has an invalid "
                        "sample ID"
                    )
                external_ids.append(
                    int(value) if isinstance(value, Integral) else value
                )
            return tuple(external_ids)
        ordinals: list[int] = []
        expected_size = 2 if self.output_kind != "heterogeneous" else 3
        for index, identity in enumerate(identities):
            if (
                isinstance(identity, (str, bytes))
                or not isinstance(identity, Sequence)
                or len(identity) != expected_size
            ):
                raise ValueError(
                    f"canonical identity at index {index} has the wrong shape"
                )
            if identity[0] != self.source_graph_id:
                raise ValueError(
                    f"canonical identity at index {index} belongs to another source"
                )
            ordinal_value = identity[-1]
            if isinstance(ordinal_value, bool) or not isinstance(
                ordinal_value,
                Integral,
            ):
                raise TypeError(
                    f"canonical identity at index {index} has a non-integer ordinal"
                )
            if int(ordinal_value) < 0:
                raise ValueError(
                    f"canonical identity at index {index} has a negative ordinal"
                )
            if self.output_kind == "heterogeneous" and (
                identity[1] != self.target_node_type
            ):
                raise ValueError(
                    f"canonical identity at index {index} has the wrong target type"
                )
            ordinals.append(int(ordinal_value))

        if self.store_path is None:
            return tuple(ordinals)
        assert self.store_state is not None
        with TypedGraphStore.from_state(self.store_state) as store:
            if store.content_sha256 != self.source_graph_id:
                raise ValueError(
                    "prediction identity store content changed: "
                    f"expected {self.source_graph_id!r}, "
                    f"received {store.content_sha256!r}"
                )
            if store.output_kind != self.output_kind:
                raise ValueError(
                    "prediction identity store output kind changed"
                )
            return tuple(store.external_ids(self.target_node_type, ordinals))


@runtime_checkable
class NonCommittingBatchProvider(Protocol):
    """Data-module boundary for isolated representative phase batches."""

    def noncommitting_probe_batches(
        self,
        phases: Sequence[Phase],
    ) -> AbstractContextManager[Mapping[Phase, Data | HeteroData]]:
        """Return one native batch per phase without advancing durable state."""
        ...


@dataclass(frozen=True)
class DataPipelineOutput:
    """Validated objects and runtime metadata produced by a data pipeline.

    The frozen container prevents replacing its references. It does not freeze
    the Lightning data module itself, whose internal state remains mutable for
    framework lifecycle hooks. ``HeterogeneousDataSpec`` and
    ``RuntimeDataCapability`` are independently immutable.
    """

    datamodule: LightningDataModule
    preprocessing_time: float
    data_spec: HeterogeneousDataSpec | None = None
    capability_spec: RuntimeDataCapability | None = None
    source_graph_id: str | None = None
    active_split_tag: str | None = None
    prediction_row_adapter: PredictionRowAdapter | None = None
    fitted_transform: FittableTransform | None = None
    fitted_state_root: Path | None = None
    reproducibility_policy: ReproducibilitySpec | None = None
    profiling_policy: ProfilingSpec | None = None
    execution_monitor: object | None = None
    qualification_report: QualificationReport | None = None
    supervision_counts: Mapping[str, int] = field(default_factory=dict)
    provenance_input: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        """Validate and normalize values crossing the pipeline boundary."""
        if not isinstance(self.datamodule, LightningDataModule):
            raise TypeError("datamodule must be a LightningDataModule")
        if self.data_spec is not None and not isinstance(
            self.data_spec,
            HeterogeneousDataSpec,
        ):
            raise TypeError(
                "data_spec must be a HeterogeneousDataSpec or None"
            )
        if self.capability_spec is not None and not isinstance(
            self.capability_spec,
            RuntimeDataCapability,
        ):
            raise TypeError(
                "capability_spec must be a RuntimeDataCapability or None"
            )
        if isinstance(self.preprocessing_time, bool) or not isinstance(
            self.preprocessing_time,
            Real,
        ):
            raise TypeError("preprocessing_time must be a real numeric scalar")

        preprocessing_time = float(self.preprocessing_time)
        if not math.isfinite(preprocessing_time):
            raise ValueError("preprocessing_time must be finite")
        if preprocessing_time < 0:
            raise ValueError("preprocessing_time must be non-negative")
        object.__setattr__(self, "preprocessing_time", preprocessing_time)

        optional_strings = {
            "source_graph_id": self.source_graph_id,
            "active_split_tag": self.active_split_tag,
        }
        for name, value in optional_strings.items():
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a non-empty string or None")
        if self.prediction_row_adapter is not None and not isinstance(
            self.prediction_row_adapter,
            PredictionRowAdapter,
        ):
            raise TypeError(
                "prediction_row_adapter must be a PredictionRowAdapter or None"
            )
        if (
            self.prediction_row_adapter is not None
            and self.source_graph_id
            != self.prediction_row_adapter.source_graph_id
        ):
            raise ValueError(
                "source_graph_id must match the prediction row adapter"
            )
        if self.fitted_transform is not None and not isinstance(
            self.fitted_transform,
            FittableTransform,
        ):
            raise TypeError(
                "fitted_transform must implement FittableTransform"
            )
        if self.fitted_state_root is not None:
            object.__setattr__(
                self,
                "fitted_state_root",
                Path(self.fitted_state_root),
            )
        if self.reproducibility_policy is not None and not isinstance(
            self.reproducibility_policy,
            ReproducibilitySpec,
        ):
            raise TypeError(
                "reproducibility_policy must be a ReproducibilitySpec or None"
            )
        if self.profiling_policy is not None and not isinstance(
            self.profiling_policy,
            ProfilingSpec,
        ):
            raise TypeError("profiling_policy must be a ProfilingSpec or None")
        if self.execution_monitor is not None and (
            not callable(getattr(self.execution_monitor, "begin", None))
            or not callable(getattr(self.execution_monitor, "finish", None))
        ):
            raise TypeError(
                "execution_monitor must expose callable begin and finish methods"
            )
        if self.qualification_report is not None and not isinstance(
            self.qualification_report,
            QualificationReport,
        ):
            raise TypeError(
                "qualification_report must be a QualificationReport or None"
            )
        counts: dict[str, int] = {}
        for phase, count in self.supervision_counts.items():
            selected_phase = _phase(phase)
            if isinstance(count, bool) or not isinstance(count, Integral):
                raise TypeError("supervision counts must be integers")
            normalized = int(count)
            if normalized < 1:
                raise ValueError("supervision counts must be positive")
            counts[selected_phase] = normalized
        object.__setattr__(
            self,
            "supervision_counts",
            MappingProxyType(counts),
        )
        if self.provenance_input is not None:
            if not isinstance(self.provenance_input, Mapping):
                raise TypeError("provenance_input must be a mapping or None")
            object.__setattr__(
                self,
                "provenance_input",
                MappingProxyType(dict(self.provenance_input)),
            )


class AbstractDataPipeline(ABC):
    """Build a data module and its runtime data contract."""

    def __init__(
        self,
        *,
        parquet_store_root: str | Path | None = None,
        parquet_store_path: str | Path | None = None,
        active_split_tag: str | None = None,
        qualified_profile: bool = True,
        fitted_transform: FittableTransform | None = None,
        fitted_state_root: str | Path | None = None,
        supervised_fit: bool = False,
        execution_monitor: object | None = None,
    ) -> None:
        if not isinstance(qualified_profile, bool):
            raise TypeError("qualified_profile must be bool")
        if not isinstance(supervised_fit, bool):
            raise TypeError("supervised_fit must be bool")
        if active_split_tag is not None and (
            not isinstance(active_split_tag, str) or not active_split_tag
        ):
            raise ValueError(
                "active_split_tag must be a non-empty string or None"
            )
        if fitted_transform is not None and not isinstance(
            fitted_transform,
            FittableTransform,
        ):
            raise TypeError(
                "fitted_transform must implement FittableTransform"
            )
        if execution_monitor is not None and (
            not callable(getattr(execution_monitor, "begin", None))
            or not callable(getattr(execution_monitor, "finish", None))
        ):
            raise TypeError(
                "execution_monitor must expose callable begin and finish methods"
            )
        self.parquet_store_root = (
            None if parquet_store_root is None else Path(parquet_store_root)
        )
        self.parquet_store_path = (
            None if parquet_store_path is None else Path(parquet_store_path)
        )
        self.active_split_tag = active_split_tag
        self.qualified_profile = qualified_profile
        self.fitted_transform = fitted_transform
        self.fitted_state_root = (
            None if fitted_state_root is None else Path(fitted_state_root)
        )
        self.supervised_fit = supervised_fit
        self.execution_monitor = execution_monitor

    def set_execution_monitor(self, execution_monitor: object | None) -> None:
        """Attach the one callback-owned monitor before source conversion."""
        if execution_monitor is not None and (
            not callable(getattr(execution_monitor, "begin", None))
            or not callable(getattr(execution_monitor, "finish", None))
        ):
            raise TypeError(
                "execution_monitor must expose callable begin and finish methods"
            )
        self.execution_monitor = execution_monitor

    @staticmethod
    def preprocess(cfg: DictConfig) -> PreProcessor:
        """Load and preprocess a dataset using the configured transforms."""
        loader = hydra.utils.instantiate(cfg.dataset.loader)
        dataset, dataset_dir = loader.load()
        transforms = (
            hydra.utils.instantiate(cfg.transforms)
            if cfg.get("transforms") is not None
            else None
        )
        return PreProcessor(dataset, dataset_dir, transforms)

    def build_parquet(
        self,
        cfg: DictConfig,
        *,
        expected_output_kind: str,
    ) -> DataPipelineOutput:
        """Build/open one exact descriptor and the shared disk data module."""
        started = time.perf_counter()
        expected_domain = {
            "homogeneous": "graph",
            "heterogeneous": "heterogeneous",
        }.get(expected_output_kind)
        if expected_domain is None:
            raise ValueError(
                "expected_output_kind must be homogeneous or heterogeneous"
            )
        qualification = qualify_dataset(cfg.dataset)
        configured_domain = OmegaConf.select(
            cfg,
            "dataset.loader.parameters.data_domain",
            default=None,
        )
        configured_output = OmegaConf.select(
            cfg,
            "dataset.loader.parameters.output_kind",
            default=None,
        )
        strategy_name = OmegaConf.select(
            cfg,
            "dataset.loader.parameters.partition.strategy",
            default=None,
        )
        backend_name = OmegaConf.select(
            cfg,
            "dataset.loader.parameters.partition.backend",
            default=None,
        )
        partition_count = OmegaConf.select(
            cfg,
            "dataset.loader.parameters.partition.num_partitions",
            default=None,
        )
        if configured_output != expected_output_kind:
            raise ValueError(
                "Parquet pipeline output_kind mismatch: "
                f"dataset.loader.parameters.output_kind={configured_output!r}, "
                f"expected {expected_output_kind!r}"
            )
        if configured_domain != expected_domain:
            raise ValueError(
                "Parquet pipeline data-domain mismatch: "
                f"dataset.loader.parameters.data_domain={configured_domain!r}, "
                f"expected {expected_domain!r}"
            )
        model_domain = OmegaConf.select(
            cfg,
            "model.model_domain",
            default=None,
        )
        if model_domain != expected_domain:
            raise ValueError(
                "Parquet pipeline/model mismatch: "
                f"cfg.model.model_domain={model_domain!r}, "
                f"expected {expected_domain!r}"
            )

        from topobench.data.capabilities import GRAPH_DATASET_MANIFEST

        capability = GRAPH_DATASET_MANIFEST["ParquetTypedGraph"]
        if not capability.supports_source(
            domain=expected_domain,
            output_kind=configured_output,
            strategy=strategy_name,
            backend=backend_name,
        ):
            raise ValueError(
                "Parquet source capability mismatch: "
                f"dataset.loader.parameters.output_kind={configured_output!r}, "
                "dataset.loader.parameters.partition.strategy="
                f"{strategy_name!r}, "
                "dataset.loader.parameters.partition.backend="
                f"{backend_name!r}"
            )
        if backend_name == "pyg" and (
            isinstance(partition_count, bool)
            or not isinstance(partition_count, Integral)
            or int(partition_count) < 2
        ):
            raise ValueError(
                "dataset.loader.parameters.partition.num_partitions must be "
                "at least 2 for backend='pyg'"
            )
        save_bundle = OmegaConf.select(
            cfg,
            (
                "dataset.loader.parameters.reproducibility."
                "save_reproducibility_bundle"
            ),
            default=None,
        )
        if self.qualified_profile and save_bundle is not True:
            raise ValueError(
                "qualified Parquet profile requires "
                "save_reproducibility_bundle=true"
            )
        if self.parquet_store_root is None:
            raise ValueError(
                "data_pipeline.parquet_store_root is required for Parquet sources"
            )

        loader = hydra.utils.instantiate(cfg.dataset.loader)
        if type(loader) is not ParquetTypedGraphLoader:
            raise TypeError(
                "Parquet pipeline requires the exact ParquetTypedGraphLoader; "
                f"received {type(loader).__module__}.{type(loader).__qualname__}"
            )
        source = loader.source
        if type(source) is not ParquetTypedGraphSource:
            raise TypeError(
                "Parquet pipeline requires the exact immutable "
                "ParquetTypedGraphSource descriptor"
            )
        spec = source.spec
        registry = spec.supervision.split_registry
        active_split_tag = (
            registry.active_tag
            if self.active_split_tag is None
            else self.active_split_tag
        )
        registered_splits = {split.tag: split for split in registry.sets}
        if active_split_tag != registry.active_tag:
            raise ValueError(
                "active split tag must remain "
                f"{registry.active_tag!r}; received {active_split_tag!r}"
            )
        if not registered_splits[active_split_tag].qualified:
            raise ValueError(
                f"active split tag {active_split_tag!r} is unqualified"
            )

        fitted_transform, fitted_state_root = self._fitted_runtime(source)
        from topobench.data.stores.typed_graph_ingestion import (
            ParquetTypedGraphIngestor,
        )
        from topobench.data.stores.typed_graph_store import (
            TypedGraphStoreWriter,
        )

        ingestor = ParquetTypedGraphIngestor(
            source,
            self.parquet_store_root,
            execution_monitor=self.execution_monitor,
        )
        if self.parquet_store_path is None:
            partition_build = ingestor.build_partitions()
            store_build = TypedGraphStoreWriter(
                ingestor,
                partition_build,
                execution_monitor=self.execution_monitor,
            ).build()
            store_path = store_build.path
            store = store_build.store
            inventory = partition_build.inventory
            qualification_report: QualificationReport | None = (
                store_build.qualification_report
            )
        else:
            inventory = ingestor.inventory()
            store_path = self.parquet_store_path
            store = TypedGraphStore.open(
                store_path,
                execution_monitor=self.execution_monitor,
            )
            qualification_report = store.qualification_report

        with store:
            self._validate_store_binding(store, source, inventory)
            strategy = self._sampling_strategy(cfg, store)
            from topobench.dataloader.disk_graph import DiskGraphDataModule

            dataloader_cfg = cfg.dataset.dataloader_params
            store_state = store.state()
            datamodule = DiskGraphDataModule(
                TypedGraphStore.from_state(store_state),
                strategy,
                active_split_tag=active_split_tag,
                num_workers=dataloader_cfg.get("num_workers", 0),
                persistent_workers=dataloader_cfg.get(
                    "persistent_workers",
                    False,
                ),
                train_shuffle=dataloader_cfg.get("train_shuffle", True),
                fitted_transform=fitted_transform,
                fitted_state_root=fitted_state_root,
                supervised_fit=self.supervised_fit,
                execution_monitor=self.execution_monitor,
            )
            counts = {
                phase: int(
                    store._manifest["splits"][active_split_tag]["phases"][
                        phase
                    ]["shape"][0]
                )
                for phase in _PHASES
            }
            observed_num_classes = (
                None
                if qualification.task != "classification"
                else _observed_classification_count(
                    (store.node_labels(store._manifest["target_node_type"]),)
                )
            )
            store_feature_widths = tuple(
                (
                    node_type,
                    int(store._node(node_type)["feature_width"]),
                )
                for node_type in store.node_types
            )
            if (
                fitted_transform is not None
                and fitted_transform.spec.target_field == "x"
            ):
                transformed_node_type = fitted_transform.spec.target_node_type
                if transformed_node_type is None:
                    if store.output_kind != "homogeneous":
                        raise ValueError(
                            "heterogeneous fitted transforms require an explicit "
                            "target node type"
                        )
                    transformed_node_type = store.node_types[0]
                if transformed_node_type not in store.node_types:
                    raise ValueError(
                        "fitted transform target node type is absent from the "
                        "typed graph store"
                    )
                transformed_width = int(fitted_transform.n_components)
                store_feature_widths = tuple(
                    (
                        node_type,
                        transformed_width
                        if node_type == transformed_node_type
                        else width,
                    )
                    for node_type, width in store_feature_widths
                )
            feature_widths = (
                (("node", store_feature_widths[0][1]),)
                if store.output_kind == "homogeneous"
                else store_feature_widths
            )
            observed_domain = {
                "homogeneous": "graph",
                "heterogeneous": "heterogeneous",
            }[store.output_kind]
            capability_spec = RuntimeDataCapability(
                selector=qualification.selector,
                data_domain=observed_domain,
                output_kind=store.output_kind,
                feature_widths=feature_widths,
                num_classes=observed_num_classes,
                target_node_type=store._manifest["target_node_type"],
            )
            if expected_output_kind == "homogeneous":
                cfg.model.feature_encoder.in_channels = feature_widths[0][1]
                data_spec = None
            else:
                if observed_num_classes is None:
                    raise ValueError(
                        "heterogeneous Parquet supervision must be classification"
                    )
                data_spec = HeterogeneousDataSpec(
                    node_types=store.node_types,
                    edge_types=store.relation_types,
                    target_node_type=store._manifest["target_node_type"],
                    num_classes=observed_num_classes,
                    input_channels=store_feature_widths,
                )
            requested_metadata = tuple(
                OmegaConf.select(
                    cfg,
                    "evaluation_artifacts.metadata_fields",
                    default=(),
                )
                or ()
            )
            if set(requested_metadata).difference({"source"}):
                raise ValueError(
                    "disk prediction metadata is restricted to the source "
                    "allowlist"
                )
            source_name = OmegaConf.select(
                cfg,
                "dataset.loader.parameters.data_name",
                default=store.content_sha256,
            )
            if not isinstance(source_name, str) or not source_name:
                raise ValueError(
                    "disk prediction metadata requires a declared source name"
                )
            adapter = PredictionRowAdapter(
                store_path=store.path,
                store_state=store_state,
                source_graph_id=store.content_sha256,
                output_kind=store.output_kind,
                target_node_type=store._manifest["target_node_type"],
                sampling_strategy=strategy.name,
                task=str(cfg.dataset.parameters.get("task", "classification")),
                task_level=str(
                    cfg.dataset.parameters.get("task_level", "node")
                ),
                class_vocabulary=tuple(
                    cfg.dataset.parameters.get("class_vocabulary", ())
                ),
                units=cfg.dataset.parameters.get("units"),
                metadata_fields=requested_metadata,
                source_metadata=(
                    {"source": source_name} if requested_metadata else {}
                ),
            )
            if qualification_report is None:
                qualification_report = store.qualification_report
            provenance = self._provenance_input(
                store,
                source,
                active_split_tag=active_split_tag,
                sampling_strategy=strategy.name,
                counts=counts,
                fitted_state_root=fitted_state_root,
                fitted_transform=fitted_transform,
                qualification_report=qualification_report,
            )
            source_graph_id = store.content_sha256

        return DataPipelineOutput(
            datamodule=datamodule,
            preprocessing_time=time.perf_counter() - started,
            data_spec=data_spec,
            capability_spec=capability_spec,
            source_graph_id=source_graph_id,
            active_split_tag=active_split_tag,
            prediction_row_adapter=adapter,
            fitted_transform=fitted_transform,
            fitted_state_root=fitted_state_root,
            reproducibility_policy=spec.reproducibility,
            profiling_policy=spec.profiling,
            execution_monitor=self.execution_monitor,
            qualification_report=qualification_report,
            supervision_counts=counts,
            provenance_input=provenance,
        )

    def _fitted_runtime(
        self,
        source: ParquetTypedGraphSource,
    ) -> tuple[FittableTransform | None, Path | None]:
        declaration = source.spec.fitted_transform
        if declaration.name == "identity":
            if self.fitted_transform is not None:
                raise ValueError(
                    "fitted transform mismatch: descriptor declares identity "
                    "but data_pipeline.fitted_transform is executable"
                )
            return None, None

        from topobench.transforms.incremental_pca import (
            IncrementalPCATransform,
        )

        if not isinstance(self.fitted_transform, IncrementalPCATransform):
            raise ValueError(
                "fitted transform mismatch: descriptor declares pca and requires "
                "IncrementalPCATransform"
            )
        declared_root = (
            None
            if declaration.state_path is None
            else source.spec.source_root / declaration.state_path
        )
        if (
            declared_root is not None
            and self.fitted_state_root is not None
            and declared_root.resolve(strict=False)
            != self.fitted_state_root.resolve(strict=False)
        ):
            raise ValueError(
                "fitted state path mismatch between descriptor and data pipeline"
            )
        state_root = declared_root or self.fitted_state_root
        if state_root is None:
            raise ValueError(
                "pca fitted transform requires data_pipeline.fitted_state_root"
            )
        return self.fitted_transform, state_root

    @staticmethod
    def _validate_store_binding(
        store: TypedGraphStore,
        source: ParquetTypedGraphSource,
        inventory: Any,
    ) -> None:
        binding = store._manifest["source_binding"]
        expected = (
            inventory.source_fingerprint,
            inventory.config_fingerprint,
        )
        observed = (
            binding.get("source_fingerprint"),
            binding.get("config_fingerprint"),
        )
        if observed != expected:
            raise ValueError(
                "Parquet store/source binding mismatch: "
                f"store={observed!r}, descriptor={expected!r}"
            )
        if store.output_kind != source.spec.output_kind:
            raise ValueError(
                "Parquet store/output mismatch: "
                f"store={store.output_kind!r}, "
                f"descriptor={source.spec.output_kind!r}"
            )
        partition = store._manifest["partition"]
        if (
            partition.get("backend") != source.spec.partition.backend
            or partition.get("num_partitions")
            != source.spec.partition.num_partitions
        ):
            raise ValueError(
                "Parquet store/partition mismatch: "
                f"store backend/count={(partition.get('backend'), partition.get('num_partitions'))!r}, "
                "descriptor backend/count="
                f"{(source.spec.partition.backend, source.spec.partition.num_partitions)!r}"
            )
        if (
            store.active_split_tag
            != source.spec.supervision.split_registry.active_tag
        ):
            raise ValueError(
                "Parquet store/active split mismatch: "
                f"store={store.active_split_tag!r}, descriptor="
                f"{source.spec.supervision.split_registry.active_tag!r}"
            )
        if not store.qualification_report.passed:
            raise ValueError(
                "Parquet store partition is unqualified: "
                f"report={store.qualification_report.report_path}"
            )

    @staticmethod
    def _sampling_strategy(cfg: DictConfig, store: TypedGraphStore) -> Any:
        from topobench.dataloader.disk_graph import (
            HeterogeneousClusterStrategy,
            HeterogeneousNeighborStrategy,
            HomogeneousClusterStrategy,
        )

        declaration = cfg.dataset.loader.parameters.partition
        dataloader = cfg.dataset.dataloader_params
        seed = int(cfg.seed)
        if store.output_kind == "homogeneous":
            if declaration.strategy != "cluster":
                raise ValueError(
                    "homogeneous Parquet output requires "
                    "partition.strategy='cluster'"
                )
            return HomogeneousClusterStrategy(
                clusters_per_batch=dataloader.get("clusters_per_batch", 1),
                partition_groups=dataloader.get("partition_groups"),
                seed=seed,
            )

        mode = dataloader.get("mode")
        if declaration.strategy == "cluster":
            if mode != "full_batch":
                raise ValueError(
                    "partition.strategy='cluster' requires "
                    "dataset.dataloader_params.mode='full_batch'"
                )
            return HeterogeneousClusterStrategy(
                clusters_per_batch=dataloader.get("clusters_per_batch", 1),
                partition_groups=dataloader.get("partition_groups"),
                seed=seed,
            )
        if declaration.strategy == "neighbor":
            if mode != "neighbor":
                raise ValueError(
                    "partition.strategy='neighbor' requires "
                    "dataset.dataloader_params.mode='neighbor'"
                )
            raw_fanout = dataloader.get("num_neighbors")
            fanout = OmegaConf.to_container(raw_fanout, resolve=True)
            return HeterogeneousNeighborStrategy(
                batch_size=dataloader.get("batch_size"),
                num_neighbors=fanout,
                seed=seed,
                replace=dataloader.get("replace", False),
                subgraph_type=dataloader.get("subgraph_type", "directional"),
                sample_direction=dataloader.get("sample_direction", "forward"),
                filter_per_worker=dataloader.get("filter_per_worker", False),
            )
        raise ValueError(
            "unsupported heterogeneous partition.strategy="
            f"{declaration.strategy!r}"
        )

    def _provenance_input(
        self,
        store: TypedGraphStore,
        source: ParquetTypedGraphSource,
        *,
        active_split_tag: str,
        sampling_strategy: str,
        counts: Mapping[str, int],
        fitted_state_root: Path | None,
        fitted_transform: FittableTransform | None,
        qualification_report: QualificationReport,
    ) -> Mapping[str, object]:
        split = store._manifest["splits"][active_split_tag]
        return MappingProxyType(
            {
                "source_graph_id": store.content_sha256,
                "partition_book_identity": store.partition_book_identity,
                "active_split_tag": active_split_tag,
                "split_fingerprint": split["fingerprint"],
                "sampling_strategy": sampling_strategy,
                "sampler_backend": source.spec.partition.backend,
                "supervision_counts": dict(counts),
                "fitted_transform": source.spec.fitted_transform.name,
                "fitted_transform_state_key": (
                    None
                    if fitted_transform is None
                    else fitted_transform.state_key
                ),
                "fitted_state_root": (
                    None
                    if fitted_state_root is None
                    else str(fitted_state_root)
                ),
                "save_reproducibility_bundle": (
                    source.spec.reproducibility.save_reproducibility_bundle
                ),
                "profiling_enabled": source.spec.profiling.enabled,
                "qualified_profile": self.qualified_profile,
                "qualification_report": str(qualification_report.report_path),
            }
        )

    @abstractmethod
    def build(self, cfg: DictConfig) -> DataPipelineOutput:
        """Build a Lightning data module and its runtime data contract."""
