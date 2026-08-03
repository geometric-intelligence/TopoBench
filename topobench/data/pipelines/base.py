"""Shared interfaces for configuration-driven data pipelines."""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

import hydra
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch_geometric.data import Data, HeteroData

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

Phase: TypeAlias = Literal["train", "val", "test"]
CanonicalPredictionIdentity: TypeAlias = (
    tuple[str, int] | tuple[str, str, int]
)
_PHASES: tuple[Phase, ...] = ("train", "val", "test")
_PARQUET_LOADER_TARGET = (
    "topobench.data.loaders.parquet.ParquetTypedGraphLoader"
)


def is_parquet_typed_graph_config(cfg: DictConfig) -> bool:
    """Select only the exact packaged immutable Parquet descriptor."""
    return (
        OmegaConf.select(cfg, "dataset.loader._target_", default=None)
        == _PARQUET_LOADER_TARGET
    )


def _phase(value: object) -> Phase:
    if not isinstance(value, str) or value not in _PHASES:
        raise ValueError(f"phase must be one of {_PHASES!r}")
    return value


def _identity_tensor(value: object, *, field_name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"prediction identity field {field_name} must be a tensor")
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
class PredictionIdentityResolver:
    """Resolve batch-local predictions to stable store and external identities.

    Canonical identities remain integer-only and are safe to carry beside model
    outputs. External identifiers are restored explicitly and lazily through the
    qualified store, which is the sole PyArrow boundary.
    """

    store_path: Path
    store_state: TypedGraphStoreState
    source_graph_id: str
    output_kind: str
    target_node_type: str
    sampling_strategy: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "store_path", Path(self.store_path))
        if not isinstance(self.store_state, TypedGraphStoreState):
            raise TypeError("store_state must be a TypedGraphStoreState")
        if Path(self.store_state.root) != self.store_path:
            raise ValueError("prediction identity store state/path mismatch")
        if self.output_kind not in {"homogeneous", "heterogeneous"}:
            raise ValueError("output_kind must be homogeneous or heterogeneous")
        for name in (
            "source_graph_id",
            "target_node_type",
            "sampling_strategy",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        expected_strategy_prefix = {
            "homogeneous": "homogeneous-",
            "heterogeneous": "heterogeneous-",
        }[self.output_kind]
        if not self.sampling_strategy.startswith(expected_strategy_prefix):
            raise ValueError(
                "prediction identity output/strategy mismatch: "
                f"output_kind={self.output_kind!r}, "
                f"strategy={self.sampling_strategy!r}"
            )

    def resolve(
        self,
        batch: Data | HeteroData,
        *,
        phase: Phase,
    ) -> tuple[CanonicalPredictionIdentity, ...]:
        """Return canonical identities aligned to the phase-owned predictions."""
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
        identities: Sequence[CanonicalPredictionIdentity],
    ) -> tuple[int | str, ...]:
        """Restore exact external IDs for already-selected canonical identities."""
        if isinstance(identities, (str, bytes)) or not isinstance(
            identities,
            Sequence,
        ):
            raise TypeError("identities must be an ordered sequence")
        ordinals: list[int] = []
        expected_size = 2 if self.output_kind == "homogeneous" else 3
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

        with TypedGraphStore.from_state(self.store_state) as store:
            if store.content_sha256 != self.source_graph_id:
                raise ValueError(
                    "prediction identity store content changed: "
                    f"expected {self.source_graph_id!r}, "
                    f"received {store.content_sha256!r}"
                )
            if store.output_kind != self.output_kind:
                raise ValueError("prediction identity store output kind changed")
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
    framework lifecycle hooks. ``HeterogeneousDataSpec`` is independently
    immutable.
    """

    datamodule: LightningDataModule
    preprocessing_time: float
    data_spec: HeterogeneousDataSpec | None = None
    source_graph_id: str | None = None
    active_split_tag: str | None = None
    prediction_identity_resolver: PredictionIdentityResolver | None = None
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
        if self.prediction_identity_resolver is not None and not isinstance(
            self.prediction_identity_resolver,
            PredictionIdentityResolver,
        ):
            raise TypeError(
                "prediction_identity_resolver must be a "
                "PredictionIdentityResolver or None"
            )
        if self.fitted_transform is not None and not isinstance(
            self.fitted_transform,
            FittableTransform,
        ):
            raise TypeError("fitted_transform must implement FittableTransform")
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
            raise ValueError("active_split_tag must be a non-empty string or None")
        if fitted_transform is not None and not isinstance(
            fitted_transform,
            FittableTransform,
        ):
            raise TypeError("fitted_transform must implement FittableTransform")
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
        if (
            backend_name == "pyg"
            and (
                isinstance(partition_count, bool)
                or not isinstance(partition_count, Integral)
                or int(partition_count) < 2
            )
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
        from topobench.data.stores.typed_graph_store import TypedGraphStoreWriter

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
            data_spec = (
                None
                if expected_output_kind == "homogeneous"
                else HeterogeneousDataSpec(
                    node_types=store.node_types,
                    edge_types=store.relation_types,
                    target_node_type=store._manifest["target_node_type"],
                    num_classes=int(cfg.dataset.parameters.num_classes),
                    input_channels=tuple(
                        (
                            node_type,
                            int(store._node(node_type)["feature_width"]),
                        )
                        for node_type in store.node_types
                    ),
                )
            )
            resolver = PredictionIdentityResolver(
                store_path=store.path,
                store_state=store_state,
                source_graph_id=store.content_sha256,
                output_kind=store.output_kind,
                target_node_type=store._manifest["target_node_type"],
                sampling_strategy=strategy.name,
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
            source_graph_id=source_graph_id,
            active_split_tag=active_split_tag,
            prediction_identity_resolver=resolver,
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

        from topobench.transforms.incremental_pca import IncrementalPCATransform

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
        if store.active_split_tag != source.spec.supervision.split_registry.active_tag:
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
                "qualification_report": str(
                    qualification_report.report_path
                ),
            }
        )

    @abstractmethod
    def build(self, cfg: DictConfig) -> DataPipelineOutput:
        """Build a Lightning data module and its runtime data contract."""
