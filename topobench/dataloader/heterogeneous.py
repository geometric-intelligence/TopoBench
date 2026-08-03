"""Separate PyG batching for native heterogeneous node classification."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from importlib import metadata
from numbers import Integral
from typing import Literal

import torch
import torch_geometric
from lightning import LightningDataModule
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.typing import EdgeType

from topobench.data.heterogeneous import HeterogeneousDataSpec
from topobench.dataloader.graph import loader_worker_options

Phase = Literal["train", "val", "test"]
SamplingMode = Literal["full_batch", "neighbor"]
EvaluationProtocol = Literal["full_graph", "sampled_neighbor_fixed"]
Fanout = list[int] | dict[EdgeType, list[int]]
_MAX_EVALUATION_SEED = 2**63 - 1


def _normalize_fanout_values(
    values: Sequence[object],
    *,
    field_name: str,
) -> list[int]:
    """Return a fresh, non-empty list of positive built-in integers."""
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{field_name} values must be positive integers")
        integer = int(value)
        if integer < 1:
            raise ValueError(f"{field_name} values must be positive")
        normalized.append(integer)
    return normalized


def _normalize_fanout(
    fanout: Sequence[int] | Mapping[EdgeType, Sequence[int]],
    edge_types: Sequence[EdgeType],
) -> Fanout:
    """Validate and deeply copy generic or relation-specific fanout."""
    expected_edge_types = tuple(edge_types)
    if isinstance(fanout, Mapping):
        received_edge_types = tuple(fanout)
        missing = tuple(
            edge_type
            for edge_type in expected_edge_types
            if edge_type not in fanout
        )
        extra = tuple(
            edge_type
            for edge_type in received_edge_types
            if edge_type not in expected_edge_types
        )
        if missing or extra:
            raise ValueError(
                "Relation fanout keys must exactly match edge_types; "
                f"missing={missing!r}, extra={extra!r}"
            )
        return {
            edge_type: _normalize_fanout_sequence(
                fanout[edge_type],
                field_name=f"Fanout for relation {edge_type!r}",
            )
            for edge_type in expected_edge_types
        }

    if isinstance(fanout, (str, bytes)) or not isinstance(fanout, Sequence):
        raise TypeError(
            "num_neighbors must be an ordered sequence or relation mapping"
        )
    return _normalize_fanout_values(
        fanout,
        field_name="num_neighbors",
    )


def _normalize_fanout_sequence(
    values: object,
    *,
    field_name: str,
) -> list[int]:
    """Validate a non-string ordered sequence before normalizing values."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be an ordered sequence")
    return _normalize_fanout_values(values, field_name=field_name)


def _normalize_integer(
    value: object,
    *,
    field_name: str,
    minimum: int,
) -> int:
    """Normalize a bounded non-boolean integral constructor option."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{field_name} must be {qualifier}")
    return normalized


def _require_bool(value: object, *, field_name: str) -> bool:
    """Require an exact boolean instead of accepting integer coercion."""
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be bool")
    return value


def _distribution_version(distribution: str) -> str | None:
    """Return an installed sampling-backend version, if available."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


class _FixedEvaluationNeighborLoader(NeighborLoader):
    """Replay deterministic evaluation sampling without retaining its batches.

    RNG isolation snapshots process-global CPU PyTorch state and assumes
    normal single-threaded Lightning traversal. Concurrent threaded iteration
    of validation or test loaders is unsupported.
    """

    def __init__(
        self,
        *args: object,
        evaluation_phase: Literal["val", "test"],
        evaluation_phase_seed: int,
        **kwargs: object,
    ) -> None:
        """Bind a phase-specific seed to a non-persistent PyG loader."""
        self._evaluation_phase = evaluation_phase
        self._evaluation_phase_seed = evaluation_phase_seed
        super().__init__(*args, **kwargs)

    def _base_iterator(self) -> Iterator[HeteroData]:
        """Create one ordinary PyG iterator for a streaming traversal."""
        return super().__iter__()

    @staticmethod
    def _release_base_iterator(iterator: Iterator[HeteroData]) -> None:
        """Release multiprocessing workers on exhaustion, failure, or close."""
        worker_iterator = getattr(iterator, "iterator", iterator)
        shutdown_workers = getattr(worker_iterator, "_shutdown_workers", None)
        if callable(shutdown_workers):
            shutdown_workers()
        close = getattr(iterator, "close", None)
        if callable(close):
            close()

    def _sampling_error(self) -> RuntimeError:
        """Describe the optional PyG backend required by neighbor sampling."""
        return RuntimeError(
            "Fixed heterogeneous sampling requires pyg-lib or torch-sparse; "
            f"cannot stream the {self._evaluation_phase} phase"
        )

    def _seeded_iterator(self) -> Iterator[HeteroData]:
        """Stream one deterministic traversal while preserving caller RNG."""
        source: Iterator[HeteroData] | None = None
        sampling_state: torch.Tensor | None = None
        yielded_batch = False
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self._evaluation_phase_seed)
                source = self._base_iterator()
                sampling_state = torch.random.get_rng_state().clone()

            while True:
                try:
                    with torch.random.fork_rng(devices=[]):
                        if sampling_state is None:
                            raise RuntimeError(
                                "Fixed evaluation RNG was not initialized"
                            )
                        torch.random.set_rng_state(sampling_state)
                        try:
                            batch = next(source)
                        finally:
                            sampling_state = (
                                torch.random.get_rng_state().clone()
                            )
                except StopIteration:
                    break
                yielded_batch = True
                yield batch

            if not yielded_batch:
                raise RuntimeError(
                    f"Fixed heterogeneous {self._evaluation_phase} sampling "
                    "produced no batches"
                )
        except ImportError as error:
            raise self._sampling_error() from error
        except RuntimeError as error:
            message = str(error).lower()
            if "pyg-lib" not in message and "torch-sparse" not in message:
                raise
            raise self._sampling_error() from error
        finally:
            if source is not None:
                self._release_base_iterator(source)

    def __iter__(self) -> Iterator[HeteroData]:
        """Return a fresh, deterministic, bounded-memory sampling stream."""
        return self._seeded_iterator()


class HeterogeneousNodeDataModule(LightningDataModule):
    """Batch one native heterogeneous graph without TopoBench collation.

    The datamodule retains the validated graph reference and treats its graph
    and loader options as canonical for its lifetime. Callers must not mutate
    the graph or public option attributes after construction. Full-graph
    validation and test hooks construct a fresh loader per call.
    Sampled validation and test loaders are memoized by phase, but every
    traversal deterministically resamples and streams fresh batches without
    retaining them. Sampled evaluation workers are non-persistent and released
    per traversal.
    """

    def __init__(
        self,
        data: HeteroData,
        spec: HeterogeneousDataSpec,
        *,
        mode: SamplingMode,
        batch_size: int = 128,
        num_neighbors: Sequence[int]
        | Mapping[EdgeType, Sequence[int]]
        | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        train_shuffle: bool = True,
        replace: bool = False,
        subgraph_type: str = "directional",
        filter_per_worker: bool = False,
        evaluation_protocol: EvaluationProtocol | None = None,
        evaluation_seed: int = 0,
    ) -> None:
        """Validate the graph contract and store immutable loader options."""
        super().__init__()
        if not isinstance(data, HeteroData):
            raise TypeError("data must be native HeteroData")
        if not isinstance(spec, HeterogeneousDataSpec):
            raise TypeError("spec must be a HeterogeneousDataSpec")
        if not isinstance(mode, str):
            raise TypeError("mode must be a string")
        if mode not in {"full_batch", "neighbor"}:
            raise ValueError(f"Unsupported heterogeneous loader mode: {mode}")
        expected_protocol = (
            "full_graph" if mode == "full_batch" else "sampled_neighbor_fixed"
        )
        if evaluation_protocol is None:
            normalized_evaluation_protocol = expected_protocol
        else:
            if not isinstance(evaluation_protocol, str):
                raise TypeError("evaluation_protocol must be a string")
            if evaluation_protocol != expected_protocol:
                raise ValueError(
                    f"evaluation_protocol {evaluation_protocol!r} does not "
                    f"match mode {mode!r}; expected {expected_protocol!r}"
                )
            normalized_evaluation_protocol = evaluation_protocol
        normalized_evaluation_seed = _normalize_integer(
            evaluation_seed,
            field_name="evaluation_seed",
            minimum=0,
        )
        if normalized_evaluation_seed > _MAX_EVALUATION_SEED:
            raise ValueError(
                "evaluation_seed must be no greater than "
                f"{_MAX_EVALUATION_SEED}"
            )

        normalized_batch_size = _normalize_integer(
            batch_size,
            field_name="batch_size",
            minimum=1,
        )
        normalized_num_workers = _normalize_integer(
            num_workers,
            field_name="num_workers",
            minimum=0,
        )
        normalized_pin_memory = _require_bool(
            pin_memory,
            field_name="pin_memory",
        )
        normalized_persistent_workers = _require_bool(
            persistent_workers,
            field_name="persistent_workers",
        )
        normalized_train_shuffle = _require_bool(
            train_shuffle,
            field_name="train_shuffle",
        )
        normalized_replace = _require_bool(replace, field_name="replace")
        normalized_filter_per_worker = _require_bool(
            filter_per_worker,
            field_name="filter_per_worker",
        )
        if normalized_persistent_workers and normalized_num_workers == 0:
            raise ValueError(
                "persistent_workers requires num_workers greater than zero"
            )
        if subgraph_type != "directional":
            raise ValueError(
                "Heterogeneous v1 supports directional sampling only"
            )

        data_metadata = (
            tuple(data.node_types),
            tuple(data.edge_types),
        )
        spec_metadata = (spec.node_types, spec.edge_types)
        if data_metadata != spec_metadata:
            raise ValueError(
                "Heterogeneous data metadata must match the validated spec; "
                f"data={data_metadata!r}, spec={spec_metadata!r}"
            )
        if (
            spec.target_node_type not in spec.node_types
            or spec.target_node_type not in data.node_types
        ):
            raise ValueError(
                f"Target node type {spec.target_node_type!r} must exist in "
                "both data and spec"
            )
        target_store = data[spec.target_node_type]
        missing_masks = tuple(
            f"{phase}_mask"
            for phase in ("train", "val", "test")
            if f"{phase}_mask" not in target_store
        )
        if missing_masks:
            raise ValueError(
                f"Target node type {spec.target_node_type!r} is missing "
                f"phase masks {missing_masks!r}"
            )

        fanout = [15, 10] if num_neighbors is None else num_neighbors
        self.data = data
        self.spec = spec
        self.mode = mode
        self.batch_size = normalized_batch_size
        self.num_neighbors = _normalize_fanout(fanout, spec.edge_types)
        self.num_workers = normalized_num_workers
        self.pin_memory = normalized_pin_memory
        self.persistent_workers = normalized_persistent_workers
        self.train_shuffle = normalized_train_shuffle
        self.replace = normalized_replace
        self.subgraph_type = subgraph_type
        self.filter_per_worker = normalized_filter_per_worker
        self.evaluation_protocol = normalized_evaluation_protocol
        self.evaluation_seed = normalized_evaluation_seed
        self._evaluation_loaders: dict[
            Literal["val", "test"],
            _FixedEvaluationNeighborLoader,
        ] = {}

        # Deliberately omit ``save_hyperparameters``: relation-keyed fanout is
        # not reliably YAML/JSON serializable, and graph/spec must never enter
        # Lightning checkpoint hyperparameters.

    def _common_kwargs(self) -> dict[str, object]:
        """Return worker options shared by full and sampled PyG loaders."""
        return loader_worker_options(
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def _full_loader(self) -> DataLoader:
        """Construct a fresh one-graph, non-shuffled PyG loader."""
        return DataLoader(
            [self.data],
            batch_size=1,
            shuffle=False,
            **self._common_kwargs(),
        )

    def _neighbor_loader(self, phase: Phase) -> NeighborLoader:
        """Construct training or retrieve a fixed evaluation neighbor loader."""
        mask = self.data[self.spec.target_node_type][f"{phase}_mask"]
        if phase == "train":
            return NeighborLoader(
                self.data,
                input_nodes=(self.spec.target_node_type, mask),
                num_neighbors=self.num_neighbors,
                batch_size=self.batch_size,
                shuffle=self.train_shuffle,
                replace=self.replace,
                subgraph_type=self.subgraph_type,
                filter_per_worker=self.filter_per_worker,
                **self._common_kwargs(),
            )
        evaluation_phase: Literal["val", "test"] = (
            "val" if phase == "val" else "test"
        )
        loader = self._evaluation_loaders.get(evaluation_phase)
        if loader is None:
            loader = _FixedEvaluationNeighborLoader(
                self.data,
                input_nodes=(self.spec.target_node_type, mask),
                num_neighbors=self.num_neighbors,
                batch_size=self.batch_size,
                shuffle=False,
                replace=self.replace,
                subgraph_type=self.subgraph_type,
                filter_per_worker=self.filter_per_worker,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=False,
                evaluation_phase=evaluation_phase,
                evaluation_phase_seed=self._phase_evaluation_seed(
                    evaluation_phase
                ),
            )
            self._evaluation_loaders[evaluation_phase] = loader
        return loader

    def _phase_evaluation_seed(
        self,
        phase: Literal["val", "test"],
    ) -> int:
        """Derive a stable, distinct torch seed for one evaluation phase."""
        payload = (
            f"topobench-fixed-evaluation:{self.evaluation_seed}:{phase}"
        ).encode()
        return (
            int.from_bytes(
                hashlib.sha256(payload).digest()[:8],
                byteorder="big",
            )
            & _MAX_EVALUATION_SEED
        )

    def evaluation_settings_descriptor(
        self,
        phase: Literal["val", "test"],
    ) -> str:
        """Return stable JSON for reproducible evaluation settings identity."""
        if phase not in {"val", "test"}:
            raise ValueError("Evaluation settings phase must be val or test")
        sampled_evaluation = (
            self.evaluation_protocol == "sampled_neighbor_fixed"
        )
        if sampled_evaluation and isinstance(self.num_neighbors, Mapping):
            fanout: object | None = [
                {
                    "edge_type": list(edge_type),
                    "num_neighbors": list(self.num_neighbors[edge_type]),
                }
                for edge_type in self.spec.edge_types
            ]
        elif sampled_evaluation:
            fanout = list(self.num_neighbors)
        else:
            fanout = None

        mask = self.data[self.spec.target_node_type][f"{phase}_mask"]
        phase_seed_ids = (
            mask.nonzero(as_tuple=False)
            .reshape(-1)
            .detach()
            .cpu()
            .to(torch.int64)
        )
        phase_seed_digest = hashlib.sha256()
        phase_seed_digest.update(
            len(phase_seed_ids).to_bytes(8, byteorder="big")
        )
        for seed_id in phase_seed_ids:
            phase_seed_digest.update(
                int(seed_id).to_bytes(8, byteorder="big", signed=True)
            )

        descriptor = {
            "batch_size": self.batch_size if sampled_evaluation else 1,
            "edge_counts": [
                int(self.data[edge_type].edge_index.size(1))
                for edge_type in self.spec.edge_types
            ],
            "edge_types": [
                list(edge_type) for edge_type in self.spec.edge_types
            ],
            "evaluation_num_workers": self.num_workers,
            "evaluation_persistent_workers": (
                False if sampled_evaluation else self.persistent_workers
            ),
            "evaluation_protocol": self.evaluation_protocol,
            "evaluation_seed": (
                self.evaluation_seed if sampled_evaluation else None
            ),
            "fanout": fanout,
            "filter_per_worker": (
                self.filter_per_worker if sampled_evaluation else None
            ),
            "mode": self.mode,
            "node_counts": {
                node_type: int(self.data[node_type].num_nodes)
                for node_type in self.spec.node_types
            },
            "node_types": list(self.spec.node_types),
            "phase": phase,
            "phase_seed": (
                self._phase_evaluation_seed(phase)
                if sampled_evaluation
                else None
            ),
            "phase_seed_count": len(phase_seed_ids),
            "phase_seed_ids_sha256": phase_seed_digest.hexdigest(),
            "replace": self.replace if sampled_evaluation else None,
            "subgraph_type": (
                self.subgraph_type if sampled_evaluation else None
            ),
            "target_node_type": self.spec.target_node_type,
            "versions": {
                "pyg-lib": (
                    _distribution_version("pyg-lib")
                    if sampled_evaluation
                    else None
                ),
                "torch-sparse": (
                    _distribution_version("torch-sparse")
                    if sampled_evaluation
                    else None
                ),
                "torch_geometric": torch_geometric.__version__,
            },
        }
        return json.dumps(descriptor, sort_keys=True, separators=(",", ":"))

    def _loader(self, phase: Phase) -> DataLoader | NeighborLoader:
        """Dispatch one phase to the configured native batching protocol."""
        if self.mode == "full_batch":
            return self._full_loader()
        return self._neighbor_loader(phase)

    @contextmanager
    def noncommitting_probe_batches(
        self,
        phases: Sequence[str],
    ) -> Iterator[dict[str, HeteroData]]:
        """Sample through a throwaway module without retaining loaders or RNG."""
        requested = tuple(phases)
        if not requested or any(
            phase not in {"train", "val", "test"} for phase in requested
        ):
            raise ValueError(
                "probe phases must contain train, val, and/or test"
            )
        rng_state = torch.random.get_rng_state().clone()
        probe = HeterogeneousNodeDataModule(
            self.data,
            self.spec,
            mode=self.mode,
            batch_size=self.batch_size,
            num_neighbors=self.num_neighbors,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=False,
            train_shuffle=self.train_shuffle,
            replace=self.replace,
            subgraph_type=self.subgraph_type,
            filter_per_worker=self.filter_per_worker,
            evaluation_protocol=self.evaluation_protocol,
            evaluation_seed=self.evaluation_seed,
        )
        iterators: list[Iterator[HeteroData]] = []
        batches: dict[str, HeteroData] = {}
        try:
            for phase in requested:
                loader = getattr(probe, f"{phase}_dataloader")()
                iterator = iter(loader)
                iterators.append(iterator)
                try:
                    batches[phase] = next(iterator)
                except StopIteration as error:
                    raise ValueError(
                        f"{phase} dataloader has no representative batch"
                    ) from error
            yield batches
        finally:
            for iterator in iterators:
                worker_iterator = getattr(iterator, "iterator", iterator)
                shutdown = getattr(worker_iterator, "_shutdown_workers", None)
                if callable(shutdown):
                    shutdown()
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()
            for loader in probe._evaluation_loaders.values():
                close = getattr(loader, "close", None)
                if callable(close):
                    close()
            probe._evaluation_loaders.clear()
            torch.random.set_rng_state(rng_state)

    def train_dataloader(self) -> DataLoader | NeighborLoader:
        """Return a fresh training loader."""
        return self._loader("train")

    def val_dataloader(self) -> DataLoader | NeighborLoader:
        """Return a mode-specific validation loader.

        Full-graph mode constructs a fresh loader per call; neighbor mode
        returns its memoized deterministic sampler.
        """
        return self._loader("val")

    def test_dataloader(self) -> DataLoader | NeighborLoader:
        """Return a mode-specific test loader.

        Full-graph mode constructs a fresh loader per call; neighbor mode
        returns its memoized deterministic sampler.
        """
        return self._loader("test")


__all__ = [
    "HeterogeneousNodeDataModule",
]
