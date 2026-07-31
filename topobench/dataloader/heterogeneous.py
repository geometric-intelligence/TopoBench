"""Separate PyG batching for native heterogeneous node classification."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from numbers import Integral
from typing import Literal

import torch
from lightning import LightningDataModule
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.typing import EdgeType

from topobench.data.heterogeneous import HeterogeneousDataSpec

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


class _FixedEvaluationNeighborLoader(NeighborLoader):
    """NeighborLoader that materializes once and replays owner-held clones."""

    def __init__(
        self,
        *args: object,
        evaluation_owner: HeterogeneousNodeDataModule,
        evaluation_phase: Literal["val", "test"],
        **kwargs: object,
    ) -> None:
        """Bind one evaluation phase to its datamodule-owned replay cache."""
        self._evaluation_owner = evaluation_owner
        self._evaluation_phase = evaluation_phase
        super().__init__(*args, **kwargs)

    def _uncached_iterator(self) -> Iterator[HeteroData]:
        """Return the underlying PyG sampling iterator exactly once."""
        return super().__iter__()

    def __iter__(self) -> Iterator[HeteroData]:
        """Replay fresh clones of the datamodule-owned evaluation cache."""
        return self._evaluation_owner._fixed_evaluation_iterator(
            self._evaluation_phase,
            self,
        )


class HeterogeneousNodeDataModule(LightningDataModule):
    """Batch one native heterogeneous graph without TopoBench collation.

    The datamodule retains the validated graph reference and treats its graph
    and loader options as canonical for its lifetime. Callers must not mutate
    the graph or public option attributes after construction. Sampled
    validation and test batches are materialized once on CPU, then replayed as
    fresh clones so trainer device transfer and consumer mutation cannot alter
    the canonical evaluation context.
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
        self._evaluation_batch_cache: dict[
            str,
            tuple[HeteroData, ...],
        ] = {}

        # Deliberately omit ``save_hyperparameters``: relation-keyed fanout is
        # not reliably YAML/JSON serializable, and graph/spec must never enter
        # Lightning checkpoint hyperparameters.

    def _common_kwargs(self) -> dict[str, object]:
        """Return worker options shared by full and sampled PyG loaders."""
        return {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
        }

    def _full_loader(self) -> DataLoader:
        """Construct a fresh one-graph, non-shuffled PyG loader."""
        return DataLoader(
            [self.data],
            batch_size=1,
            shuffle=False,
            **self._common_kwargs(),
        )

    def _neighbor_loader(self, phase: Phase) -> NeighborLoader:
        """Construct a fresh target-mask-seeded PyG neighbor loader."""
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
        return _FixedEvaluationNeighborLoader(
            self.data,
            input_nodes=(self.spec.target_node_type, mask),
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            shuffle=False,
            replace=self.replace,
            subgraph_type=self.subgraph_type,
            filter_per_worker=self.filter_per_worker,
            **self._common_kwargs(),
            evaluation_owner=self,
            evaluation_phase=evaluation_phase,
        )

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

    def _materialize_fixed_evaluation(
        self,
        phase: Literal["val", "test"],
        loader: _FixedEvaluationNeighborLoader,
    ) -> tuple[HeteroData, ...]:
        """Sample one phase under isolated RNG state and cache CPU clones."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._phase_evaluation_seed(phase))
            try:
                batches = tuple(
                    batch.clone().cpu()
                    for batch in loader._uncached_iterator()
                )
            except ImportError as error:
                raise RuntimeError(
                    "Fixed heterogeneous sampling requires pyg-lib or "
                    f"torch-sparse; cannot materialize the {phase} phase"
                ) from error
            except RuntimeError as error:
                message = str(error).lower()
                if "pyg-lib" not in message and "torch-sparse" not in message:
                    raise
                raise RuntimeError(
                    "Fixed heterogeneous sampling requires pyg-lib or "
                    f"torch-sparse; cannot materialize the {phase} phase"
                ) from error
        if not batches:
            raise RuntimeError(
                f"Fixed heterogeneous {phase} sampling produced no batches"
            )
        self._evaluation_batch_cache[phase] = batches
        return batches

    def _fixed_evaluation_iterator(
        self,
        phase: Literal["val", "test"],
        loader: _FixedEvaluationNeighborLoader,
    ) -> Iterator[HeteroData]:
        """Yield fresh clones from the shared phase cache."""
        batches = self._evaluation_batch_cache.get(phase)
        if batches is None:
            batches = self._materialize_fixed_evaluation(phase, loader)
        return (batch.clone() for batch in batches)

    def evaluation_cache_descriptor(self, phase: Phase) -> str:
        """Return stable JSON describing one phase's evaluation identity."""
        if phase not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported phase for cache identity: {phase}")
        if isinstance(self.num_neighbors, Mapping):
            fanout: object = [
                {
                    "edge_type": list(edge_type),
                    "num_neighbors": list(self.num_neighbors[edge_type]),
                }
                for edge_type in self.spec.edge_types
            ]
        else:
            fanout = list(self.num_neighbors)
        descriptor = {
            "batch_size": self.batch_size,
            "evaluation_protocol": self.evaluation_protocol,
            "evaluation_seed": self.evaluation_seed,
            "fanout": fanout,
            "phase": phase,
            "replace": self.replace,
            "subgraph_type": self.subgraph_type,
        }
        return json.dumps(descriptor, sort_keys=True, separators=(",", ":"))

    def _loader(self, phase: Phase) -> DataLoader | NeighborLoader:
        """Dispatch one phase to the configured native batching protocol."""
        if self.mode == "full_batch":
            return self._full_loader()
        return self._neighbor_loader(phase)

    def train_dataloader(self) -> DataLoader | NeighborLoader:
        """Return a fresh training loader."""
        return self._loader("train")

    def val_dataloader(self) -> DataLoader | NeighborLoader:
        """Return a fresh validation loader."""
        return self._loader("val")

    def test_dataloader(self) -> DataLoader | NeighborLoader:
        """Return a fresh test loader."""
        return self._loader("test")


__all__ = [
    "HeterogeneousNodeDataModule",
]
