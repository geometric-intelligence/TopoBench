"""Separate PyG batching for native heterogeneous node classification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Literal

from lightning import LightningDataModule
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.typing import EdgeType

from topobench.data.heterogeneous import HeterogeneousDataSpec

Phase = Literal["train", "val", "test"]
SamplingMode = Literal["full_batch", "neighbor"]
Fanout = list[int] | dict[EdgeType, list[int]]


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


class HeterogeneousNodeDataModule(LightningDataModule):
    """Batch one native heterogeneous graph without TopoBench collation."""

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
        return NeighborLoader(
            self.data,
            input_nodes=(self.spec.target_node_type, mask),
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle if phase == "train" else False,
            replace=self.replace,
            subgraph_type=self.subgraph_type,
            filter_per_worker=self.filter_per_worker,
            **self._common_kwargs(),
        )

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
