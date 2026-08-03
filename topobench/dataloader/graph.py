"""Native PyG batching for homogeneous graph data."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import torch
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader

LearningSetting = Literal["inductive", "transductive"]
_VALIDATION_ATTRIBUTE = "_topobench_hypergraph_validation"
_VALIDATION_SCHEMA_VERSION = 1
_VALIDATED_TENSOR_FIELDS = (
    "x",
    "y",
    "hyperedge_index",
    "num_hyperedges",
    "train_mask",
    "val_mask",
    "test_mask",
    "batch",
    "ptr",
)
_VALIDATED_SCALAR_FIELDS = (
    "num_hyperedges",
    "representation_version",
    "num_graphs",
)


@dataclass(frozen=True)
class _TensorValidationSnapshot:
    """Constant-size mutation evidence for one boundary-validated tensor."""

    field: str
    tensor: Tensor
    version: int
    shape: torch.Size
    dtype: torch.dtype
    layout: torch.layout
    device: torch.device
    pinned: bool


@dataclass(frozen=True)
class _HypergraphValidationContext:
    """Pipeline qualification attached to current validation evidence."""

    selector: str | None
    num_classes: object


@dataclass(frozen=True)
class _HypergraphValidationMarker:
    """Evidence that exhaustive validation already ran at the data boundary."""

    tensors: tuple[_TensorValidationSnapshot, ...]
    scalars: tuple[tuple[str, object], ...]
    context: _HypergraphValidationContext
    schema_version: int


def _tensor_snapshot(field: str, tensor: Tensor) -> _TensorValidationSnapshot:
    """Capture metadata that detects replacement and in-place mutation."""
    return _TensorValidationSnapshot(
        field=field,
        tensor=tensor,
        version=tensor._version,
        shape=tensor.shape,
        dtype=tensor.dtype,
        layout=tensor.layout,
        device=tensor.device,
        pinned=tensor.is_pinned(),
    )


def mark_hypergraph_validated(
    data: Data,
    *,
    selector: str | None = None,
    num_classes: object = None,
) -> Data:
    """Mark one exhaustively validated hypergraph without scanning tensors."""
    if not isinstance(data, Data):
        raise TypeError("validated hypergraph data must be native PyG Data")
    tensors = tuple(
        _tensor_snapshot(field, value)
        for field in _VALIDATED_TENSOR_FIELDS
        if isinstance((value := data.get(field)), Tensor)
    )
    scalars = tuple(
        (field, value)
        for field in _VALIDATED_SCALAR_FIELDS
        if (value := data.get(field)) is not None
        and not isinstance(value, Tensor)
    )
    existing = getattr(data, _VALIDATION_ATTRIBUTE, None)
    context = (
        existing.context
        if selector is None
        and num_classes is None
        and isinstance(existing, _HypergraphValidationMarker)
        else _HypergraphValidationContext(
            selector=selector,
            num_classes=num_classes,
        )
    )
    setattr(
        data,
        _VALIDATION_ATTRIBUTE,
        _HypergraphValidationMarker(
            tensors=tensors,
            scalars=scalars,
            context=context,
            schema_version=_VALIDATION_SCHEMA_VERSION,
        ),
    )
    return data


def has_hypergraph_validation(
    data: Data,
    *,
    selector: str | None = None,
    num_classes: object = None,
) -> bool:
    """Return whether native data carries current matching evidence."""
    marker = getattr(data, _VALIDATION_ATTRIBUTE, None)
    if not isinstance(marker, _HypergraphValidationMarker):
        return False
    if selector is not None and (
        marker.context.selector != selector
        or type(marker.context.num_classes) is not type(num_classes)
        or marker.context.num_classes != num_classes
    ):
        return False
    try:
        require_hypergraph_validation(data)
    except (TypeError, ValueError):
        return False
    return True


def require_hypergraph_validation(
    data: Data,
) -> _HypergraphValidationContext:
    """Reject post-boundary injection using constant-size mutation evidence."""
    marker = getattr(data, _VALIDATION_ATTRIBUTE, None)
    if not isinstance(marker, _HypergraphValidationMarker):
        raise ValueError("batch is not boundary-validated hypergraph data")
    if marker.schema_version != _VALIDATION_SCHEMA_VERSION:
        raise ValueError("batch hypergraph validation marker schema is stale")
    expected_tensor_fields = tuple(
        field
        for field in _VALIDATED_TENSOR_FIELDS
        if isinstance(data.get(field), Tensor)
    )
    if tuple(snapshot.field for snapshot in marker.tensors) != (
        expected_tensor_fields
    ):
        raise ValueError("batch hypergraph validation marker schema is stale")
    expected_scalar_fields = tuple(
        field
        for field in _VALIDATED_SCALAR_FIELDS
        if data.get(field) is not None
        and not isinstance(data.get(field), Tensor)
    )
    if tuple(field for field, _ in marker.scalars) != expected_scalar_fields:
        raise ValueError("batch hypergraph validation marker schema is stale")
    for field, expected in marker.scalars:
        current = data.get(field)
        if type(current) is not type(expected) or current != expected:
            raise ValueError(
                f"batch.{field} changed after boundary validation"
            )

    replacements: list[tuple[_TensorValidationSnapshot, Tensor]] = []
    for snapshot in marker.tensors:
        current = data.get(snapshot.field)
        if not isinstance(current, Tensor):
            raise ValueError(
                f"batch.{snapshot.field} changed after boundary validation"
            )
        if current is snapshot.tensor:
            if current._version != snapshot.version:
                raise ValueError(
                    f"batch.{snapshot.field} changed after boundary validation"
                )
            continue
        if (
            current.shape != snapshot.shape
            or current.dtype != snapshot.dtype
            or current.layout != snapshot.layout
        ):
            raise ValueError(
                f"batch.{snapshot.field} changed after boundary validation"
            )
        replacements.append((snapshot, current))

    if not replacements:
        return marker.context

    device_transfer = all(
        current.device != snapshot.device for snapshot, current in replacements
    )
    pin_transfer = all(
        current.device == snapshot.device
        and not snapshot.pinned
        and current.is_pinned()
        for snapshot, current in replacements
    )
    if not device_transfer and not pin_transfer:
        changed_field = replacements[0][0].field
        raise ValueError(
            f"batch.{changed_field} changed after boundary validation"
        )
    mark_hypergraph_validated(data)
    refreshed = getattr(data, _VALIDATION_ATTRIBUTE)
    return refreshed.context


def hypergraph_validation_context(
    data: Data,
) -> _HypergraphValidationContext | None:
    """Return current evidence context when data is a marked hypergraph."""
    marker = getattr(data, _VALIDATION_ATTRIBUTE, None)
    if not isinstance(marker, _HypergraphValidationMarker):
        return None
    return require_hypergraph_validation(data)


def _singleton_hypergraph_view(data: Data) -> Data:
    """Create one batch-shaped store view while aliasing validated tensors."""
    require_hypergraph_validation(data)
    view = copy(data)
    x = view.get("x")
    if not isinstance(x, Tensor) or x.ndim != 2:
        raise ValueError("validated hypergraph data requires rank-2 x")
    view.batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    view.ptr = torch.tensor(
        [0, x.size(0)],
        dtype=torch.long,
        device=x.device,
    )
    view.num_graphs = 1
    object.__setattr__(view, "_num_graphs", 1)
    return mark_hypergraph_validated(view)


def _identity_collate(data: Data) -> Data:
    """Return the singleton data object without PyG recollation."""
    return data


def _validate_integer(
    name: str,
    value: int,
    *,
    minimum: int,
    range_message: str,
) -> int:
    """Validate a loader integer without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(range_message)
    return normalized


def _require_non_empty(name: str, dataset: Dataset[Data]) -> None:
    """Reject an empty phase dataset with a stable boundary error."""
    if len(dataset) == 0:
        raise ValueError(f"dataset_{name} must not be empty")


def loader_worker_options(
    *,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> dict[str, int | bool]:
    """Return the worker lifecycle options shared by native graph loaders."""
    return {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
    }


class GraphDataModule(LightningDataModule):
    """Batch native homogeneous PyG graphs for one learning setting.

    Inductive phases are separate, non-empty dataset views and only training is
    shuffled. Transductive loading accepts one source graph, exposes that same
    singleton dataset in every phase, and never accepts phase-specific copies.

    Parameters
    ----------
    dataset_train : Dataset[Data]
        Training view, or the one source dataset in transductive mode.
    dataset_val : Dataset[Data] | None, optional
        Non-empty validation view required only for inductive mode.
    dataset_test : Dataset[Data] | None, optional
        Non-empty test view required only for inductive mode.
    learning_setting : {"inductive", "transductive"}
        Explicit graph learning mode.
    batch_size : int, default=1
        Positive number of graphs per batch. Must be one for transductive mode.
    num_workers : int, default=0
        Non-negative PyTorch data-loading worker count.
    pin_memory : bool, default=False
        Forwarded to the PyG loader.
    persistent_workers : bool, default=False
        Forwarded when workers are enabled; forced off for zero workers.
    """

    def __init__(
        self,
        dataset_train: Dataset[Data],
        dataset_val: Dataset[Data] | None = None,
        dataset_test: Dataset[Data] | None = None,
        *,
        learning_setting: LearningSetting,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ) -> None:
        super().__init__()
        batch_size = _validate_integer(
            "batch_size",
            batch_size,
            minimum=1,
            range_message="batch_size must be positive",
        )
        num_workers = _validate_integer(
            "num_workers",
            num_workers,
            minimum=0,
            range_message="num_workers must be non-negative",
        )
        _require_non_empty("train", dataset_train)
        singleton_hypergraph = False

        if learning_setting == "transductive":
            if len(dataset_train) != 1:
                raise ValueError(
                    "transductive graph loading requires exactly one graph"
                )
            if dataset_val is not None or dataset_test is not None:
                raise ValueError(
                    "transductive phases must reuse the source graph"
                )
            if batch_size != 1:
                raise ValueError(
                    "transductive graph loading requires batch_size=1"
                )
            source_data = dataset_train[0]
            if has_hypergraph_validation(source_data):
                dataset_train = [_singleton_hypergraph_view(source_data)]
                singleton_hypergraph = True
            dataset_val = dataset_test = dataset_train
        elif learning_setting == "inductive":
            if dataset_val is None or dataset_test is None:
                raise ValueError(
                    "inductive loading requires train, validation, and test views"
                )
            if not all(
                isinstance(dataset, Subset)
                for dataset in (dataset_train, dataset_val, dataset_test)
            ):
                raise ValueError(
                    "inductive loading requires index-backed Subset views"
                )
            if not (
                dataset_train.dataset
                is dataset_val.dataset
                is dataset_test.dataset
            ):
                raise ValueError(
                    "inductive phase views must share one source dataset"
                )
            _require_non_empty("validation", dataset_val)
            _require_non_empty("test", dataset_test)
        else:
            raise ValueError(
                f"unsupported learning_setting: {learning_setting!r}"
            )

        self.learning_setting = learning_setting
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.batch_size = batch_size
        self.loader_kwargs = loader_worker_options(
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        self._singleton_hypergraph = singleton_hypergraph

    def _loader(
        self,
        dataset: Dataset[Data],
        *,
        shuffle: bool,
    ) -> TorchDataLoader | GeometricDataLoader:
        """Use a direct singleton view or ordinary native PyG collation."""
        if self._singleton_hypergraph:
            return TorchDataLoader(
                dataset,
                batch_size=None,
                shuffle=False,
                num_workers=0,
                pin_memory=bool(self.loader_kwargs["pin_memory"]),
                collate_fn=_identity_collate,
            )
        return GeometricDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            **self.loader_kwargs,
        )

    @contextmanager
    def noncommitting_probe_batches(
        self,
        phases: Sequence[str],
    ) -> Iterator[dict[str, Data]]:
        """Yield one phase batch while restoring loader RNG and releasing workers."""
        requested = tuple(phases)
        if not requested or any(
            phase not in {"train", "val", "test"} for phase in requested
        ):
            raise ValueError(
                "probe phases must contain train, val, and/or test"
            )
        rng_state = torch.random.get_rng_state().clone()
        iterators: list[Iterator[Data]] = []
        batches: dict[str, Data] = {}
        try:
            for phase in requested:
                loader = getattr(self, f"{phase}_dataloader")()
                iterator = iter(loader)
                iterators.append(iterator)
                try:
                    batch = next(iterator)
                    disposable = batch.clone()
                    batches[phase] = (
                        mark_hypergraph_validated(disposable)
                        if has_hypergraph_validation(batch)
                        else disposable
                    )
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
            torch.random.set_rng_state(rng_state)

    def train_dataloader(self) -> TorchDataLoader | GeometricDataLoader:
        """Return the training graph loader."""
        return self._loader(
            self.dataset_train,
            shuffle=self.learning_setting == "inductive",
        )

    def val_dataloader(self) -> TorchDataLoader | GeometricDataLoader:
        """Return the deterministic validation graph loader."""
        return self._loader(self.dataset_val, shuffle=False)

    def test_dataloader(self) -> TorchDataLoader | GeometricDataLoader:
        """Return the deterministic test graph loader."""
        return self._loader(self.dataset_test, shuffle=False)
