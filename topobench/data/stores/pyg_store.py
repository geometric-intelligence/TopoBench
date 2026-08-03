"""Read-only PyG protocol views over one validated typed graph store."""

from __future__ import annotations

import sys
import warnings
from typing import Any

import torch

# PyG treats PyArrow as optional, but its package initializer otherwise imports
# it eagerly. Mask that optional dependency only while importing the protocol
# classes; explicit external-ID restoration remains the sole PyArrow boundary.
_mask_pyarrow = "pyarrow" not in sys.modules
if _mask_pyarrow:
    sys.modules["pyarrow"] = None
try:
    from torch_geometric.data import (
        EdgeAttr,
        FeatureStore,
        GraphStore,
        TensorAttr,
    )
    from torch_geometric.data.feature_store import _FieldStatus
    from torch_geometric.data.graph_store import EdgeLayout
finally:
    if _mask_pyarrow:
        sys.modules.pop("pyarrow", None)

from topobench.data.stores.typed_graph_store import (  # noqa: E402
    TypedGraphStore,
    _row_indices,
)


class _SafeTensorAttr(TensorAttr):
    """Work around PyG 2.8 NumPy-index equality in ``TensorAttr.is_set``."""

    @classmethod
    def cast(cls, *args: Any, **kwargs: Any) -> _SafeTensorAttr:
        if len(args) == 1 and not kwargs and isinstance(args[0], TensorAttr):
            attr = args[0]
            return cls(attr.group_name, attr.attr_name, attr.index)
        return super().cast(*args, **kwargs)

    def is_set(self, key: str) -> bool:
        return getattr(self, key) is not _FieldStatus.UNSET


class PyGTypedFeatureStore(FeatureStore):
    """Actual installed PyG FeatureStore protocol with bounded row reads."""

    def __init__(self, store: TypedGraphStore) -> None:
        if not isinstance(store, TypedGraphStore):
            raise TypeError("store must be a TypedGraphStore")
        super().__init__(_SafeTensorAttr)
        self.store = store

    def _put_tensor(self, tensor: Any, attr: TensorAttr) -> bool:
        raise PermissionError("immutable typed feature store is read-only")

    def _get_tensor(self, attr: TensorAttr) -> Any:
        group = attr.group_name
        name = attr.attr_name
        if group not in self.store.node_types or name not in {"x", "y"}:
            return None
        try:
            return self.store.node_array(group, name, attr.index)
        except KeyError:
            return None

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        raise PermissionError("immutable typed feature store is read-only")

    def _get_tensor_size(self, attr: TensorAttr) -> tuple[int, ...] | None:
        try:
            node = self.store._node(attr.group_name)
            record = (
                node[attr.attr_name] if attr.attr_name in {"x", "y"} else None
            )
        except (KeyError, TypeError):
            return None
        if record is None:
            return None
        shape = tuple(record["shape"])
        indices = _row_indices(attr.index, shape[0])
        if indices is None:
            return shape
        selected = (
            len(range(*indices.indices(shape[0])))
            if isinstance(indices, slice)
            else len(indices)
        )
        return (selected, *shape[1:])

    def get_all_tensor_attrs(self) -> list[TensorAttr]:
        return [
            TensorAttr(node["name"], name, None)
            for node in self.store._manifest["nodes"].values()
            for name in ("x", "y")
            if name == "x" or node["y"] is not None
        ]


class PyGTypedGraphStore(GraphStore):
    """Actual installed PyG GraphStore protocol exposing canonical CSC mmaps."""

    def __init__(self, store: TypedGraphStore) -> None:
        if not isinstance(store, TypedGraphStore):
            raise TypeError("store must be a TypedGraphStore")
        super().__init__()
        self.store = store
        self.meta = {"is_hetero": store.output_kind == "heterogeneous"}
        self._edge_tensors: dict[
            tuple[str, str, str],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}

    def _put_edge_index(self, edge_index: Any, edge_attr: EdgeAttr) -> bool:
        raise PermissionError("immutable typed graph store is read-only")

    def _get_edge_index(self, edge_attr: EdgeAttr) -> Any:
        if edge_attr.layout != EdgeLayout.CSC:
            return None
        expected = self._attr_by_type().get(tuple(edge_attr.edge_type))
        if expected is None:
            return None
        if edge_attr.size is not None and tuple(edge_attr.size) != tuple(
            expected.size
        ):
            return None
        relation = tuple(expected.edge_type)
        self.store._ensure_open()
        cached = self._edge_tensors.get(relation)
        if cached is None:
            row, colptr = self.store.relation_csc(relation)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The given NumPy array is not writable.*",
                    category=UserWarning,
                )
                cached = (
                    torch.from_numpy(row),
                    torch.from_numpy(colptr),
                )
            self._edge_tensors[relation] = cached
        return cached

    def _remove_edge_index(self, edge_attr: EdgeAttr) -> bool:
        raise PermissionError("immutable typed graph store is read-only")

    def get_all_edge_attrs(self) -> list[EdgeAttr]:
        manifest = self.store._manifest
        return [
            EdgeAttr(
                tuple(record["relation"]),
                EdgeLayout.CSC,
                is_sorted=True,
                size=(record["source_count"], record["destination_count"]),
            )
            for record in manifest["relations"].values()
        ]

    def _attr_by_type(self) -> dict[tuple[str, str, str], EdgeAttr]:
        return {
            tuple(attr.edge_type): attr for attr in self.get_all_edge_attrs()
        }


__all__ = ["PyGTypedFeatureStore", "PyGTypedGraphStore"]
