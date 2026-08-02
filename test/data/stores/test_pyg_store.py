"""Installed PyG FeatureStore and GraphStore protocol behavior."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import EdgeAttr, FeatureStore, GraphStore, TensorAttr
from torch_geometric.loader import NeighborLoader
from torch_geometric.data.graph_store import EdgeLayout

from topobench.data.stores.pyg_store import (
    PyGTypedFeatureStore,
    PyGTypedGraphStore,
)
from topobench.data.stores.typed_graph_store import TypedGraphStore
from test.data.stores.test_typed_graph_store import (
    QualifiedStoreFixture,
    qualified_stores,
)


def test_feature_store_implements_exact_read_only_protocol(
    qualified_stores: dict[str, QualifiedStoreFixture],
) -> None:
    typed = TypedGraphStore.open(
        qualified_stores["heterogeneous"].store_build.path
    )
    store = PyGTypedFeatureStore(typed)
    assert isinstance(store, FeatureStore)
    attrs = store.get_all_tensor_attrs()
    assert attrs == [
        TensorAttr("entity-kind", "x", None),
        TensorAttr("entity.kind", "x", None),
        TensorAttr("entity.kind", "y", None),
    ]
    assert typed.mapped_paths == ()
    assert store.get_tensor_size(TensorAttr("entity-kind", "x", None)) == (5, 3)
    assert typed.mapped_paths == ()
    selected_attr = TensorAttr(
        "entity-kind",
        "x",
        np.array([1, 3], dtype=np.int64),
    )
    selected = store.get_tensor(selected_attr)
    np.testing.assert_array_equal(
        selected,
        np.array([[2.0, 12.0, 22.0], [4.0, 14.0, 24.0]]),
    )
    assert selected.shape == (2, 3)
    assert selected.flags.writeable is False
    assert store.get_tensor_size(selected_attr) == (2, 3)
    np.testing.assert_array_equal(
        store.get_tensor(TensorAttr("entity.kind", "y", np.array([3, 0]))),
        np.array([3, 0], dtype=np.int64),
    )
    with pytest.raises(PermissionError, match="read-only"):
        store.put_tensor(np.zeros((1, 3)), selected_attr)
    with pytest.raises(PermissionError, match="read-only"):
        store.remove_tensor(selected_attr)
    typed.close()


def test_graph_store_exposes_existing_csc_mmaps_and_layout_metadata(
    qualified_stores: dict[str, QualifiedStoreFixture],
) -> None:
    typed = TypedGraphStore.open(
        qualified_stores["heterogeneous"].store_build.path
    )
    store = PyGTypedGraphStore(typed)
    assert isinstance(store, GraphStore)
    expected_relations = tuple(typed.relation_types)
    attrs = store.get_all_edge_attrs()
    assert [attr.edge_type for attr in attrs] == list(expected_relations)
    assert all(attr.layout == EdgeLayout.CSC for attr in attrs)
    assert all(attr.is_sorted for attr in attrs)
    assert [attr.size for attr in attrs] == [
        (5, 5),
        (5, 4),
        (4, 5),
    ]
    relation = ("entity.kind", "writes", "entity-kind")
    attr = next(item for item in attrs if item.edge_type == relation)
    row, colptr = store.get_edge_index(attr)
    assert isinstance(row, torch.Tensor)
    assert isinstance(colptr, torch.Tensor)
    assert row.dtype == torch.int64 and colptr.dtype == torch.int64
    assert row.device.type == "cpu" and colptr.device.type == "cpu"
    same_row, same_colptr = typed.relation_csc(relation)
    assert row.data_ptr() == same_row.ctypes.data
    assert colptr.data_ptr() == same_colptr.ctypes.data
    assert row.to(dtype=colptr.dtype) is row
    cached_row, cached_colptr = store.get_edge_index(attr)
    assert cached_row is row and cached_colptr is colptr

    row_dict, colptr_dict, permutations = store.csc()
    assert row_dict[relation] is row
    assert colptr_dict[relation] is colptr
    assert permutations[relation] is None
    with pytest.raises(PermissionError, match="read-only"):
        store.put_edge_index((row, colptr), attr)
    with pytest.raises(PermissionError, match="read-only"):
        store.remove_edge_index(attr)
    typed.close()


def test_neighbor_loader_accepts_mmap_backed_csc_tensor_views(
    qualified_stores: dict[str, QualifiedStoreFixture],
) -> None:
    typed = TypedGraphStore.open(
        qualified_stores["heterogeneous"].store_build.path
    )
    graph_store = PyGTypedGraphStore(typed)
    loader = NeighborLoader(
        (PyGTypedFeatureStore(typed), graph_store),
        num_neighbors=[1],
        input_nodes=("entity.kind", torch.tensor([0, 1])),
        batch_size=1,
        shuffle=False,
    )
    batch = next(iter(loader))
    assert batch["entity.kind"].batch_size == 1
    assert all(
        value.dtype == torch.int64
        for value in graph_store.csc()[0].values()
    )
    typed.close()


def test_homogeneous_graph_store_reports_csc_without_heterogeneous_wrapping(
    qualified_stores: dict[str, QualifiedStoreFixture],
) -> None:
    typed = TypedGraphStore.open(
        qualified_stores["homogeneous"].store_build.path
    )
    store = PyGTypedGraphStore(typed)
    attr = EdgeAttr(
        ("node", "links", "node"),
        EdgeLayout.CSC,
        is_sorted=True,
        size=(3, 3),
    )
    row, colptr = store.get_edge_index(attr)
    csc_row, csc_colptr, permutation = store.csc()
    assert csc_row is row and csc_colptr is colptr and permutation is None
    typed.close()
