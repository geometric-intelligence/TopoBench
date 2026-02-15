"""Unit tests for cluster-related utilities and ClusterOnDisk."""

from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from topobench.data.utils import (
    ClusterOnDisk,
    _tensor_schema_entry,
    build_cluster_transform,
    to_bool_mask,
)


@pytest.fixture
def tiny_data() -> Data:
    """
    Create a tiny test graph.

    Returns
    -------
    Data
        Simple 3-node graph with features, labels and edge attributes.
    """
    edge_index = torch.tensor(
        [[0, 1, 1, 2],
         [1, 0, 2, 1]],
        dtype=torch.long,
    )
    x = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([0, 1, 0], dtype=torch.long)
    edge_attr = torch.arange(edge_index.size(1), dtype=torch.float32).view(-1, 1)
    return Data(edge_index=edge_index, x=x, y=y, edge_attr=edge_attr)


@pytest.fixture
def graph_getter(tiny_data: Data):
    """
    Build a graph_getter callable for ClusterOnDisk.

    Parameters
    ----------
    tiny_data : Data
        Tiny test graph.

    Returns
    -------
    callable
        Function returning a cloned copy of the graph.
    """
    def _getter() -> Data:
        return tiny_data.clone()

    return _getter


def test_build_cluster_transform_none_returns_none():
    """
    Test that empty configs return no transform.

    Notes
    -----
    Both empty dict and ``None`` should yield ``None``.
    """
    assert build_cluster_transform({}) is None
    assert build_cluster_transform(None) is None


def test_build_cluster_transform_single_transform(monkeypatch: pytest.MonkeyPatch):
    """
    Test building a single transform from config.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to patch hydra and DataTransform.
    """
    calls: dict[str, Any] = {}

    def fake_instantiate(cfg):
        calls["cfg"] = cfg
        return object()

    class DummyTransform:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, data: Data) -> Data:
            data.marked = True
            return data

    monkeypatch.setattr(
        hydra.utils,
        "instantiate",
        fake_instantiate,
    )
    monkeypatch.setattr(
        "topobench.transforms.data_transform.DataTransform",
        DummyTransform,
        raising=True,
    )

    cfg = {
        "t1": {"_target_": "dummy.target", "foo": 1},
    }
    transform = build_cluster_transform(cfg)

    assert isinstance(transform, DummyTransform)
    assert "t1" in calls["cfg"]

    d = Data()
    d2 = transform(d)
    assert d2.marked is True


def test_build_cluster_transform_multiple_transforms(
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Test building a composed transform from multiple configs.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to patch hydra and DataTransform.
    """
    def fake_instantiate(cfg):
        return object()

    class DummyTransform:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, data: Data) -> Data:
            data.calls = getattr(data, "calls", 0) + 1
            return data

    monkeypatch.setattr(
        hydra.utils,
        "instantiate",
        fake_instantiate,
    )
    monkeypatch.setattr(
        "topobench.transforms.data_transform.DataTransform",
        DummyTransform,
        raising=True,
    )

    cfg = {
        "liftings": {
            "t1": {"_target_": "dummy.t1"},
            "t2": {"_target_": "dummy.t2"},
        }
    }
    transform = build_cluster_transform(cfg)

    from torch_geometric.transforms import Compose

    assert isinstance(transform, Compose)
    assert len(transform.transforms) == 2
    assert all(isinstance(t, DummyTransform) for t in transform.transforms)

    d = Data()
    d2 = transform(d)
    assert d2.calls == 2


def test_to_bool_mask_various_inputs():
    """
    Test to_bool_mask for several input types.

    Notes
    -----
    Covers boolean masks, index lists and score vectors.
    """
    N = 5

    m_bool = torch.tensor([True, False, True, False, False])
    out = to_bool_mask(m_bool, N)
    assert out.dtype == torch.bool
    assert out.tolist() == m_bool.tolist()

    m_idx = torch.tensor([0, 2, 4], dtype=torch.long)
    out = to_bool_mask(m_idx, N)
    assert out.tolist() == [True, False, True, False, True]

    m_scores = torch.tensor([0, 1, 0, 0, 1], dtype=torch.float32)
    out = to_bool_mask(m_scores, N)
    assert out.tolist() == [False, True, False, False, True]

    m_bad = torch.tensor([1.0, 1.0])
    out = to_bool_mask(m_bad, N)
    assert out.tolist() == [False] * N


def test_tensor_schema_entry_scalar_and_nd():
    """
    Test _tensor_schema_entry for scalar and 2D tensors.

    Notes
    -----
    Ensures correct mapping to Python types and dict schema.
    """
    t_int = torch.tensor(3, dtype=torch.int64)
    assert _tensor_schema_entry(t_int) is int

    t_float = torch.tensor(1.5, dtype=torch.float32)
    assert _tensor_schema_entry(t_float) is float

    t_bool = torch.tensor(True, dtype=torch.bool)
    assert _tensor_schema_entry(t_bool) is bool

    t_2d = torch.zeros(4, 7, dtype=torch.float32)
    entry_2d = _tensor_schema_entry(t_2d)
    assert isinstance(entry_2d, dict)
    assert entry_2d["dtype"] == torch.float32
    assert entry_2d["size"] == (-1, 7)


@pytest.fixture
def cluster_ds(tmp_path: Path, graph_getter):
    """
    Create a ClusterOnDisk dataset instance.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory for dataset storage.
    graph_getter : callable
        Function returning the full graph.

    Returns
    -------
    ClusterOnDisk
        Prepared on-disk dataset.
    """
    root = tmp_path / "cluster_on_disk_ds"
    ds = ClusterOnDisk(
        root=str(root),
        graph_getter=graph_getter,
        num_parts=2,
        recursive=False,
        keep_inter_cluster_edges=False,
        sparse_format="csr",
        backend="sqlite",
    )
    return ds


def test_cluster_on_disk_schema_and_meta(cluster_ds: ClusterOnDisk):
    """
    Test schema discovery and stored metadata.

    Parameters
    ----------
    cluster_ds : ClusterOnDisk
        On-disk dataset under test.
    """
    assert "edge_index" in cluster_ds.schema
    assert "x" in cluster_ds.schema
    assert "y" in cluster_ds.schema
    assert "edge_attr" in cluster_ds.schema
    assert "num_nodes" in cluster_ds.schema

    meta = cluster_ds.meta
    assert "num_parts" in meta
    assert "recursive" in meta
    assert "keep_inter_cluster_edges" in meta
    assert "sparse_format" in meta
    assert "partition" in meta

    assert cluster_ds.num_parts == int(meta["num_parts"])
    assert cluster_ds.recursive == bool(meta["recursive"])
    assert cluster_ds.keep_inter_cluster_edges == bool(meta["keep_inter_cluster_edges"])
    assert cluster_ds.sparse_format == str(meta["sparse_format"])


def test_cluster_on_disk_len_and_getitem(cluster_ds: ClusterOnDisk):
    """
    Test length and indexing for ClusterOnDisk.

    Parameters
    ----------
    cluster_ds : ClusterOnDisk
        On-disk dataset under test.
    """
    n_parts_meta = cluster_ds.num_parts
    assert len(cluster_ds) == n_parts_meta

    data0 = cluster_ds[0]
    assert isinstance(data0, Data)
    assert getattr(data0, "edge_index", None) is not None

    for key in data0.keys():
        assert key in cluster_ds.schema


def test_cluster_on_disk_serialize_deserialize(cluster_ds: ClusterOnDisk):
    """
    Test serialize and deserialize of a single cluster.

    Parameters
    ----------
    cluster_ds : ClusterOnDisk
        On-disk dataset under test.
    """
    data0 = cluster_ds[0]
    row = cluster_ds.serialize(data0)
    assert "edge_index" in row

    data_rec = cluster_ds.deserialize(row)
    assert isinstance(data_rec, Data)
    assert torch.equal(data_rec.edge_index, data0.edge_index)
    if hasattr(data0, "x"):
        assert torch.equal(data_rec.x, data0.x)


def test_cluster_on_disk_perm_memmaps(cluster_ds: ClusterOnDisk, tiny_data: Data):
    """
    Test shapes and permutation logic of permuted memmaps.

    Parameters
    ----------
    cluster_ds : ClusterOnDisk
        On-disk dataset under test.
    tiny_data : Data
        Original tiny input graph.
    """
    mm_dir = Path(cluster_ds._memmap_dir())

    partptr = np.load(mm_dir / "partptr.npy")
    indptr = np.load(mm_dir / "indptr.npy")
    indices = np.load(mm_dir / "indices.npy")
    perm_to_global = np.load(mm_dir / "perm_to_global.npy")
    global_to_perm = np.load(mm_dir / "global_to_perm.npy")

    N = tiny_data.num_nodes
    assert perm_to_global.shape == (N,)
    assert global_to_perm.shape == (N,)

    for perm_idx, global_id in enumerate(perm_to_global.tolist()):
        assert global_to_perm[global_id] == perm_idx

    assert partptr.ndim == 1
    assert indptr.ndim == 1
    assert indices.ndim == 1

    if hasattr(tiny_data, "x"):
        X_perm = np.load(mm_dir / "X_perm.npy")
        assert X_perm.shape == (N, tiny_data.x.size(1))
        x_np = tiny_data.x.cpu().numpy()
        for i in range(N):
            assert np.allclose(X_perm[i], x_np[perm_to_global[i]])

    if hasattr(tiny_data, "y"):
        y_perm = np.load(mm_dir / "y_perm.npy")
        y_np = tiny_data.y.view(-1).cpu().numpy()
        assert y_perm.shape == (N,)
        for i in range(N):
            assert y_perm[i] == y_np[perm_to_global[i]]

    if hasattr(tiny_data, "edge_attr"):
        ea_perm = np.load(mm_dir / "edge_attr_perm.npy")
        assert ea_perm.ndim == 2
