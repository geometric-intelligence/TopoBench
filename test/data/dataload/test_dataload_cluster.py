"""Unit tests for Cluster-GCN dataloader and collator."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from topobench.dataloader.dataload_cluster import (
    BlockCSRBatchCollator,
    ClusterGCNDataModule,
)


class FakeAdapter:
    """Minimal adapter to mimic _HandleAdapter.

    Parameters
    ----------
    handle : dict
        Handle dictionary passed from the data module.
    """

    def __init__(self, handle: dict[str, Any]) -> None:
        paths = handle.get("paths", {})
        self.processed_dir: str = str(paths["processed_dir"])
        self.sparse_format: str = handle.get("sparse_format", "csr")
        self.num_parts: int = int(handle.get("num_parts", 2))


@pytest.fixture
def memmap_dir(tmp_path: Path) -> Path:
    """Create a synthetic perm_memmap directory.

    Returns
    -------
    Path
        Directory that plays the role of processed_dir.
    """
    processed_dir = tmp_path / "processed"
    mm_dir = processed_dir / "perm_memmap"
    mm_dir.mkdir(parents=True, exist_ok=True)

    # Part pointer: two parts, [0,2) and [2,4)
    partptr = np.array([0, 2, 4], dtype=np.int64)

    # CSR for a simple chain 0-1-2-3 (undirected via adjacency)
    # Row 0: [1]
    # Row 1: [0, 2]
    # Row 2: [1, 3]
    # Row 3: [2]
    indptr = np.array([0, 1, 3, 5, 6], dtype=np.int64)
    indices = np.array([1, 0, 2, 1, 3, 2], dtype=np.int64)

    X = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    edge_attr = np.arange(indices.shape[0], dtype=np.float32)

    train_mask = np.array([True, True, False, False], dtype=bool)
    val_mask = np.array([False, False, True, False], dtype=bool)
    test_mask = np.array([False, False, False, True], dtype=bool)

    np.save(mm_dir / "partptr.npy", partptr)
    np.save(mm_dir / "indptr.npy", indptr)
    np.save(mm_dir / "indices.npy", indices)
    np.save(mm_dir / "X_perm.npy", X)
    np.save(mm_dir / "y_perm.npy", y)
    np.save(mm_dir / "edge_attr_perm.npy", edge_attr)
    np.save(mm_dir / "train_mask_perm.npy", train_mask)
    np.save(mm_dir / "val_mask_perm.npy", val_mask)
    np.save(mm_dir / "test_mask_perm.npy", test_mask)

    return processed_dir


@pytest.fixture
def fake_adapter(memmap_dir: Path) -> FakeAdapter:
    """Return a FakeAdapter pointing to the synthetic memmap directory."""
    handle = {
        "paths": {"processed_dir": str(memmap_dir)},
        "sparse_format": "csr",
        "num_parts": 2,
    }
    return FakeAdapter(handle)


def test_block_csr_collator_active_mask_array(fake_adapter: FakeAdapter):
    """Test that _active_mask_array returns correct split masks."""
    coll_train = BlockCSRBatchCollator(fake_adapter, active_split="train")
    coll_val = BlockCSRBatchCollator(fake_adapter, active_split="val")
    coll_test = BlockCSRBatchCollator(fake_adapter, active_split="test")

    train_mask = coll_train._active_mask_array()
    val_mask = coll_val._active_mask_array()
    test_mask = coll_test._active_mask_array()

    assert train_mask.tolist() == [True, True, False, False]
    assert val_mask.tolist() == [False, False, True, False]
    assert test_mask.tolist() == [False, False, False, True]


def test_block_csr_collator_call_single_part(fake_adapter: FakeAdapter):
    """Test batching for a single cluster ID."""
    coll = BlockCSRBatchCollator(fake_adapter, active_split="train", with_edge_attr=True)
    data = coll([0])  # part 0: nodes 0,1

    assert data.num_nodes == 2
    assert data.x.shape == (2, 2)
    assert data.y.shape == (2,)
    assert data.edge_index.shape[0] == 2
    assert data.supervised_mask.tolist() == [True, True]
    assert data.global_nid.tolist() == [0, 1]
    assert torch.allclose(data.x_0, data.x)
    assert data.batch_0.shape == (data.num_nodes,)


def test_block_csr_collator_call_two_parts(fake_adapter: FakeAdapter):
    """Test batching for two cluster IDs in one mini-batch."""
    coll = BlockCSRBatchCollator(fake_adapter, active_split="val", with_edge_attr=True)
    data = coll([0, 1])

    assert data.num_nodes == 4
    assert data.x.shape == (4, 2)
    assert data.y.shape == (4,)
    assert data.edge_attr.shape[0] == data.edge_index.shape[1]
    assert data.supervised_mask.tolist() == [False, False, True, False]
    assert data.global_nid.tolist() == [0, 1, 2, 3]


@pytest.fixture
def handle_with_parts(memmap_dir: Path, tmp_path: Path) -> dict[str, Any]:
    """Create a handle dict with paths and precomputed part lists."""
    parts_dir = tmp_path / "parts_lists"
    parts_dir.mkdir(parents=True, exist_ok=True)

    train_parts = np.array([0], dtype=np.int64)
    val_parts = np.array([1], dtype=np.int64)
    test_parts = np.array([0, 1], dtype=np.int64)

    train_path = parts_dir / "parts_with_train.npy"
    val_path = parts_dir / "parts_with_val.npy"
    test_path = parts_dir / "parts_with_test.npy"

    np.save(train_path, train_parts)
    np.save(val_path, val_parts)
    np.save(test_path, test_parts)

    return {
        "paths": {
            "processed_dir": str(memmap_dir),
            "parts_with_train": str(train_path),
            "parts_with_val": str(val_path),
            "parts_with_test": str(test_path),
        },
        "sparse_format": "csr",
        "num_parts": 2,
    }


@pytest.fixture
def datamodule_with_patch(
    monkeypatch: pytest.MonkeyPatch, handle_with_parts: dict[str, Any]
) -> ClusterGCNDataModule:
    """Return ClusterGCNDataModule with _HandleAdapter patched to FakeAdapter."""
    def fake_handle_adapter(handle: dict[str, Any]) -> FakeAdapter:
        return FakeAdapter(handle)

    monkeypatch.setattr(
        "topobench.dataloader.dataload_cluster._HandleAdapter",
        fake_handle_adapter,
    )

    dm = ClusterGCNDataModule(
        data_handle=handle_with_parts,
        q=1,
        num_workers=0,
        pin_memory=False,
        with_edge_attr=True,
        device=None,
    )
    return dm


def test_part_ids_for_split_uses_precomputed(datamodule_with_patch: ClusterGCNDataModule):
    """Test that _part_ids_for_split prefers precomputed parts_with_* arrays."""
    train_ids = datamodule_with_patch._part_ids_for_split("train")
    val_ids = datamodule_with_patch._part_ids_for_split("val")
    test_ids = datamodule_with_patch._part_ids_for_split("test")

    assert np.array_equal(train_ids, np.array([0], dtype=np.int64))
    assert np.array_equal(val_ids, np.array([1], dtype=np.int64))
    assert np.array_equal(test_ids, np.array([0, 1], dtype=np.int64))


def test_part_ids_for_split_fallback_all_parts(
    monkeypatch: pytest.MonkeyPatch, memmap_dir: Path
):
    """Test that _part_ids_for_split falls back to all parts if lists missing."""
    def fake_handle_adapter(handle: dict[str, Any]) -> FakeAdapter:
        return FakeAdapter(handle)

    monkeypatch.setattr(
        "topobench.dataloader.dataload_cluster._HandleAdapter",
        fake_handle_adapter,
    )

    handle = {
        "paths": {"processed_dir": str(memmap_dir)},
        "sparse_format": "csr",
        "num_parts": 2,
    }

    dm = ClusterGCNDataModule(data_handle=handle, q=1)
    expected = np.arange(2, dtype=np.int64)

    assert np.array_equal(dm._part_ids_for_split("train"), expected)
    assert np.array_equal(dm._part_ids_for_split("val"), expected)
    assert np.array_equal(dm._part_ids_for_split("test"), expected)


def test_build_loader_and_single_batch(datamodule_with_patch: ClusterGCNDataModule):
    """Test that _build_loader constructs a working DataLoader and batch."""
    loader = datamodule_with_patch._build_loader(split="train", shuffle=False)
    assert isinstance(loader, DataLoader)

    batch = next(iter(loader))
    assert batch.num_nodes == 2
    assert batch.supervised_mask.tolist() == [True, True]


def test_train_val_test_dataloaders(datamodule_with_patch: ClusterGCNDataModule):
    """Test that train/val/test_dataloader methods return working loaders."""
    train_loader = datamodule_with_patch.train_dataloader()
    val_loader = datamodule_with_patch.val_dataloader()
    test_loader = datamodule_with_patch.test_dataloader()

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    test_batches = list(iter(test_loader))

    # Train: only part 0 -> nodes 0,1 (both supervised in train split)
    assert train_batch.num_nodes == 2
    assert train_batch.supervised_mask.tolist() == [True, True]

    # Val: only part 1 -> nodes 2,3 (only node 2 supervised in val split)
    assert val_batch.num_nodes == 2
    assert val_batch.supervised_mask.tolist() == [True, False]

    # Test: parts 0 and 1, but q=1 -> two batches
    assert len(test_batches) == 2

    tb0, tb1 = test_batches

    # First test batch: part 0 -> nodes 0,1 (no supervised nodes in test split)
    assert tb0.num_nodes == 2
    assert tb0.supervised_mask.tolist() == [False, False]

    # Second test batch: part 1 -> nodes 2,3 (only node 3 supervised in test split)
    assert tb1.num_nodes == 2
    assert tb1.supervised_mask.tolist() == [False, True]
