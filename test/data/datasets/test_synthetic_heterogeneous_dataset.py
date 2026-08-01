"""Tests for the deterministic native heterogeneous graph fixture."""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch
from torch_geometric.data import HeteroData

from topobench.data.datasets import (
    SyntheticHeterogeneousDataset,
    make_synthetic_heterogeneous_data,
)
from topobench.data.datasets.synthetic_heterogeneous_dataset import (
    _stratified_masks,
)


def _assert_valid_size_contract(data: HeteroData) -> None:
    """Assert invariants that must hold for every accepted fixture size."""
    labels = data["author"].y
    masks = [
        data["author"].train_mask,
        data["author"].val_mask,
        data["author"].test_mask,
    ]
    for mask in masks:
        assert mask.dtype == torch.bool
        assert torch.any(mask)
    assert not torch.any(masks[0] & masks[1])
    assert not torch.any(masks[0] & masks[2])
    assert not torch.any(masks[1] & masks[2])
    assert torch.all(masks[0] | masks[1] | masks[2])
    assert torch.equal(
        labels[masks[0]].unique(sorted=True),
        labels.unique(sorted=True),
    )

    writes = data["author", "writes", "paper"].edge_index
    published_in = data["paper", "published_in", "venue"].edge_index
    for edge_index, source_type, target_type in (
        (writes, "author", "paper"),
        (published_in, "paper", "venue"),
    ):
        assert int(edge_index.min()) >= 0
        assert int(edge_index[0].max()) < data[source_type].num_nodes
        assert int(edge_index[1].max()) < data[target_type].num_nodes
        assert edge_index[0].unique().numel() == data[source_type].num_nodes
        assert edge_index[1].unique().numel() == data[target_type].num_nodes

    author_ids, paper_ids = writes
    assert torch.equal(data["author"].x[:, :2].argmax(dim=-1), labels)
    assert torch.equal(
        data["paper"].x[paper_ids, :2].argmax(dim=-1),
        labels[author_ids],
    )


def test_synthetic_heterogeneous_schema_is_native_and_deterministic() -> None:
    """The factory should return the same native schema for a fixed seed."""
    first = make_synthetic_heterogeneous_data(seed=7)
    second = make_synthetic_heterogeneous_data(seed=7)

    assert isinstance(first, HeteroData)
    assert first.node_types == ["author", "paper", "venue"]
    assert set(first.edge_types) == {
        ("author", "writes", "paper"),
        ("paper", "published_in", "venue"),
    }
    assert first["author"].num_nodes == 36
    assert first["paper"].num_nodes == 24
    assert first["venue"].num_nodes == 6
    assert first["author"].x.shape == (36, 8)
    assert first["paper"].x.shape == (24, 5)
    assert "x" not in first["venue"]
    assert first["author"].y.dtype == torch.long
    assert torch.equal(first["author"].y, torch.arange(36) % 2)
    assert torch.equal(
        first["author"].y.bincount(),
        torch.tensor([18, 18]),
    )
    assert torch.equal(first["author"].x, second["author"].x)
    assert torch.equal(first["paper"].x, second["paper"].x)
    for mask_name in ("train_mask", "val_mask", "test_mask"):
        assert torch.equal(
            first["author"][mask_name],
            second["author"][mask_name],
        )
    assert torch.equal(
        first["author", "writes", "paper"].edge_index,
        second["author", "writes", "paper"].edge_index,
    )
    assert torch.equal(
        first["paper", "published_in", "venue"].edge_index,
        second["paper", "published_in", "venue"].edge_index,
    )


def test_synthetic_heterogeneous_supervision_and_signal_contract() -> None:
    """Labels, masks, typed signal, and graph coverage should be valid."""
    data = make_synthetic_heterogeneous_data(seed=11)

    labeled_node_types = [
        node_type for node_type in data.node_types if "y" in data[node_type]
    ]
    assert labeled_node_types == ["author"]

    labels = data["author"].y
    masks = [
        data["author"].train_mask,
        data["author"].val_mask,
        data["author"].test_mask,
    ]
    for mask in masks:
        assert mask.dtype == torch.bool
        assert torch.any(mask)
    assert torch.all(sum(mask.to(torch.int8) for mask in masks) == 1)
    assert torch.equal(
        labels[data["author"].train_mask].unique(sorted=True),
        labels.unique(sorted=True),
    )

    assert torch.equal(data["author"].x[:, :2].argmax(dim=-1), labels)
    writes = data["author", "writes", "paper"].edge_index
    author_ids, paper_ids = writes
    assert torch.equal(
        data["paper"].x[paper_ids, :2].argmax(dim=-1),
        labels[author_ids],
    )

    published_in = data["paper", "published_in", "venue"].edge_index
    assert int(writes[0].min()) >= 0
    assert int(writes[0].max()) < data["author"].num_nodes
    assert int(writes[1].min()) >= 0
    assert int(writes[1].max()) < data["paper"].num_nodes
    assert int(published_in[0].min()) >= 0
    assert int(published_in[0].max()) < data["paper"].num_nodes
    assert int(published_in[1].min()) >= 0
    assert int(published_in[1].max()) < data["venue"].num_nodes
    assert writes[0].unique().numel() == data["author"].num_nodes
    assert writes[1].unique().numel() == data["paper"].num_nodes
    assert published_in[0].unique().numel() == data["paper"].num_nodes
    assert published_in[1].unique().numel() == data["venue"].num_nodes


@pytest.mark.parametrize(
    ("num_authors", "num_papers", "num_venues"),
    [
        (12, 4, 2),
        (14, 4, 3),
        (12, 100, 7),
    ],
)
def test_synthetic_heterogeneous_valid_size_contract(
    num_authors: int,
    num_papers: int,
    num_venues: int,
) -> None:
    """Every documented valid size should preserve graph invariants."""
    data = make_synthetic_heterogeneous_data(
        seed=17,
        num_authors=num_authors,
        num_papers=num_papers,
        num_venues=num_venues,
    )

    _assert_valid_size_contract(data)


def test_synthetic_heterogeneous_factory_preserves_global_rng_state() -> None:
    """Fixture construction should own an isolated random generator."""
    state_before = torch.random.get_rng_state()

    make_synthetic_heterogeneous_data(seed=23)

    assert torch.equal(torch.random.get_rng_state(), state_before)


def test_stratified_masks_use_exact_per_class_proportions() -> None:
    """Each class should independently use 60/20 percent and a remainder."""
    labels = torch.arange(20) % 2

    masks = _stratified_masks(
        labels,
        generator=torch.Generator().manual_seed(7),
    )

    for class_id in labels.unique():
        class_members = labels == class_id
        assert [int(mask[class_members].sum()) for mask in masks] == [6, 2, 2]


def test_stratified_masks_reject_class_without_three_splits() -> None:
    """Every class must have enough nodes to populate all three splits."""
    labels = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)

    with pytest.raises(
        ValueError,
        match="Each synthetic class needs train, validation, and test nodes",
    ):
        _stratified_masks(
            labels,
            generator=torch.Generator().manual_seed(7),
        )


def test_synthetic_heterogeneous_dataset_wraps_canonical_factory() -> None:
    """The dataset wrapper should contain the canonical factory output."""
    expected = make_synthetic_heterogeneous_data(seed=5)
    dataset = SyntheticHeterogeneousDataset(seed=5)

    assert len(dataset) == 1
    actual = dataset[0]
    assert isinstance(actual, HeteroData)
    assert actual.to_dict().keys() == expected.to_dict().keys()
    assert torch.equal(actual["author"].x, expected["author"].x)
    assert torch.equal(actual["author"].y, expected["author"].y)
    for mask_name in ("train_mask", "val_mask", "test_mask"):
        assert torch.equal(
            actual["author"][mask_name],
            expected["author"][mask_name],
        )
    assert torch.equal(actual["paper"].x, expected["paper"].x)
    assert actual["venue"].num_nodes == expected["venue"].num_nodes
    assert torch.equal(
        actual["author", "writes", "paper"].edge_index,
        expected["author", "writes", "paper"].edge_index,
    )
    assert torch.equal(
        actual["paper", "published_in", "venue"].edge_index,
        expected["paper", "published_in", "venue"].edge_index,
    )


@pytest.mark.parametrize(
    "parameters",
    [
        {"num_authors": 10},
        {"num_authors": 13},
        {"num_papers": 25},
        {"num_venues": 1},
        {"num_papers": 4, "num_venues": 6},
    ],
)
def test_synthetic_heterogeneous_factory_rejects_invalid_sizes(
    parameters: dict[str, int],
) -> None:
    """Invalid sizes should fail at the public factory boundary."""
    with pytest.raises(ValueError):
        make_synthetic_heterogeneous_data(**parameters)


def test_synthetic_heterogeneous_exports_work_in_clean_process() -> None:
    """Both public exports should resolve without cached import state."""
    code = """
import pickle
import torch
from topobench.data.datasets import (
    SyntheticHeterogeneousDataset,
    make_synthetic_heterogeneous_data,
)
from topobench.data.datasets.synthetic_heterogeneous_dataset import (
    SyntheticHeterogeneousDataset as DirectDataset,
)
assert SyntheticHeterogeneousDataset is DirectDataset
dataset = SyntheticHeterogeneousDataset(seed=3)
assert isinstance(dataset[0], type(
    make_synthetic_heterogeneous_data(seed=3)
))
restored = pickle.loads(pickle.dumps(dataset))
assert isinstance(restored, DirectDataset)
assert torch.equal(restored[0]["author"].x, dataset[0]["author"].x)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
