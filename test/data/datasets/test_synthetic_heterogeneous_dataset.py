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
    assert first["author"].x.shape[1] != first["paper"].x.shape[1]
    assert "x" not in first["venue"]
    assert torch.equal(first["author"].x, second["author"].x)
    assert torch.equal(
        first["author", "writes", "paper"].edge_index,
        second["author", "writes", "paper"].edge_index,
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


def test_synthetic_heterogeneous_factory_preserves_global_rng_state() -> None:
    """Fixture construction should own an isolated random generator."""
    state_before = torch.random.get_rng_state()

    make_synthetic_heterogeneous_data(seed=23)

    assert torch.equal(torch.random.get_rng_state(), state_before)


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
    assert torch.equal(
        actual["author", "writes", "paper"].edge_index,
        expected["author", "writes", "paper"].edge_index,
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
from topobench.data.datasets import (
    SyntheticHeterogeneousDataset,
    make_synthetic_heterogeneous_data,
)
from topobench.data.datasets.synthetic_heterogeneous_dataset import (
    SyntheticHeterogeneousDataset as DirectDataset,
)
assert SyntheticHeterogeneousDataset is DirectDataset
assert isinstance(SyntheticHeterogeneousDataset(seed=3)[0], type(
    make_synthetic_heterogeneous_data(seed=3)
))
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
