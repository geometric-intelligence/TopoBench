"""Tests for dependency-free hypergraph and download utilities."""

import pytest
import torch

from topobench.data.utils.hypergraph_io import incidence_pairs


def test_incidence_pairs_deterministically_remaps_mixed_sparse_ids():
    """Mixed raw IDs map to canonical contiguous hyperedge IDs."""
    index, count = incidence_pairs(
        {
            "edge-z": [2, 0, 2],
            100: [3, 1],
            "edge-a": [1],
        },
        num_nodes=4,
    )

    assert count == 3
    assert torch.equal(
        index,
        torch.tensor(
            [[0, 1, 1, 2, 3], [2, 0, 1, 2, 0]],
            dtype=torch.long,
        ),
    )


def test_incidence_pairs_returns_explicit_empty_shape():
    """An empty mapping has an unambiguous dense incidence shape."""
    index, count = incidence_pairs({}, num_nodes=3)

    assert count == 0
    assert index.dtype == torch.long
    assert index.shape == (2, 0)


@pytest.mark.parametrize("node", [-1, 3])
def test_incidence_pairs_rejects_out_of_bounds_nodes(node):
    """Node indices must fall inside the declared feature-row range."""
    with pytest.raises(ValueError, match="out-of-bounds node"):
        incidence_pairs({"edge": [node]}, num_nodes=3)


@pytest.mark.parametrize("node", [True, 1.5, "1"])
def test_incidence_pairs_rejects_non_integer_nodes(node):
    """Integer-looking values are not silently coerced into node IDs."""
    with pytest.raises(TypeError, match="node IDs must be integers"):
        incidence_pairs({"edge": [node]}, num_nodes=3)


def test_incidence_pairs_rejects_declared_empty_hyperedges():
    """Native v2 incidence cannot encode a declared empty hyperedge."""
    with pytest.raises(ValueError, match="empty hyperedge"):
        incidence_pairs({"empty": []}, num_nodes=3)
