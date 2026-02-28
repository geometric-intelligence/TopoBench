"""Tests for the NegativeSamplingTransform."""

import pytest
import torch
from torch_geometric.data import Data

from topobench.transforms.data_manipulations.negative_links_sampling import (
    NegativeSamplingTransform,
)


class TestNegativeSamplingTransform:
    """Unit tests for NegativeSamplingTransform."""

    @pytest.fixture
    def toy_data(self):
        """Create a small graph with positive edge labels."""
        # 4 nodes, 3 positive edges
        edge_index = torch.tensor([[0, 1, 2],
                                   [1, 2, 3]])
        num_nodes = 4

        # Use the same edges as positive labels
        edge_label_index = edge_index.clone()

        data = Data(
            edge_index=edge_index,
            edge_label_index=edge_label_index,
            num_nodes=num_nodes,
        )
        return data

    def test_basic_sampling_shapes_and_values(self, toy_data):
        """Check shapes, label values and counts after sampling."""
        transform = NegativeSamplingTransform(neg_pos_ratio=1.0, seed=0)

        out = transform(toy_data)

        # We expect original data untouched
        assert not hasattr(toy_data, "edge_label")

        # New data should have labels and extended edge_label_index
        assert hasattr(out, "edge_label_index")
        assert hasattr(out, "edge_label")

        pos_edges = toy_data.edge_label_index.size(1)
        all_edges = out.edge_label_index.size(1)
        labels = out.edge_label

        # There should be pos + neg edges
        assert all_edges == labels.numel()

        # Labels must be 0 or 1 (float or long)
        unique_vals = set(labels.tolist())
        assert unique_vals <= {0.0, 1.0, 0, 1}

        # At least all positives preserved
        num_pos = int(labels.sum().item())
        assert num_pos == pos_edges

        # At least one negative has been added
        num_neg = all_edges - num_pos
        assert num_neg >= 1

    def test_neg_pos_ratio_respected(self, toy_data):
        """Check that neg_pos_ratio roughly controls negative count."""
        pos_edges = toy_data.edge_label_index.size(1)

        transform = NegativeSamplingTransform(neg_pos_ratio=2.0, seed=0)
        out = transform(toy_data)

        labels = out.edge_label
        num_pos = int(labels.sum().item())
        num_neg = labels.numel() - num_pos

        # Positives should match original
        assert num_pos == pos_edges

        # Negatives should be approx 2 * pos (within 1 due to max(1, int(...)))
        assert num_neg == max(1, int(2.0 * pos_edges))

    def test_original_data_unchanged(self, toy_data):
        """Ensure transform does not modify the original Data in-place."""
        original_edge_label_index = toy_data.edge_label_index.clone()
        transform = NegativeSamplingTransform(neg_pos_ratio=1.0, seed=0)

        _ = transform(toy_data)

        # Original edge_label_index must remain unchanged
        assert torch.equal(toy_data.edge_label_index, original_edge_label_index)
        # And original data should not gain an edge_label attribute
        assert not hasattr(toy_data, "edge_label")

    def test_transform_can_be_called_multiple_times(self, toy_data):
        """Transform can be applied multiple times without errors."""
        transform = NegativeSamplingTransform(neg_pos_ratio=1.0, seed=42)

        out1 = transform(toy_data)
        out2 = transform(toy_data)

        # Both outputs should have valid edge_label_index and edge_label
        for out in (out1, out2):
            assert hasattr(out, "edge_label_index")
            assert hasattr(out, "edge_label")
            assert out.edge_label_index.size(1) == out.edge_label.numel()
            unique_vals = set(out.edge_label.tolist())
            assert unique_vals <= {0.0, 1.0, 0, 1}

    def test_missing_edge_index_raises(self):
        """Missing edge_index should raise AttributeError."""
        data = Data(
            edge_label_index=torch.tensor([[0, 1], [1, 2]]),
            num_nodes=3,
        )
        # edge_index is None by default in PyG when not provided
        transform = NegativeSamplingTransform()

        # We just care that an AttributeError is raised, not the exact message
        with pytest.raises(AttributeError):
            _ = transform(data)
            
    def test_missing_edge_label_index_raises(self):
        """Missing edge_label_index should raise AttributeError."""
        data = Data(
            edge_index=torch.tensor([[0, 1], [1, 2]]),
            num_nodes=3,
        )
        transform = NegativeSamplingTransform()

        with pytest.raises(AttributeError, match="edge_label_index"):
            _ = transform(data)

    def test_no_positive_edges_raises(self):
        """Empty edge_label_index should raise ValueError."""
        data = Data(
            edge_index=torch.tensor([[0, 1], [1, 2]]),
            edge_label_index=torch.empty((2, 0), dtype=torch.long),
            num_nodes=3,
        )
        transform = NegativeSamplingTransform()

        with pytest.raises(ValueError, match="No positive edges"):
            _ = transform(data)