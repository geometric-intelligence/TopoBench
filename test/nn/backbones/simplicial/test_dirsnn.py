"""Unit tests for DirSNN backbone."""

import torch
import pytest
from topobench.nn.backbones.simplicial.dirsnn import DirSNNLayer, DirSNN


class DummyBatch:
    """Mock TopoBench batch using correct framework field names."""
    def __init__(self, x_1, incidence_1, incidence_2):
        self.x_1 = x_1
        self.incidence_1 = incidence_1
        self.incidence_2 = incidence_2


def make_valid_sparse(rows, cols, n_rows, n_cols, values=None):
    """Helper to create a valid sparse COO tensor."""
    idx = torch.tensor([rows, cols], dtype=torch.long)
    val = torch.ones(len(rows)) if values is None else torch.tensor(values, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, (n_rows, n_cols)).coalesce()


class TestDirSNNLayer:
    """Tests for DirSNNLayer."""

    def test_output_shape(self):
        """Test output shape is correct."""
        layer = DirSNNLayer(16, 32)
        x = torch.randn(5, 16)
        L_down = make_valid_sparse([0,1,2], [1,2,3], 5, 5)
        L_up = make_valid_sparse([0,1], [1,2], 5, 5)
        out = layer(x, L_down, L_up)
        assert out.shape == (5, 32)

    def test_no_nans(self):
        """Test output contains no NaNs."""
        layer = DirSNNLayer(8, 16)
        x = torch.randn(4, 8)
        L_down = make_valid_sparse([0,1], [1,2], 4, 4)
        L_up = make_valid_sparse([0,1], [1,0], 4, 4)
        out = layer(x, L_down, L_up)
        assert not torch.isnan(out).any()

    def test_source_sink_no_crash(self):
        """Test that isolated edges (all-zero rows in L_down/L_up) do not crash."""
        layer = DirSNNLayer(8, 16)
        x = torch.randn(4, 8)
        # Empty adjacency — all edges are isolated (worst-case source/sink)
        L_down = torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0),
            (4, 4)
        ).coalesce()
        L_up = torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0),
            (4, 4)
        ).coalesce()
        out = layer(x, L_down, L_up)
        assert out.shape == (4, 16)
        assert not torch.isnan(out).any()


class TestDirSNN:
    """Tests for the full DirSNN backbone."""

    def test_output_shape(self):
        """Test full backbone output shape."""
        backbone = DirSNN(16, 32, 8, num_layers=2)
        B1 = make_valid_sparse([0,1,1,2], [0,0,1,1], 4, 4,
                                values=[-1.,1.,-1.,1.])
        B2 = make_valid_sparse([0,1,2], [0,0,0], 4, 1)
        batch = DummyBatch(torch.randn(4, 16), B1, B2)
        out = backbone(batch)
        assert out.shape == (4, 8)

    def test_sparsification_branch(self):
        """Test _sparsify_boundary fires when max_upper_degree=0."""
        backbone = DirSNN(8, 16, 4, num_layers=2, max_upper_degree=0)
        backbone.train()
        B1 = make_valid_sparse([0,1,1,2], [0,0,1,1], 4, 4,
                                values=[-1.,1.,-1.,1.])
        B2 = make_valid_sparse([0,1,2], [0,0,0], 4, 1)
        batch = DummyBatch(torch.randn(4, 8), B1, B2)
        out = backbone(batch)
        assert out.shape == (4, 4)

    def test_single_layer(self):
        """Test num_layers=1 does not crash."""
        backbone = DirSNN(8, 16, 4, num_layers=1)
        B1 = make_valid_sparse([0,1,1,2], [0,0,1,1], 4, 4,
                                values=[-1.,1.,-1.,1.])
        B2 = make_valid_sparse([0,1,2], [0,0,0], 4, 1)
        batch = DummyBatch(torch.randn(4, 8), B1, B2)
        out = backbone(batch)
        assert out.shape == (4, 4)
