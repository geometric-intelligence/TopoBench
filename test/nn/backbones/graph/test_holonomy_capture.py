"""Unit tests for the sheaf holonomy capture harness."""

import math

import pytest
import torch

from topobench.nn.backbones.graph.nsd_utils.holonomy_capture import (
    blocks_to_transports,
    sheaf_transports,
)
from topobench.nn.backbones.graph.nsd_utils.sheaf_holonomy import (
    enumerate_triangles,
    triangle_holonomies,
)
from topobench.nn.backbones.graph.nsp import NSPEncoder


def _triangle():
    """A single triangle 0-1-2 as a directed edge_index (pre-symmetrise).

    Returns
    -------
    torch.Tensor
        Edge indices of shape [2, 3].
    """
    return torch.tensor([[0, 1, 2], [1, 2, 0]]).long()


class TestBlocksToTransports:
    """Test the pure block -> transport-dict packaging."""

    def test_matrix_blocks_both_directions(self):
        """Each block yields both orientations (reverse = transpose)."""
        m0 = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        m1 = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        blocks = torch.stack([-m0, -m1])  # builder stores -(F_a^T F_b)
        idx = torch.tensor([[0, 1], [1, 2]])
        transports = blocks_to_transports(blocks, idx)
        assert torch.allclose(transports[(0, 1)], m0)
        assert torch.allclose(transports[(1, 0)], m0.T)
        assert torch.allclose(transports[(1, 2)], m1)
        assert torch.allclose(transports[(2, 1)], m1.T)

    def test_diagonal_blocks_become_diag_matrices(self):
        """1-D (diagonal) blocks are embedded as diagonal matrices."""
        blocks = torch.tensor([[-1.0, -2.0]])  # -(F_a ⊙ F_b) for one edge
        idx = torch.tensor([[0], [1]])
        transports = blocks_to_transports(blocks, idx)
        assert transports[(0, 1)].shape == (2, 2)
        assert torch.allclose(
            transports[(0, 1)], torch.diag(torch.tensor([1.0, 2.0]))
        )
        assert transports[(0, 1)][0, 1] == 0.0


class TestSheafTransportsIntegration:
    """Capture transports from a real forward pass of the sheaf models."""

    def _encoder(self, sheaf_type, seed):
        """Build a small NSP encoder of the given restriction-map family.

        Parameters
        ----------
        sheaf_type : str
            One of ``"diag"``, ``"bundle"``, ``"general"``.
        seed : int
            Manual seed for reproducibility.

        Returns
        -------
        NSPEncoder
            An initialised (not yet run) encoder.
        """
        torch.manual_seed(seed)
        return NSPEncoder(
            input_dim=8,
            hidden_dim=32,
            num_layers=2,
            d=2,
            sheaf_type=sheaf_type,
        )

    def test_bundle_transports_are_orthogonal(self):
        """O(d) bundle blocks recover orthogonal parallel-transport matrices."""
        ei = _triangle()
        enc = self._encoder("bundle", seed=0)
        enc(torch.randn(3, 8), ei)
        transports = sheaf_transports(enc.get_sheaf_propagation_model(), ei)
        eye = torch.eye(2)
        assert len(transports) == 6  # 3 undirected edges, both orientations
        for mat in transports.values():
            assert torch.allclose(
                mat @ mat.transpose(-1, -2), eye, atol=1e-4
            )

    def test_bundle_triangle_holonomy_is_a_finite_rotation(self):
        """Captured bundle transports give a finite rotation-angle holonomy."""
        ei = _triangle()
        enc = self._encoder("bundle", seed=1)
        enc(torch.randn(3, 8), ei)
        transports = sheaf_transports(enc.get_sheaf_propagation_model(), ei)
        tris = enumerate_triangles(ei)
        assert tris == [(0, 1, 2)]
        out = triangle_holonomies(transports, tris)
        assert out["angle"].shape == (1,)
        angle = out["angle"][0].item()
        assert 0.0 <= angle <= math.pi
        assert math.isfinite(out["magnitude"][0].item())

    def test_diag_transports_are_diagonal(self):
        """Diagonal sheaf blocks recover diagonal transport matrices."""
        ei = _triangle()
        enc = self._encoder("diag", seed=2)
        enc(torch.randn(3, 8), ei)
        transports = sheaf_transports(enc.get_sheaf_propagation_model(), ei)
        for mat in transports.values():
            off = mat - torch.diag(torch.diagonal(mat))
            assert torch.allclose(off, torch.zeros_like(off))

    def test_layer_selects_a_learner(self):
        """An explicit layer index reads that learner's stored transports."""
        ei = _triangle()
        enc = self._encoder("bundle", seed=3)
        enc(torch.randn(3, 8), ei)
        prop = enc.get_sheaf_propagation_model()
        first = sheaf_transports(prop, ei, layer=0)
        last = sheaf_transports(prop, ei, layer=-1)
        assert set(first) == set(last)  # same edges, per-layer values


class TestSheafTransportsErrors:
    """Test the guard rails."""

    def test_no_forward_raises(self):
        """Reading transports before any forward pass raises."""
        torch.manual_seed(0)
        enc = NSPEncoder(input_dim=8, hidden_dim=32, d=2, sheaf_type="bundle")
        with pytest.raises(RuntimeError, match="forward"):
            sheaf_transports(enc.get_sheaf_propagation_model(), _triangle())

    def test_wrong_edge_index_raises(self):
        """A mismatched edge_index (different edge count) raises."""
        torch.manual_seed(0)
        ei = _triangle()
        enc = NSPEncoder(
            input_dim=8, hidden_dim=32, num_layers=2, d=2, sheaf_type="bundle"
        )
        enc(torch.randn(3, 8), ei)
        wrong = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]).long()  # 4 edges
        with pytest.raises(RuntimeError, match="does not match"):
            sheaf_transports(enc.get_sheaf_propagation_model(), wrong)
