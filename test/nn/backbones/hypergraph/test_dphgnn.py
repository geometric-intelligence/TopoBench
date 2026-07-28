"""Unit tests for DPHGNN."""

import pytest
import torch

from topobench.nn.backbones.hypergraph import dphgnn
from topobench.nn.backbones.hypergraph.dphgnn import DPHGNN


def _toy_incidence():
    """Build a toy incidence matrix.

    n=6 hypernodes, m=4 hyperedges: e0={0,1,2}, e1={2,3}, e2={4}
    (singleton), e3={} (empty column), node 5 is isolated.
    """
    rows = [0, 1, 2, 2, 3, 4]
    cols = [0, 0, 0, 1, 1, 2]
    values = torch.ones(len(rows))
    indices = torch.tensor([rows, cols], dtype=torch.long)
    return torch.sparse_coo_tensor(indices, values, (6, 4)).coalesce()


def _toy_features(seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(6, 8, generator=generator, dtype=torch.float64)


def _permute_incidence(incidence, perm):
    dense = incidence.to_dense()
    return dense[perm].to_sparse().coalesce()


class TestDPHGNN:
    """Unit tests for DPHGNN."""

    def setup_method(self):
        """Set up a toy hypergraph and a default model instance."""
        self.incidence = _toy_incidence()
        self.x = _toy_features()
        self.model = DPHGNN(hidden_channels=8).double()

    def test_output_shapes(self):
        """Output shapes must match the wrapper contract."""
        x_0, x_1 = self.model(self.x, self.incidence)
        assert x_0.shape == (6, 8)
        assert x_1.shape == (4, 8)

    def test_permutation_equivariance(self):
        """model(PX, PH) == P . model(X, H) on x_0 (Proposition 4.1)."""
        self.model.eval()
        perm = torch.tensor([3, 1, 4, 0, 5, 2])
        x_perm = self.x[perm]
        h_perm = _permute_incidence(self.incidence, perm)

        with torch.no_grad():
            x_0, _ = self.model(self.x, self.incidence)
            x_0_perm, _ = self.model(x_perm, h_perm)

        assert torch.allclose(x_0_perm, x_0[perm], atol=1e-5)

    def test_block_diagonal(self):
        """Two disjoint hypergraphs give the concatenation of outputs."""
        self.model.eval()
        incidence_2 = _toy_incidence()
        x_2 = _toy_features(seed=1)

        with torch.no_grad():
            x_0_a, x_1_a = self.model(self.x, self.incidence)
            x_0_b, x_1_b = self.model(x_2, incidence_2)

        n, m = 6, 4
        combined_x = torch.cat([self.x, x_2], dim=0)
        dense_a = self.incidence.to_dense()
        dense_b = incidence_2.to_dense()
        top = torch.cat(
            [dense_a, torch.zeros(n, m, dtype=torch.float64)], dim=1
        )
        bottom = torch.cat(
            [torch.zeros(n, m, dtype=torch.float64), dense_b], dim=1
        )
        combined_incidence = (
            torch.cat([top, bottom], dim=0).to_sparse().coalesce()
        )

        with torch.no_grad():
            x_0_combined, x_1_combined = self.model(
                combined_x, combined_incidence
            )

        assert torch.allclose(
            x_0_combined, torch.cat([x_0_a, x_0_b], dim=0), atol=1e-5
        )
        assert torch.allclose(
            x_1_combined, torch.cat([x_1_a, x_1_b], dim=0), atol=1e-5
        )

    def test_degenerate_cases_no_nan(self):
        """Isolated node, singleton and empty hyperedges must not yield nan."""
        x_0, x_1 = self.model(self.x, self.incidence)
        assert not torch.isnan(x_0).any()
        assert not torch.isinf(x_0).any()
        assert not torch.isnan(x_1).any()
        assert not torch.isinf(x_1).any()

    def test_gradient_flow(self):
        """Every parameter must receive a gradient after backward."""
        x_0, x_1 = self.model(self.x, self.incidence)
        loss = x_0.sum() + x_1.sum()
        loss.backward()
        for name, param in self.model.named_parameters():
            assert param.grad is not None, f"no grad for {name}"

    def test_determinism_in_eval(self):
        """Two eval-mode forwards with no randomness give the same output."""
        self.model.eval()
        with torch.no_grad():
            x_0_first, _ = self.model(self.x, self.incidence)
            x_0_second, _ = self.model(self.x, self.incidence)
        assert torch.equal(x_0_first, x_0_second)

    def test_structures_by_hand(self):
        """Derived structures must match hand-computed values."""
        incidence = self.incidence
        d_v, d_e = dphgnn._compute_degrees(incidence)
        assert torch.equal(d_v, torch.tensor([1.0, 1.0, 2.0, 1.0, 1.0, 0.0]))
        assert torch.equal(d_e, torch.tensor([3.0, 2.0, 1.0, 0.0]))

        a_c = dphgnn._clique_adjacency(incidence, 6).to_dense()
        expected_a_c = torch.zeros(6, 6)
        for i, j in [(0, 1), (0, 2), (1, 2), (2, 3)]:
            expected_a_c[i, j] = 1.0
            expected_a_c[j, i] = 1.0
        assert torch.equal(a_c, expected_a_c)

    def test_no_label_dependency(self):
        """forward only takes (x_0, incidence): no access to labels."""
        import inspect

        params = inspect.signature(DPHGNN.forward).parameters
        assert list(params.keys()) == ["self", "x_0", "incidence_hyperedges"]

    @pytest.mark.parametrize("supernode_init", ["zeros", "mean"])
    @pytest.mark.parametrize("with_mediators", [False, True])
    @pytest.mark.parametrize("taa_neighborhood", ["clique", "self"])
    def test_config_options(
        self, supernode_init, with_mediators, taa_neighborhood
    ):
        """Every config option must produce a valid forward pass."""
        model = DPHGNN(
            hidden_channels=8,
            supernode_init=supernode_init,
            with_mediators=with_mediators,
            taa_neighborhood=taa_neighborhood,
        ).double()
        x_0, x_1 = model(self.x, self.incidence)
        assert x_0.shape == (6, 8)
        assert x_1.shape == (4, 8)

    def test_invalid_supernode_init_raises(self):
        """An unknown supernode_init must raise ValueError."""
        with pytest.raises(ValueError):
            DPHGNN(hidden_channels=8, supernode_init="invalid")

    def test_invalid_taa_neighborhood_raises(self):
        """An unknown taa_neighborhood must raise ValueError at forward."""
        model = DPHGNN(hidden_channels=8, taa_neighborhood="invalid").double()
        with pytest.raises(ValueError):
            model(self.x, self.incidence)

    def test_heads_must_divide_hidden_channels(self):
        """A head count not dividing hidden_channels must raise."""
        with pytest.raises(ValueError):
            DPHGNN(hidden_channels=8, taa_heads=3)

    def test_repr(self):
        """__repr__ must return a non-empty string."""
        assert "DPHGNN" in repr(self.model)
        for name, module in self.model.named_children():
            assert module.__class__.__name__ in repr(module), name

    def test_hypergcn_adjacency_all_singletons(self):
        """No hyperedge with cardinality >= 2 yields an empty adjacency."""
        rows = [0, 1, 2]
        cols = [0, 1, 2]
        values = torch.ones(3, dtype=torch.float64)
        incidence = torch.sparse_coo_tensor(
            torch.tensor([rows, cols]), values, (3, 3)
        ).coalesce()
        x = torch.randn(3, 8, dtype=torch.float64)
        adjacency = dphgnn._hypergcn_adjacency(incidence, x, 3, False)
        assert adjacency._nnz() == 0
        assert adjacency.shape == (3, 3)
