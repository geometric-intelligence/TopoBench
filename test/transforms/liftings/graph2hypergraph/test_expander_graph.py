"""Test the message passing module."""

from topobench.data.utils.utils import load_manual_graph
from topobench.transforms.liftings.graph2hypergraph.expander_graph_lifting import (
    ExpanderGraphLifting,
)


class TestExpanderGraph:
    """Test the HypergraphKHopLifting class."""

    def setup_method(self):
        """Load the manual graph data and initialize the ExpanderGraphLifting class."""
        self.data = load_manual_graph()

        self.lifting = ExpanderGraphLifting(node_degree=2)

    def test_lift_topology(self):
        """Test the lift_topology method."""
        lifted_data = self.lifting(self.data)

        # Expected number of non-zero entries in the expander graph incidence matrix
        expected_nnz = self.data.num_nodes * self.lifting.node_degree
        observed_nnz = lifted_data.incidence_hyperedges._nnz()
        assert_message_nnz = (
            f"Expected {expected_nnz} non-zero entries but got {observed_nnz}."
        )

        assert observed_nnz == expected_nnz, assert_message_nnz
