"""Test the message passing module."""

import torch

from topobench.data.utils.utils import load_manual_hypergraph
from topobench.transforms.liftings.hypergraph2combinatorial.universal_strict_lifting import (
    UniversalStrictLifting,
)


class TestUniversalStrictLifting:
    """Test the UniversalStrictLifting class."""

    def setup_method(self):
        """Setup the test."""
        # Load the hypergraph
        self.data = load_manual_hypergraph()

        # Initialise the UniversalStrictLifting class
        self.lifting = UniversalStrictLifting(complex_dim=3)

    def test_lift_topology(self):
        """Test the lift_topology method."""

        # Test the lift_topology method
        lifted_data = self.lifting.forward(self.data.clone())

        expected_incidence_1 = torch.tensor(
            [
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            ]
        )

        assert (
            abs(expected_incidence_1) == lifted_data.incidence_1.to_dense()
        ).all(), "Something is wrong with incidence_1 (nodes to edges)."

        # Notice that all faces contain 3 edges except from the square, which contains 4.
        expected_incidence_2 = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )

        assert (
            abs(expected_incidence_2) == lifted_data.incidence_2.to_dense()
        ).all(), "Something is wrong with incidence_2 (edges to faces)."

        # Notice that the unique 3-cell contains four faces, but it does not contain the square:
        expected_incidence_3 = torch.tensor([[1.0], [1.0], [1.0], [1.0], [0.0]])

        assert (
            abs(expected_incidence_3) == lifted_data.incidence_3.to_dense()
        ).all(), "Something is wrong with incidence_3 (faces to 3-cells)."

        expected_adjacency_0 = torch.tensor(
            [
                [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            ]
        )

        assert (
            abs(expected_adjacency_0) == lifted_data.adjacency_0.to_dense()
        ).all(), "Something is wrong with adjacency_0 (node adjacencies)."

        # Notice that edge with index 6 is the one joining the tetrahedron and the square, and is therefore adjacent to no other edge
        expected_adjacency_1 = torch.tensor(
            [
                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            ]
        )

        assert (
            abs(expected_adjacency_1) == lifted_data.adjacency_1.to_dense()
        ).all(), "Something is wrong with adjacency_1 (edge adjacencies)."

        # Notice that the faces of the tetrahedron are all adjacent to each other, but none of them is adjacent to the square
        expected_adjacency_2 = torch.tensor(
            [
                [0.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        assert (
            abs(expected_adjacency_2) == lifted_data.adjacency_2.to_dense()
        ).all(), "Something is wrong with adjacency_2 (face adjacencies)."

        expected_adjacency_3 = torch.tensor([[0.0]])
        # Notice that the unique 3-cell is adjacent to no other 3-cell.
        assert (
            abs(expected_adjacency_3) == lifted_data.adjacency_3.to_dense()
        ).all(), "Something is wrong with adjacency_3 (3-cell adjacencies)"
