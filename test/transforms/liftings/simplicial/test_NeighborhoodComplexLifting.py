""" Tests for the NeighborhoodComplexLifting class."""
import networkx as nx
import torch
from torch_geometric.utils.convert import from_networkx

from topobenchmark.data.utils import load_manual_graph
from topobenchmark.transforms.liftings import (
    Graph2SimplicialLiftingTransform,
    NeighborhoodComplexLifting,
)


class TestNeighborhoodComplexLifting:
    """Test the NeighborhoodComplexLifting class."""

    def setup_method(self):
        """Setup the NeighborhoodComplexLifting tests."""
        self.data = load_manual_graph()

        self.lifting_signed = Graph2SimplicialLiftingTransform(
            NeighborhoodComplexLifting(complex_dim=3),
            signed=True,
            to_undirected=True,
        )
        self.lifting_unsigned = Graph2SimplicialLiftingTransform(
            NeighborhoodComplexLifting(complex_dim=3),
            signed=False,
            to_undirected=True,
        )
        self.lifting_high = Graph2SimplicialLiftingTransform(
            NeighborhoodComplexLifting(
                complex_dim=7,
            ),
            to_undirected=True,
        )

        self.empty_graph = nx.empty_graph(10)
        self.empty_data = from_networkx(self.empty_graph)
        self.empty_data["x"] = torch.rand((10, 10))

        self.star_graph = nx.star_graph(5)
        self.star_data = from_networkx(self.star_graph)
        self.star_data["x"] = torch.rand((6, 1))

        self.random_graph = nx.fast_gnp_random_graph(5, 0.5, seed=0)
        self.random_data = from_networkx(self.random_graph)
        self.random_data["x"] = torch.rand((5, 1))

    def _has_neighbour(
        self, simplex_points: list[set]
    ) -> tuple[bool, set[int]]:
        """Check that the lifting works.
        
        Verify that the maximal simplices
        of Data representation of a simplicial complex
        share a neighbour.
        
        Parameters
        ----------
        simplex_points : list[set]
            List of sets containing the points of the simplices.
            
        Returns
        -------
        bool
            True if the simplices share a neighbour, False otherwise.
        """
        for simplex_point_a in simplex_points:
            for simplex_point_b in simplex_points:
                if simplex_point_a == simplex_point_b:
                    continue
                for node in self.random_graph.nodes:
                    if self.random_graph.has_edge(
                        simplex_point_a, node
                    ) and self.random_graph.has_edge(
                        simplex_point_b, node
                    ):
                        return True
        return False

    def test_lift_topology_random_graph(self):
        """Verify lifting procedure on a random graph."""
        graph = self.lifting_high.data2domain(self.random_data)
        simplicial_complex = self.lifting_high.lifting(graph)

        for simplex_points in simplicial_complex.skeleton(simplicial_complex.dim):
            share_neighbour = self._has_neighbour(simplex_points)
            assert share_neighbour, f"The simplex {simplex_points} does not have a common neighbour with all the nodes."

    def test_lift_topology_star_graph(self):
        """Verify lifting procedure on a small star graph."""
        graph = self.lifting_high.data2domain(self.star_data)
        simplicial_complex = self.lifting_high.lifting(graph)

        for simplex_points in simplicial_complex.skeleton(simplicial_complex.dim):
            share_neighbour = self._has_neighbour(simplex_points)
            assert share_neighbour, f"The simplex {simplex_points} does not have a common neighbour with all the nodes."

    def test_lift_topology_empty_graph(self):
        """Test the lift_topology method with an empty graph."""

        lifted_data_signed = self.lifting_signed.forward(self.empty_data)

        assert (
            lifted_data_signed.incidence_1.shape[1] == 0
        ), "Something is wrong with signed incidence_1 (nodes to edges)."

        assert (
            lifted_data_signed.incidence_2.shape[1] == 0
        ), "Something is wrong with signed incidence_2 (edges to triangles)."

    def test_lift_topology(self):
        """Test the lift_topology method."""

        lifted_data_signed = self.lifting_signed.forward(self.data.clone())
        lifted_data_unsigned = self.lifting_unsigned.forward(self.data.clone())

        expected_incidence_1 = torch.tensor(
            [
                [
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    -1.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    -1.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                ],
            ]
        )
        assert (
            abs(expected_incidence_1)
            == lifted_data_unsigned.incidence_1.to_dense()
        ).all(), (
            "Something is wrong with unsigned incidence_1 (nodes to edges)."
        )
        assert (
            expected_incidence_1 == lifted_data_signed.incidence_1.to_dense()
        ).all(), "Something is wrong with signed incidence_1 (nodes to edges)."

        expected_incidence_2 = torch.tensor(
            [
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [1.0],
                [-1.0],
                [0.0],
                [0.0],
                [0.0],
                [1.0],
            ]
        )

        assert (
            abs(expected_incidence_2)
            == lifted_data_unsigned.incidence_2.to_dense()
        ).all(), "Something is wrong with unsigned incidence_2 (edges to triangles)."
        assert (
            expected_incidence_2 == lifted_data_signed.incidence_2.to_dense()
        ).all(), (
            "Something is wrong with signed incidence_2 (edges to triangles)."
        )
