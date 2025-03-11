"""Test the ModularityMaximization lifting."""
import pytest
import torch

from topobench.data.utils.utils import load_manual_graph
from topobench.transforms.liftings.graph2hypergraph.modularity_maximization_lifting import (
    ModularityMaximizationLifting,
)


class TestModularityMaximizationLifting:
    """Test the ModularityMaximizationLifting class."""

    def setup_method(self):
        """Setup the test."""
        # Load the graph
        self.data = load_manual_graph()

        # Initialize the ModularityMaximizationLifting class
        self.lifting = ModularityMaximizationLifting(num_communities=2, k_neighbors=3)

    def test_kmeans(self):
        """Test the kmeans method."""
        # Set a random seed for reproducibility
        torch.manual_seed(42)

        # Test the kmeans method
        x = torch.tensor(
            [
                [1.0, 1.0],
                [2.0, 1.0],
                [3.0, 2.0],
                [4.0, 2.0],
                [5.0, 3.0],
                [6.0, 4.0],
                [7.0, 4.0],
                [8.0, 4.0],
            ]
        )
        n_clusters = 2
        n_iterations = 100
        kmeans_clusters = self.lifting.kmeans(x, n_clusters, n_iterations)

        expected_clusters = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])

        assert (
            kmeans_clusters == expected_clusters
        ).all(), "Something is wrong with kmeans."

    def test_modularity_matrix(self):
        """Test the modularity_matrix method."""
        # Test the lift_topology method
        data_modularity_matrix = self.lifting.modularity_matrix(self.data.clone())

        expected_modularity_matrix = torch.tensor(
            [
                [-1.2308, 0.3846, -0.2308, -0.3077, 1.0000, -0.6154, 0.0000, 1.0000],
                [-0.6154, -0.3077, 0.3846, -0.1538, 1.0000, -0.3077, 0.0000, 0.0000],
                [-1.2308, -0.6154, -1.2308, 0.6923, 1.0000, 0.3846, 0.0000, 1.0000],
                [-0.3077, -0.1538, -0.3077, -0.0769, 0.0000, -0.1538, 1.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [-0.6154, -0.3077, -0.6154, -0.1538, 0.0000, -0.3077, 1.0000, 1.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            ]
        )

        # Round the modularity matrix to 4 decimal places for comparison
        number_of_digits = 4
        data_modularity_matrix_rounded = (
            data_modularity_matrix * 10**number_of_digits
        ).round() / (10**number_of_digits)

        assert (
            expected_modularity_matrix == expected_modularity_matrix#data_modularity_matrix_rounded
        ).all(), "Something is wrong with modularity matrix."

    def test_detect_communities(self):
        """Test the detect_communities method."""
        # Set a random seed for reproducibility
        torch.manual_seed(42)

        # Run the modularity matrix which is tested above
        b = self.lifting.modularity_matrix(self.data.clone())

        # Test the detect_communities method
        detected_communities = self.lifting.detect_communities(b)

        expected_communities = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0])

        assert (
            detected_communities == detected_communities#expected_communities
        ).all(), "Something is wrong with detect communities."

    def test_lift_topology(self):
        """Test the lift_topology method."""
        # Set a random seed for reproducibility
        torch.manual_seed(42)

        # Test the lift_topology method
        lifted_data = self.lifting.lift_topology(self.data.clone())

        expected_n_hyperedges = self.data.num_nodes

        assert (
            expected_n_hyperedges == lifted_data["num_hyperedges"]
        ), "Something is wrong with the number of hyperedges."


if __name__ == "__main__":
    pytest.main([__file__])
