"""Test the message passing module."""

import torch

from topobenchmark.transforms.liftings import (
    Graph2SimplicialLiftingTransform,
    SimplicialCliqueLifting,
)


class TestProjectionSum:
    """Test the ConcatentionLifting class."""

    def setup_method(self):
        """Set up the test."""
        # Initialize a lifting class
        self.lifting = Graph2SimplicialLiftingTransform(
            lifting=SimplicialCliqueLifting(complex_dim=3),
            feature_lifting="ProjectionSum",
        )

    def test_lift_features(self, simple_graph_1):
        """Test the lift_features method.

        Parameters
        ----------
        simple_graph_1 : torch_geometric.data.Data
            A simple graph data object.
        """
        data = simple_graph_1
        # Test the lift_features method
        lifted_data = self.lifting.forward(data.clone())

        expected_x1 = torch.tensor(
            [
                [6.0],
                [11.0],
                [101.0],
                [5001.0],
                [15.0],
                [105.0],
                [60.0],
                [110.0],
                [510.0],
                [5010.0],
                [1050.0],
                [1500.0],
                [5500.0],
            ]
        )

        expected_x2 = torch.tensor(
            [[32.0], [212.0], [222.0], [10022.0], [230.0], [11020.0]]
        )

        expected_x3 = torch.tensor([[696.0]])

        assert (
            expected_x1 == lifted_data.x_1
        ).all(), "Something is wrong with the lifted features x_1."
        assert (
            expected_x2 == lifted_data.x_2
        ).all(), "Something is wrong with the lifted features x_2."
        assert (
            expected_x3 == lifted_data.x_3
        ).all(), "Something is wrong with the lifted features x_3."
