"""Test the SimplicialEccentricityLifting lifting."""

import torch

from topobench.data.utils.utils import load_manual_graph
from topobench.transforms.liftings.graph2simplicial.eccentricity_lifting import (
    SimplicialEccentricityLifting,
)


class TestSimplicialEccentricityLifting:
    """Test the SimplicialEccentricityLifting class."""

    def setup_method(self):
        """Setup the test."""
        # Load the graph
        self.data = load_manual_graph()

        # Initialise the SimplicialCliqueLifting class
        self.lifting_signed = SimplicialEccentricityLifting(complex_dim=3, signed=True)
        self.lifting_unsigned = SimplicialEccentricityLifting(
            complex_dim=3, signed=False
        )

    def test_lift_topology(self):
        """Test the lift_topology method."""

        # Test the lift_topology method
        lifted_data_signed = self.lifting_signed.forward(self.data.clone())
        lifted_data_unsigned = self.lifting_unsigned.forward(self.data.clone())

        expected_incidence_1_singular_values_unsigned = torch.tensor(
            [3.7417, 2.4495, 2.4495, 2.4495, 2.4495, 2.4495, 2.4495, 2.4495]
        )

        expected_incidence_1_singular_values_signed = torch.tensor(
            [
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                6.8993e-08,
            ]
        )

        U, S_unsigned, V = torch.svd(lifted_data_unsigned.incidence_1.to_dense())
        U, S_signed, V = torch.svd(lifted_data_signed.incidence_1.to_dense())

        assert torch.allclose(
            expected_incidence_1_singular_values_unsigned, S_unsigned, atol=1.0e-04
        ), "Something is wrong with unsigned incidence_1 (nodes to edges)."
        assert torch.allclose(
            expected_incidence_1_singular_values_signed, S_signed, atol=1.0e-04
        ), "Something is wrong with signed incidence_1 (nodes to edges)."

        expected_incidence_2_singular_values_unsigned = torch.tensor(
            [
                4.2426,
                3.1623,
                3.1623,
                3.1623,
                3.1623,
                3.1623,
                3.1623,
                3.1623,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
                2.0000,
            ]
        )

        expected_incidence_2_singular_values_signed = torch.tensor(
            [
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                2.8284e00,
                7.0866e-07,
                4.0955e-07,
                3.2154e-07,
                2.9976e-07,
                2.8069e-07,
                2.3097e-07,
                9.4821e-08,
            ]
        )

        U, S_unsigned, V = torch.svd(lifted_data_unsigned.incidence_2.to_dense())
        U, S_signed, V = torch.svd(lifted_data_signed.incidence_2.to_dense())
        assert torch.allclose(
            expected_incidence_2_singular_values_unsigned, S_unsigned, atol=1.0e-04
        ), "Something is wrong with unsigned incidence_2 (edges to triangles)."
        assert torch.allclose(
            expected_incidence_2_singular_values_signed, S_signed, atol=1.0e-04
        ), "Something is wrong with signed incidence_2 (edges to triangles)."
