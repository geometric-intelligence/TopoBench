"""Test the SimplicialDnDLifting lifting."""

import random

import torch

from topobench.data.utils.utils import load_manual_graph
from topobench.transforms.liftings.graph2simplicial.dnd_lifting import (
    SimplicialDnDLifting,
)


class TestSimplicialDnDLifting:
    """Test the SimplicialDnDLifting class."""

    def setup_method(self):
        """Setup the test."""
        # Load the graph
        self.data = load_manual_graph()

        # Initialise the SimplicialCliqueLifting class
        self.lifting_signed = SimplicialDnDLifting(complex_dim=6, signed=True)
        self.lifting_unsigned = SimplicialDnDLifting(complex_dim=6, signed=False)

        random.seed(42)

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

        expected_incidence_3_singular_values_unsigned = torch.tensor(
            [
                4.4721,
                3.4641,
                3.4641,
                3.4641,
                3.4641,
                3.4641,
                3.4641,
                3.4641,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
            ]
        )

        expected_incidence_3_singular_values_signed = torch.tensor(
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
                8.7545e-07,
                6.7006e-07,
                6.4589e-07,
                6.0707e-07,
                5.1336e-07,
                5.1305e-07,
                4.4262e-07,
                4.2898e-07,
                4.2363e-07,
                3.4737e-07,
                3.3576e-07,
                3.2610e-07,
                3.1345e-07,
                3.0790e-07,
                2.7054e-07,
                2.4754e-07,
                2.4124e-07,
                1.7451e-07,
                1.3131e-07,
                8.0757e-08,
                5.5274e-08,
            ]
        )

        U, S_unsigned, V = torch.svd(lifted_data_unsigned.incidence_3.to_dense())
        U, S_signed, V = torch.svd(lifted_data_signed.incidence_3.to_dense())
        assert torch.allclose(
            expected_incidence_3_singular_values_unsigned, S_unsigned, atol=1.0e-04
        ), "Something is wrong with unsigned incidence_3 (edges to tetrahedrons)."
        assert torch.allclose(
            expected_incidence_3_singular_values_signed, S_signed, atol=1.0e-04
        ), "Something is wrong with signed incidence_3 (edges to tetrahedrons)."

        expected_incidence_4_singular_values_unsigned = torch.tensor(
            [
                4.4721,
                3.4641,
                3.4641,
                3.4641,
                3.4641,
                3.4641,
                3.4641,
                3.4641,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                2.4495,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
                1.4142,
            ]
        )

        expected_incidence_4_singular_values_signed = torch.tensor(
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
                1.3487e-06,
                8.4104e-07,
                7.6933e-07,
                6.2891e-07,
                6.0374e-07,
                5.7482e-07,
                5.0095e-07,
                4.0435e-07,
                3.7420e-07,
                3.7217e-07,
                3.6098e-07,
                3.6064e-07,
                3.3918e-07,
                2.9081e-07,
                2.8067e-07,
                2.7714e-07,
                2.6352e-07,
                2.4713e-07,
                1.7254e-07,
                1.4730e-07,
                1.0029e-07,
            ]
        )

        U, S_unsigned, V = torch.svd(lifted_data_unsigned.incidence_4.to_dense())
        U, S_signed, V = torch.svd(lifted_data_signed.incidence_4.to_dense())
        assert torch.allclose(
            expected_incidence_4_singular_values_unsigned, S_unsigned, atol=1.0e-04
        ), "Something is wrong with unsigned incidence_4."
        assert torch.allclose(
            expected_incidence_4_singular_values_signed, S_signed, atol=1.0e-04
        ), "Something is wrong with signed incidence_4."

        expected_incidence_5_singular_values_unsigned = torch.tensor(
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

        expected_incidence_5_singular_values_signed = torch.tensor(
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
                1.3485e-06,
                5.5655e-07,
                3.6771e-07,
                2.7281e-07,
                2.4915e-07,
                2.0324e-07,
                1.5627e-07,
            ]
        )

        U, S_unsigned, V = torch.svd(lifted_data_unsigned.incidence_5.to_dense())
        U, S_signed, V = torch.svd(lifted_data_signed.incidence_5.to_dense())
        assert torch.allclose(
            expected_incidence_5_singular_values_unsigned, S_unsigned, atol=1.0e-04
        ), "Something is wrong with unsigned incidence_5."
        assert torch.allclose(
            expected_incidence_5_singular_values_signed, S_signed, atol=1.0e-04
        ), "Something is wrong with signed incidence_5."
