"""Test Forman-Ricci Curvature Lifting."""

import pytest
import torch

from topobench.data.utils.utils import load_manual_graph
from topobench.transforms.liftings.graph2hypergraph.forman_ricci_curvature_lifting import (
    HypergraphFormanRicciCurvatureLifting,
)


class TestHypergraphFormanRicciCurvatureLifting:
    """Test the HypergraphFormanRicciCurvatureLifting class."""

    def setup_method(self):
        """Set up the test."""
        self.data = load_manual_graph()

        self.lifting = HypergraphFormanRicciCurvatureLifting(
            network_type="weighted",
            threshold_type="quantile",
            threshold_direction="above",
            threshold=0.6,
        )

    def test_lift_topology(self):
        """Test the lift_topology method."""
        lifted_data = self.lifting.forward(self.data.clone())

        expected_n_hyperedges = 2

        expected_incidence_1 = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        )

        assert (
            expected_incidence_1 == lifted_data.incidence_hyperedges.to_dense()
        ).all(), "Something is wrong with incidence_hyperedges."
        assert (
            expected_n_hyperedges == lifted_data.num_hyperedges
        ), "Something is wrong with the number of hyperedges."

        ###

        self.lifting.network_type = "unweighted"
        lifted_data = self.lifting.forward(self.data.clone())

        expected_incidence_1 = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        )

        assert (
            expected_incidence_1 == lifted_data.incidence_hyperedges.to_dense()
        ).all(), "Something is wrong with incidence_hyperedges."
        assert (
            expected_n_hyperedges == lifted_data.num_hyperedges
        ), "Something is wrong with the number of hyperedges."

        self.lifting.network_type = "unweighted"
        self.lifting.threshold_type = "absolute"
        self.lifting.threshold = -200
        lifted_data = self.lifting.forward(self.data.clone())

        expected_incidence_1 = torch.tensor(
            [
                [1.0, 0, 0, 0, 0, 0, 0, 0],
                [0.0, 1.0, 0, 0, 0, 0, 0, 0],
                [0.0, 0, 1.0, 0, 0, 0, 0, 0],
                [0.0, 0, 0, 1.0, 0, 0, 0, 0],
                [0.0, 0, 0, 0, 1.0, 0, 0, 0],
                [0.0, 0, 0, 0, 0, 1.0, 0, 0],
                [0.0, 0, 0, 0, 0, 0, 1.0, 0],
                [0.0, 0, 0, 0, 0, 0, 0, 1.0],
            ]
        )

        expected_n_hyperedges = 8

        assert (
            expected_incidence_1 == lifted_data.incidence_hyperedges.to_dense()
        ).all(), "Something is wrong with incidence_hyperedges."
        assert (
            expected_n_hyperedges == lifted_data.num_hyperedges
        ), "Something is wrong with the number of hyperedges."

        self.lifting.threshold = -1
        lifted_data = self.lifting.forward(self.data.clone())

        expected_incidence_1 = torch.tensor(
            [
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
            ]
        )

        expected_n_hyperedges = 1

        assert (
            expected_incidence_1 == lifted_data.incidence_hyperedges.to_dense()
        ).all(), "Something is wrong with incidence_hyperedges."
        assert (
            expected_n_hyperedges == lifted_data.num_hyperedges
        ), "Something is wrong with the number of hyperedges."

        self.lifting.threshold = -1
        self.lifting.threshold_direction = "below"
        lifted_data = self.lifting.forward(self.data.clone())

        expected_incidence_1 = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )

        expected_n_hyperedges = 6

        assert (
            expected_incidence_1 == lifted_data.incidence_hyperedges.to_dense()
        ).all(), "Something is wrong with incidence_hyperedges."
        assert (
            expected_n_hyperedges == lifted_data.num_hyperedges
        ), "Something is wrong with the number of hyperedges."

    def test_validations(self):
        """Test the validations of the class."""
        with pytest.raises(NotImplementedError):
            self.lifting.threshold_type = "relative"
            self.lifting.forward(self.data.clone())

        with pytest.raises(NotImplementedError):
            self.lifting.threshold_direction = "middle"
            self.lifting.forward(self.data.clone())
