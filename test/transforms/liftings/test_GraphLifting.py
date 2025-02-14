"""Test the GraphLifting class."""

import torch
import torch_geometric
from torch_geometric.data import Data

from topobenchmark.transforms.feature_liftings.projection_sum import (
    ProjectionSum,
)
from topobenchmark.transforms.liftings.base import LiftingMap, LiftingTransform


def _data_has_edge_attr(data: torch_geometric.data.Data) -> bool:
    r"""Check if the input data object has edge attributes.

    Parameters
    ----------
    data : torch_geometric.data.Data
        The input data.

    Returns
    -------
    bool
        Whether the data object has edge attributes.
    """
    return hasattr(data, "edge_attr") and data.edge_attr is not None


class ConcreteGraphLifting(LiftingMap):
    """Concrete implementation of GraphLifting for testing."""

    def lift(self, data):
        """Implement the abstract lift_topology method.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            Empty dictionary for testing purposes.
        """
        return {}


class TestGraphLifting:
    """Test the GraphLifting class."""

    def setup_method(self):
        """Set up test fixtures before each test method.

        Creates an instance of ConcreteGraphLifting with default parameters.
        """
        self.lifting = LiftingTransform(
            ConcreteGraphLifting(), feature_lifting=ProjectionSum()
        )

    def test_data_has_edge_attr(self):
        """Test _data_has_edge_attr method with different data configurations."""

        # Test case 1: Data with edge attributes
        data_with_edge_attr = Data(
            x=torch.tensor([[1.0], [2.0]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_attr=torch.tensor([[1.0], [1.0]]),
        )
        assert _data_has_edge_attr(data_with_edge_attr) is True

        # Test case 2: Data without edge attributes
        data_without_edge_attr = Data(
            x=torch.tensor([[1.0], [2.0]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
        )
        assert _data_has_edge_attr(data_without_edge_attr) is False

        # Test case 3: Data with edge_attr set to None
        data_with_none_edge_attr = Data(
            x=torch.tensor([[1.0], [2.0]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_attr=None,
        )
        assert _data_has_edge_attr(data_with_none_edge_attr) is False

    def test_data_has_edge_attr_empty_data(self):
        """Test _data_has_edge_attr method with empty data object."""
        empty_data = Data()
        assert _data_has_edge_attr(empty_data) is False

    def test_data_has_edge_attr_different_edge_formats(self):
        """Test _data_has_edge_attr method with different edge attribute formats."""

        # Test with float edge attributes
        data_float_attr = Data(
            x=torch.tensor([[1.0], [2.0]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_attr=torch.tensor([[0.5], [0.5]]),
        )
        assert _data_has_edge_attr(data_float_attr) is True

        # Test with integer edge attributes
        data_int_attr = Data(
            x=torch.tensor([[1.0], [2.0]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_attr=torch.tensor([[1], [1]], dtype=torch.long),
        )
        assert _data_has_edge_attr(data_int_attr) is True

        # Test with multi-dimensional edge attributes
        data_multidim_attr = Data(
            x=torch.tensor([[1.0], [2.0]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_attr=torch.tensor([[1.0, 2.0], [2.0, 1.0]]),
        )
        assert _data_has_edge_attr(data_multidim_attr) is True
