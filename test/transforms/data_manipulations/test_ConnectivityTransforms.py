"""Test connectivity inference transforms."""

import torch
from torch_geometric.data import Data
from topobenchmark.transforms.data_manipulations import (
    InfereKNNConnectivity,
    InfereRadiusConnectivity,
)


class TestConnectivityTransforms:
    """Test connectivity inference transforms."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.x = torch.tensor([
            [2, 2],
            [2.2, 2],
            [2.1, 1.5],
            [-3, 2],
            [-2.7, 2],
            [-2.5, 1.5],
        ])

        self.data = Data(
            x=self.x,
            num_nodes=len(self.x),
        )

        # Initialize transforms
        self.infere_by_knn = InfereKNNConnectivity(args={"k": 3})
        self.infere_by_radius = InfereRadiusConnectivity(args={"r": 1.0})

    def test_infere_knn_connectivity(self):
        """Test inferring connectivity using k-nearest neighbors."""
        data = self.infere_by_knn(self.data.clone())
        assert "edge_index" in data, "No edges in Data object"
        assert data.edge_index.size(0) == 2
        assert data.edge_index.size(1) > 0

    def test_radius_connectivity(self):
        """Test inferring connectivity by radius."""
        data = self.infere_by_radius(self.data.clone())
        assert "edge_index" in data, "No edges in Data object"
        assert data.edge_index.size(0) == 2
        assert data.edge_index.size(1) > 0