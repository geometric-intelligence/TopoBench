"""Test the message passing module."""

import torch

from topobench.data.utils.utils import load_manual_pointcloud
from topobench.transforms.liftings.pointcloud2hypergraph.voronoi_lifting import (
    VoronoiLifting,
)


class TestVoronoiLifting:
    """Test the SimplicialCliqueLifting class."""

    def setup_method(self):
        """Setup the test."""
        # Load the graph
        self.data = load_manual_pointcloud(pos_to_x=True)

        # Initialise the VoronoiLifting class
        self.lifting = VoronoiLifting(support_ratio=0.26)

    def test_lift_topology(self):
        """Test the lift_topology method."""

        # Test the lift_topology method
        lifted_data = self.lifting.forward(self.data.clone())

        expected_cluster_sizes = torch.tensor([1, 3, 4, 4])
        cluster_sizes = torch.sort(
            torch.unique(
                lifted_data.incidence_hyperedges.coalesce().indices()[1],
                return_counts=True,
            )[1]
        )[0]

        assert (
            expected_cluster_sizes == cluster_sizes
        ).all(), "Something is wrong with edge_index (mismatched cluster sizes)."
