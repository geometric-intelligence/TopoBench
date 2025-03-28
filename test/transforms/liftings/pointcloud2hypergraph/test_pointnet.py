"""Test the message passing module."""

import torch
import torch_geometric as pyg

from topobench.transforms.liftings.pointcloud2hypergraph.pointnet_lifting import (
    PointNetLifting,
)


class TestPointNet:
    """Test the PointNetLifting class."""

    def setup_method(self):
        """Setup the test."""
        # Point cloud of equidistantly-spaced grid
        pos = (
            torch.stack(
                torch.meshgrid((torch.arange(4, dtype=torch.float),) * 3, indexing="ij")
            )
            .view(3, -1)
            .T
        )
        self.data = pyg.data.Data(x=torch.rand(pos.size(0)), pos=pos)

        self.lifting = PointNetLifting(sampling_ratio=0.25, cluster_radius=2.5)

    def test_lift_topology(self):
        """Test the lift_topology method."""
        lifted_data = self.lifting(self.data)

        # Expected shape of the incidence matrix
        expected_shape = (
            self.data.num_nodes,
            self.lifting.sampling_ratio * self.data.num_nodes,
        )

        assert (
            lifted_data.incidence_hyperedges.shape == expected_shape
        ), "Incidence matrix not shaped as expected."
