import torch

from modules.data.utils.utils import load_point_cloud
from modules.transforms.liftings.pointcloud2simplicial.delaunay_lifting import (
    DelaunayLifting,
)


class TestDelaunayLifting:
    """Test the DelaunayLifting class."""

    def setup_method(self):
        # Load the point cloud
        SEED = 42
        self.data = load_point_cloud(num_samples=5, seed=SEED)

        # Initialise the DelaunayLifting class
        self.lifting_signed = DelaunayLifting(signed=True)
        self.lifting_unsigned = DelaunayLifting(signed=False)

    def test_lift_topology(self):
        """Test the lift_topology method."""

        # Test the lift_topology method
        lifted_data_signed = self.lifting_signed.forward(self.data.clone())
        lifted_data_unsigned = self.lifting_unsigned.forward(self.data.clone())

        expected_incidence_1 = torch.tensor(
            [
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            ]
        )

        assert (
            abs(expected_incidence_1) == lifted_data_unsigned.incidence_1.to_dense()
        ).all(), "Something is wrong with unsigned incidence_1 (nodes to edges)."
        assert (
            expected_incidence_1 == lifted_data_signed.incidence_1.to_dense()
        ).all(), "Something is wrong with signed incidence_1 (nodes to edges)."

        expected_incidence_2 = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        )

        assert (
            abs(expected_incidence_2) == lifted_data_unsigned.incidence_2.to_dense()
        ).all(), "Something is wrong with unsigned incidence_2 (edges to triangles)."
        assert (
            expected_incidence_2 == lifted_data_signed.incidence_2.to_dense()
        ).all(), "Something is wrong with signed incidence_2 (edges to triangles)."
