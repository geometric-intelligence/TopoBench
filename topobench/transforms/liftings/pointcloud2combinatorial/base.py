"""Abstract class for lifting pointcloud to combinatorial complexes."""

from topobench.transforms.liftings.liftings import PointCloudLifting


class PointCloud2CombinatorialLifting(PointCloudLifting):
    r"""Abstract class for lifting graphs to combinatorial complexes.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "pointcloud2combinatorial"
