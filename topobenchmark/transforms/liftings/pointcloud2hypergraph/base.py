from modules.transforms.liftings.lifting import PointCloudLifting


class PointCloud2HypergraphLifting(PointCloudLifting):
    r"""Abstract class for lifting pointclouds to hypergraphs.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "pointcloud2hypergraph"
