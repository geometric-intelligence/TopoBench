import gudhi
import torch_geometric
from toponetx.classes import SimplicialComplex

from topobenchmark.transforms.liftings.pointcloud2simplicial.base import (
    PointCloud2SimplicialLifting,
)


class AlphaComplexLifting(PointCloud2SimplicialLifting):
    r"""Lifts point clouds to simplicial complex domain by generating the alpha complex using the Gudhi library. The alpha complex is a simplicial complex constructed from the finite cells of a Delaunay Triangulation. It has the same persistent homology as the Čech complex and is significantly smaller.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, alpha: float, **kwargs):
        self.alpha = alpha
        super().__init__(**kwargs)

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a point cloud to the alpha complex.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        ac = gudhi.AlphaComplex(data.pos)
        stree = ac.create_simplex_tree()
        stree.prune_above_filtration(self.alpha)
        stree.prune_above_dimension(self.complex_dim)
        sc = SimplicialComplex(
            s for s, filtration_value in stree.get_simplices()
        )
        lifted_topolgy = self._get_lifted_topology(sc)
        lifted_topolgy["x_0"] = data.x
        return lifted_topolgy
